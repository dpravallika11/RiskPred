import pytest

from app.investigation.schemas import (
    InvestigationContext,
    MLPredictionEvidence,
    SHAPEvidence,
    GraphEvidence,
    NetworkRiskEvidence,
    ClusterEvidence,
    RiskAgentResult,
)
from app.investigation.risk_agent import RiskAgent


@pytest.fixture
def agent():
    return RiskAgent()


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_accepts_valid_context(self, agent):
        ctx = InvestigationContext(transaction_id="txn-001")
        result = agent.analyze(ctx)
        assert isinstance(result, RiskAgentResult)

    def test_returns_expected_fields(self, agent):
        ctx = InvestigationContext(transaction_id="txn-002")
        result = agent.analyze(ctx)
        assert result.transaction_id == "txn-002"
        assert isinstance(result.risk_level, str)
        assert isinstance(result.risk_score, (int, float))
        assert isinstance(result.assessment, str)
        assert isinstance(result.reasons, list)
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.risk_reducers, list)
        assert isinstance(result.evidence_summary, dict)

    def test_output_is_deterministic(self, agent):
        ctx = InvestigationContext(
            transaction_id="txn-003",
            ml_prediction=MLPredictionEvidence(
                fraud_probability=0.85,
                risk_score=85,
                risk_level="HIGH",
                recommended_action="MANUAL_REVIEW",
            ),
        )
        r1 = agent.analyze(ctx)
        r2 = agent.analyze(ctx)
        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# ML evidence
# ---------------------------------------------------------------------------


class TestMLEvidence:
    def test_high_ml_risk(self, agent):
        ctx = InvestigationContext(
            transaction_id="txn-ml-high",
            ml_prediction=MLPredictionEvidence(
                fraud_probability=0.92,
                risk_score=92,
                risk_level="HIGH",
                recommended_action="MANUAL_REVIEW",
            ),
        )
        result = agent.analyze(ctx)
        assert result.risk_level == "HIGH"
        assert result.risk_score == 92.0
        assert any("High ML risk score" in r for r in result.reasons)
        assert "MANUAL_REVIEW" in result.assessment

    def test_medium_ml_risk(self, agent):
        ctx = InvestigationContext(
            transaction_id="txn-ml-med",
            ml_prediction=MLPredictionEvidence(
                fraud_probability=0.45,
                risk_score=45,
                risk_level="MEDIUM",
                recommended_action="VERIFY",
            ),
        )
        result = agent.analyze(ctx)
        assert result.risk_level == "MEDIUM"
        assert result.risk_score == 45.0
        assert any("Moderate ML risk score" in r for r in result.reasons)

    def test_low_ml_risk(self, agent):
        ctx = InvestigationContext(
            transaction_id="txn-ml-low",
            ml_prediction=MLPredictionEvidence(
                fraud_probability=0.10,
                risk_score=10,
                risk_level="LOW",
                recommended_action="ALLOW",
            ),
        )
        result = agent.analyze(ctx)
        assert result.risk_level == "LOW"
        assert result.risk_score == 10.0
        assert any("Low ML risk score" in r for r in result.reasons)

    def test_ml_score_not_modified(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.73,
            risk_score=73,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(transaction_id="txn-integrity", ml_prediction=ml)
        result = agent.analyze(ctx)
        assert result.risk_score == 73.0

    def test_ml_level_propagated(self, agent):
        for level in ("LOW", "MEDIUM", "HIGH"):
            ml = MLPredictionEvidence(
                fraud_probability=0.5,
                risk_score=50,
                risk_level=level,
                recommended_action="VERIFY",
            )
            ctx = InvestigationContext(transaction_id=f"txn-{level}", ml_prediction=ml)
            result = agent.analyze(ctx)
            assert result.risk_level == level

    def test_ml_evidence_in_summary(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.60,
            risk_score=60,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(transaction_id="txn-summary", ml_prediction=ml)
        result = agent.analyze(ctx)
        assert result.evidence_summary["ml_available"] is True
        assert result.evidence_summary["ml_risk_score"] == 60
        assert result.evidence_summary["fraud_probability"] == 0.60
        assert result.evidence_summary["recommended_action"] == "VERIFY"


# ---------------------------------------------------------------------------
# SHAP evidence
# ---------------------------------------------------------------------------


class TestSHAPEvidence:
    def test_risk_factors_appear_in_result(self, agent):
        shap = SHAPEvidence(
            risk_factors=[
                {"feature": "amount", "impact": 0.3},
                {"feature": "velocity_5m", "impact": 0.2},
            ],
            risk_reducers=[],
        )
        ctx = InvestigationContext(
            transaction_id="txn-shap-factors",
            shap_explanation=shap,
        )
        result = agent.analyze(ctx)
        assert len(result.risk_factors) == 2
        assert result.risk_factors[0]["feature"] == "amount"

    def test_risk_reducers_appear_in_result(self, agent):
        shap = SHAPEvidence(
            risk_factors=[],
            risk_reducers=[
                {"feature": "avg_transaction_amount", "impact": -0.15},
            ],
        )
        ctx = InvestigationContext(
            transaction_id="txn-shap-reducers",
            shap_explanation=shap,
        )
        result = agent.analyze(ctx)
        assert len(result.risk_reducers) == 1
        assert result.risk_reducers[0]["feature"] == "avg_transaction_amount"

    def test_missing_shap_no_fake_factors(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-shap")
        result = agent.analyze(ctx)
        assert result.risk_factors == []
        assert result.risk_reducers == []
        assert result.evidence_summary["shap_available"] is False

    def test_shap_factors_in_reasons(self, agent):
        shap = SHAPEvidence(
            risk_factors=[
                {"feature": "amount", "impact": 0.4},
                {"feature": "is_new_device", "impact": 0.25},
            ],
            risk_reducers=[],
        )
        ctx = InvestigationContext(
            transaction_id="txn-shap-reason",
            shap_explanation=shap,
        )
        result = agent.analyze(ctx)
        assert any("risk factor(s)" in r for r in result.reasons)

    def test_shap_reducers_in_reasons(self, agent):
        shap = SHAPEvidence(
            risk_factors=[],
            risk_reducers=[
                {"feature": "avg_amount", "impact": -0.1},
            ],
        )
        ctx = InvestigationContext(
            transaction_id="txn-shap-reducer-reason",
            shap_explanation=shap,
        )
        result = agent.analyze(ctx)
        assert any("risk reducer(s)" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Network evidence
# ---------------------------------------------------------------------------


class TestNetworkEvidence:
    def test_network_risk_incorporated(self, agent):
        network = NetworkRiskEvidence(
            network_risk_score=75.0,
            network_risk_level="HIGH",
            combined_risk_score=82.0,
            combined_risk_level="HIGH",
            factors=[],
            neighbor_count=5,
            suspicious_neighbor_count=3,
        )
        ctx = InvestigationContext(
            transaction_id="txn-net",
            network_risk=network,
        )
        result = agent.analyze(ctx)
        assert result.risk_score == 82.0
        assert result.risk_level == "HIGH"

    def test_suspicious_neighbors_in_reasons(self, agent):
        network = NetworkRiskEvidence(
            network_risk_score=60.0,
            network_risk_level="MEDIUM",
            combined_risk_score=65.0,
            combined_risk_level="MEDIUM",
            factors=[],
            neighbor_count=4,
            suspicious_neighbor_count=2,
        )
        graph = GraphEvidence(
            graph_available=True,
            total_connections=4,
            suspicious_neighbor_count=2,
        )
        ctx = InvestigationContext(
            transaction_id="txn-susp-neighbors",
            graph=graph,
            network_risk=network,
        )
        result = agent.analyze(ctx)
        assert any("suspicious neighboring" in r.lower() for r in result.reasons)

    def test_network_risk_level_in_reasons(self, agent):
        network = NetworkRiskEvidence(
            network_risk_score=80.0,
            network_risk_level="HIGH",
            combined_risk_score=85.0,
            combined_risk_level="HIGH",
            factors=[],
            neighbor_count=6,
            suspicious_neighbor_count=4,
        )
        graph = GraphEvidence(graph_available=True, total_connections=6)
        ctx = InvestigationContext(
            transaction_id="txn-net-level",
            graph=graph,
            network_risk=network,
        )
        result = agent.analyze(ctx)
        assert any("Network risk level is HIGH" in r for r in result.reasons)

    def test_missing_graph_no_false_network_claims(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-graph")
        result = agent.analyze(ctx)
        assert result.evidence_summary["graph_available"] is False
        assert any("Graph/network evidence is unavailable" in r for r in result.reasons)

    def test_graph_present_but_network_none(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            suspicious_neighbor_count=1,
        )
        ctx = InvestigationContext(
            transaction_id="txn-graph-no-net",
            graph=graph,
            network_risk=None,
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["network_risk_available"] is False
        assert any("Network risk calculation is unavailable" in r for r in result.reasons)

    def test_network_evidence_in_summary(self, agent):
        network = NetworkRiskEvidence(
            network_risk_score=55.0,
            network_risk_level="MEDIUM",
            combined_risk_score=62.0,
            combined_risk_level="MEDIUM",
            factors=[],
            neighbor_count=3,
            suspicious_neighbor_count=1,
        )
        ctx = InvestigationContext(
            transaction_id="txn-net-summary",
            network_risk=network,
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["network_risk_available"] is True
        assert result.evidence_summary["network_risk_score"] == 55.0
        assert result.evidence_summary["combined_risk_score"] == 62.0


# ---------------------------------------------------------------------------
# Cluster evidence
# ---------------------------------------------------------------------------


class TestClusterEvidence:
    def test_high_risk_cluster(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2", "t3", "t4"],
            total_transactions=4,
            entity_count=3,
            entity_types=["device", "card"],
            suspicious_transaction_count=3,
            suspicious_ratio=0.75,
            shared_identifiers={"device": ["dev-1"]},
            risk_level="HIGH",
            avg_risk_score=78.0,
            strong_entity_types=["device", "card"],
            weak_entity_types=[],
        )
        graph = GraphEvidence(graph_available=True, total_connections=4)
        ctx = InvestigationContext(
            transaction_id="txn-cluster-high",
            graph=graph,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert any("HIGH-risk fraud cluster" in r for r in result.reasons)
        assert result.evidence_summary["cluster_available"] is True
        assert result.evidence_summary["cluster_risk_level"] == "HIGH"

    def test_medium_risk_cluster(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2", "t3"],
            total_transactions=3,
            entity_count=2,
            entity_types=["card"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.33,
            shared_identifiers={"card": ["card-1"]},
            risk_level="MEDIUM",
            avg_risk_score=45.0,
            strong_entity_types=[],
            weak_entity_types=["card"],
        )
        graph = GraphEvidence(graph_available=True, total_connections=3)
        ctx = InvestigationContext(
            transaction_id="txn-cluster-med",
            graph=graph,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert any("MEDIUM-risk cluster" in r for r in result.reasons)

    def test_cluster_not_found(self, agent):
        cluster = ClusterEvidence(found=False)
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-cluster-notfound",
            graph=graph,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["cluster_available"] is False

    def test_missing_cluster_evidence(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=1)
        ctx = InvestigationContext(
            transaction_id="txn-cluster-missing",
            graph=graph,
            cluster=None,
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["cluster_available"] is False
        assert any("Cluster evidence is unavailable" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# No hallucination
# ---------------------------------------------------------------------------


class TestNoHallucination:
    def test_empty_context_no_fake_neighbors(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty")
        result = agent.analyze(ctx)
        assert "suspicious_neighbor_count" not in result.evidence_summary
        assert "total_connections" not in result.evidence_summary

    def test_empty_context_no_fake_shap(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty-shap")
        result = agent.analyze(ctx)
        assert result.risk_factors == []
        assert result.risk_reducers == []
        assert "risk_factor_count" not in result.evidence_summary

    def test_empty_context_no_fake_cluster(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty-cluster")
        result = agent.analyze(ctx)
        assert "cluster_total_transactions" not in result.evidence_summary
        assert "cluster_risk_level" not in result.evidence_summary

    def test_empty_context_no_fake_scores(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-scores")
        result = agent.analyze(ctx)
        assert "ml_risk_score" not in result.evidence_summary
        assert "fraud_probability" not in result.evidence_summary

    def test_empty_context_no_fake_entities(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-entities")
        result = agent.analyze(ctx)
        assert "entity_count" not in result.evidence_summary


# ---------------------------------------------------------------------------
# Score integrity
# ---------------------------------------------------------------------------


class TestScoreIntegrity:
    def test_combined_score_used_when_available(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        network = NetworkRiskEvidence(
            network_risk_score=40.0,
            network_risk_level="MEDIUM",
            combined_risk_score=47.0,
            combined_risk_level="MEDIUM",
            factors=[],
            neighbor_count=2,
            suspicious_neighbor_count=0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-combined",
            ml_prediction=ml,
            network_risk=network,
        )
        result = agent.analyze(ctx)
        assert result.risk_score == 47.0

    def test_ml_score_fallback_when_no_network(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.80,
            risk_score=80,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(
            transaction_id="txn-fallback",
            ml_prediction=ml,
        )
        result = agent.analyze(ctx)
        assert result.risk_score == 80.0

    def test_zero_score_when_no_evidence(self, agent):
        ctx = InvestigationContext(transaction_id="txn-zero")
        result = agent.analyze(ctx)
        assert result.risk_score == 0.0
        assert result.risk_level == "UNKNOWN"

    def test_network_score_does_not_mutate_ml(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.60,
            risk_score=60,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        network = NetworkRiskEvidence(
            network_risk_score=90.0,
            network_risk_level="HIGH",
            combined_risk_score=66.0,
            combined_risk_level="MEDIUM",
            factors=[],
            neighbor_count=8,
            suspicious_neighbor_count=5,
        )
        ctx = InvestigationContext(
            transaction_id="txn-no-mutate",
            ml_prediction=ml,
            network_risk=network,
        )
        _ = agent.analyze(ctx)
        assert ml.risk_score == 60
        assert ml.risk_level == "MEDIUM"


# ---------------------------------------------------------------------------
# Assessment text
# ---------------------------------------------------------------------------


class TestAssessment:
    def test_unknown_when_no_evidence(self, agent):
        ctx = InvestigationContext(transaction_id="txn-assess-unknown")
        result = agent.analyze(ctx)
        assert "Insufficient evidence" in result.assessment

    def test_high_assessment_text(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.90,
            risk_score=90,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(
            transaction_id="txn-assess-high",
            ml_prediction=ml,
        )
        result = agent.analyze(ctx)
        assert "HIGH risk" in result.assessment
        assert "MANUAL_REVIEW" in result.assessment

    def test_low_assessment_text(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.05,
            risk_score=5,
            risk_level="LOW",
            recommended_action="ALLOW",
        )
        ctx = InvestigationContext(
            transaction_id="txn-assess-low",
            ml_prediction=ml,
        )
        result = agent.analyze(ctx)
        assert "LOW risk" in result.assessment

    def test_cluster_in_assessment(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            risk_level="HIGH",
        )
        graph = GraphEvidence(graph_available=True)
        ctx = InvestigationContext(
            transaction_id="txn-assess-cluster",
            graph=graph,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert "cluster of 2" in result.assessment


# ---------------------------------------------------------------------------
# Missing evidence handling
# ---------------------------------------------------------------------------


class TestMissingEvidence:
    def test_no_ml_prediction(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-ml")
        result = agent.analyze(ctx)
        assert result.evidence_summary["ml_available"] is False
        assert any("ML prediction evidence is unavailable" in r for r in result.reasons)

    def test_no_shap(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-shap2")
        result = agent.analyze(ctx)
        assert result.evidence_summary["shap_available"] is False
        assert any("SHAP explainability evidence is unavailable" in r for r in result.reasons)

    def test_no_graph_built(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-graph2")
        result = agent.analyze(ctx)
        assert result.evidence_summary["graph_available"] is False
        assert any("transaction graph has not been built" in r for r in result.reasons)

    def test_all_evidence_missing(self, agent):
        ctx = InvestigationContext(transaction_id="txn-all-missing")
        result = agent.analyze(ctx)
        assert result.evidence_summary["ml_available"] is False
        assert result.evidence_summary["shap_available"] is False
        assert result.evidence_summary["graph_available"] is False
        assert result.evidence_summary["network_risk_available"] is False
        assert result.evidence_summary["cluster_available"] is False
        assert len(result.reasons) >= 3


# ---------------------------------------------------------------------------
# Integration with full context
# ---------------------------------------------------------------------------


class TestFullContext:
    def test_all_evidence_present(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.88,
            risk_score=88,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        shap = SHAPEvidence(
            risk_factors=[
                {"feature": "amount", "impact": 0.35},
                {"feature": "velocity_5m", "impact": 0.20},
            ],
            risk_reducers=[
                {"feature": "avg_amount", "impact": -0.10},
            ],
        )
        graph = GraphEvidence(
            connected_transactions=[{"id": "t1"}, {"id": "t2"}],
            total_connections=2,
            entities=[{"type": "device", "value": "dev-1"}],
            entity_count=1,
            suspicious_neighbors=[{"id": "t1"}],
            suspicious_neighbor_count=1,
            shared_entity_types=["device"],
            graph_available=True,
        )
        network = NetworkRiskEvidence(
            network_risk_score=65.0,
            network_risk_level="MEDIUM",
            combined_risk_score=80.0,
            combined_risk_level="HIGH",
            factors=[],
            neighbor_count=2,
            suspicious_neighbor_count=1,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-full", "t1"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            shared_identifiers={"device": ["dev-1"]},
            risk_level="MEDIUM",
            avg_risk_score=72.0,
            strong_entity_types=["device"],
            weak_entity_types=[],
        )
        ctx = InvestigationContext(
            transaction_id="txn-full",
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )
        result = agent.analyze(ctx)

        assert result.transaction_id == "txn-full"
        assert result.risk_level == "HIGH"
        assert result.risk_score == 80.0
        assert len(result.risk_factors) == 2
        assert len(result.risk_reducers) == 1
        assert result.evidence_summary["ml_available"] is True
        assert result.evidence_summary["shap_available"] is True
        assert result.evidence_summary["graph_available"] is True
        assert result.evidence_summary["network_risk_available"] is True
        assert result.evidence_summary["cluster_available"] is True
        assert len(result.reasons) >= 3

    def test_only_graph_evidence(self, agent):
        graph = GraphEvidence(
            connected_transactions=[{"id": "t1"}],
            total_connections=1,
            suspicious_neighbors=[{"id": "t1"}],
            suspicious_neighbor_count=1,
            shared_entity_types=["card"],
            graph_available=True,
        )
        ctx = InvestigationContext(
            transaction_id="txn-graph-only",
            graph=graph,
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["graph_available"] is True
        assert result.evidence_summary["ml_available"] is False
        assert result.evidence_summary["network_risk_available"] is False
        assert result.risk_level == "UNKNOWN"
        assert result.risk_score == 0.0


# ---------------------------------------------------------------------------
# Regression: existing schemas still work
# ---------------------------------------------------------------------------


class TestSchemaRegression:
    def test_investigation_context_still_works(self):
        ctx = InvestigationContext(
            transaction_id="txn-regression",
            ml_prediction=MLPredictionEvidence(
                fraud_probability=0.5,
                risk_score=50,
                risk_level="MEDIUM",
                recommended_action="VERIFY",
            ),
        )
        assert ctx.transaction_id == "txn-regression"
        assert ctx.ml_prediction.risk_score == 50

    def test_risk_agent_result_is_valid_pydantic(self):
        result = RiskAgentResult(
            transaction_id="test",
            risk_level="LOW",
            risk_score=10.0,
            assessment="Low risk.",
            reasons=["reason"],
            risk_factors=[],
            risk_reducers=[],
            evidence_summary={"ml_available": False},
        )
        dumped = result.model_dump()
        assert dumped["transaction_id"] == "test"
        assert dumped["risk_level"] == "LOW"
