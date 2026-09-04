import pytest

from app.investigation.schemas import (
    InvestigationContext,
    MLPredictionEvidence,
    SHAPEvidence,
    GraphEvidence,
    NetworkRiskEvidence,
    ClusterEvidence,
    EvidenceItem,
    EvidenceAgentResult,
)
from app.investigation.evidence_agent import EvidenceAgent


@pytest.fixture
def agent():
    return EvidenceAgent()


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_accepts_valid_context(self, agent):
        ctx = InvestigationContext(transaction_id="txn-001")
        result = agent.analyze(ctx)
        assert isinstance(result, EvidenceAgentResult)

    def test_returns_expected_fields(self, agent):
        ctx = InvestigationContext(transaction_id="txn-002")
        result = agent.analyze(ctx)
        assert result.transaction_id == "txn-002"
        assert isinstance(result.evidence, list)
        assert isinstance(result.evidence_count, int)
        assert isinstance(result.summary, str)
        assert isinstance(result.availability, dict)

    def test_output_is_deterministic(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.85,
            risk_score=85,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(transaction_id="txn-det", ml_prediction=ml)
        r1 = agent.analyze(ctx)
        r2 = agent.analyze(ctx)
        assert r1.model_dump() == r2.model_dump()

    def test_evidence_count_matches_evidence_list(self, agent):
        ctx = InvestigationContext(transaction_id="txn-count")
        result = agent.analyze(ctx)
        assert result.evidence_count == len(result.evidence)

    def test_all_evidence_items_are_valid_pydantic(self, agent):
        ctx = InvestigationContext(transaction_id="txn-valid")
        result = agent.analyze(ctx)
        for item in result.evidence:
            assert isinstance(item, EvidenceItem)


# ---------------------------------------------------------------------------
# Transaction evidence
# ---------------------------------------------------------------------------


class TestTransactionEvidence:
    def test_transaction_included_when_available(self, agent):
        txn = {"amount": 1500.0, "merchant_id": "M001", "currency": "USD"}
        ctx = InvestigationContext(transaction_id="txn-t1", transaction=txn)
        result = agent.analyze(ctx)
        txn_items = [e for e in result.evidence if e.evidence_type == "transaction"]
        assert len(txn_items) == 1
        assert txn_items[0].available is True

    def test_transaction_fields_preserved(self, agent):
        txn = {"amount": 2500.0, "device_id": "DEV-123", "is_new_device": True}
        ctx = InvestigationContext(transaction_id="txn-t2", transaction=txn)
        result = agent.analyze(ctx)
        txn_item = next(e for e in result.evidence if e.evidence_type == "transaction")
        assert txn_item.details["amount"] == 2500.0
        assert txn_item.details["device_id"] == "DEV-123"
        assert txn_item.details["is_new_device"] is True

    def test_missing_transaction_data(self, agent):
        ctx = InvestigationContext(transaction_id="txn-t3", transaction=None)
        result = agent.analyze(ctx)
        txn_item = next(e for e in result.evidence if e.evidence_type == "transaction")
        assert txn_item.available is False
        assert "unavailable" in txn_item.description.lower()

    def test_transaction_source_is_transaction(self, agent):
        txn = {"amount": 100.0}
        ctx = InvestigationContext(transaction_id="txn-t4", transaction=txn)
        result = agent.analyze(ctx)
        txn_item = next(e for e in result.evidence if e.evidence_type == "transaction")
        assert txn_item.source == "transaction"


# ---------------------------------------------------------------------------
# ML evidence
# ---------------------------------------------------------------------------


class TestMLEvidence:
    def test_ml_evidence_included_when_available(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.92,
            risk_score=92,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(transaction_id="txn-ml1", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_items = [e for e in result.evidence if e.evidence_type == "ml_prediction"]
        assert len(ml_items) == 1
        assert ml_items[0].available is True

    def test_fraud_probability_preserved(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.73,
            risk_score=73,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(transaction_id="txn-ml2", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["fraud_probability"] == 0.73

    def test_risk_score_preserved(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.45,
            risk_score=45,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(transaction_id="txn-ml3", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["risk_score"] == 45

    def test_risk_level_preserved(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.10,
            risk_score=10,
            risk_level="LOW",
            recommended_action="ALLOW",
        )
        ctx = InvestigationContext(transaction_id="txn-ml4", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["risk_level"] == "LOW"

    def test_recommended_action_preserved(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.60,
            risk_score=60,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(transaction_id="txn-ml5", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["recommended_action"] == "VERIFY"

    def test_missing_ml_evidence(self, agent):
        ctx = InvestigationContext(transaction_id="txn-ml6", ml_prediction=None)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.available is False
        assert "unavailable" in ml_item.description.lower()

    def test_ml_source_is_ml_prediction(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(transaction_id="txn-ml7", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.source == "ml_prediction"


# ---------------------------------------------------------------------------
# SHAP evidence
# ---------------------------------------------------------------------------


class TestSHAPEvidence:
    def test_shap_risk_factors_preserved(self, agent):
        shap = SHAPEvidence(
            risk_factors=[
                {"feature": "amount", "impact": 0.3},
                {"feature": "velocity_5m", "impact": 0.2},
            ],
            risk_reducers=[],
        )
        ctx = InvestigationContext(transaction_id="txn-shap1", shap_explanation=shap)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert len(shap_item.details["risk_factors"]) == 2
        assert shap_item.details["risk_factors"][0]["feature"] == "amount"

    def test_shap_risk_reducers_preserved(self, agent):
        shap = SHAPEvidence(
            risk_factors=[],
            risk_reducers=[
                {"feature": "avg_transaction_amount", "impact": -0.15},
            ],
        )
        ctx = InvestigationContext(transaction_id="txn-shap2", shap_explanation=shap)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert len(shap_item.details["risk_reducers"]) == 1
        assert shap_item.details["risk_reducers"][0]["feature"] == "avg_transaction_amount"

    def test_missing_shap_no_explanations(self, agent):
        ctx = InvestigationContext(transaction_id="txn-shap3", shap_explanation=None)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert shap_item.available is False
        assert "risk_factors" not in shap_item.details
        assert "risk_reducers" not in shap_item.details

    def test_shap_factor_count_preserved(self, agent):
        shap = SHAPEvidence(
            risk_factors=[
                {"feature": "a", "impact": 0.1},
                {"feature": "b", "impact": 0.2},
                {"feature": "c", "impact": 0.3},
            ],
            risk_reducers=[{"feature": "d", "impact": -0.1}],
        )
        ctx = InvestigationContext(transaction_id="txn-shap4", shap_explanation=shap)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert shap_item.details["risk_factor_count"] == 3
        assert shap_item.details["risk_reducer_count"] == 1

    def test_shap_source_is_shap(self, agent):
        shap = SHAPEvidence(risk_factors=[], risk_reducers=[])
        ctx = InvestigationContext(transaction_id="txn-shap5", shap_explanation=shap)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert shap_item.source == "shap"

    def test_empty_shap_factors_description(self, agent):
        shap = SHAPEvidence(risk_factors=[], risk_reducers=[])
        ctx = InvestigationContext(transaction_id="txn-shap6", shap_explanation=shap)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert "no significant factors" in shap_item.description.lower()


# ---------------------------------------------------------------------------
# Graph evidence
# ---------------------------------------------------------------------------


class TestGraphEvidence:
    def test_graph_included_when_available(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            entity_count=2,
            suspicious_neighbor_count=1,
            shared_entity_types=["device"],
        )
        ctx = InvestigationContext(transaction_id="txn-g1", graph=graph)
        result = agent.analyze(ctx)
        graph_items = [e for e in result.evidence if e.evidence_type == "graph"]
        assert len(graph_items) == 1
        assert graph_items[0].available is True

    def test_connection_counts_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=5,
            entity_count=4,
            suspicious_neighbor_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-g2", graph=graph)
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert graph_item.details["total_connections"] == 5
        assert graph_item.details["entity_count"] == 4

    def test_suspicious_neighbor_info_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            entity_count=1,
            suspicious_neighbor_count=1,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.8, "risk_level": "HIGH"}
            ],
        )
        ctx = InvestigationContext(transaction_id="txn-g3", graph=graph)
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert graph_item.details["suspicious_neighbor_count"] == 1
        assert len(graph_item.details["suspicious_neighbors"]) == 1

    def test_entity_info_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=1,
            entity_count=2,
            entities=[
                {"type": "device", "value": "dev-1"},
                {"type": "card", "value": "card-1"},
            ],
        )
        ctx = InvestigationContext(transaction_id="txn-g4", graph=graph)
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert len(graph_item.details["entities"]) == 2

    def test_missing_graph_evidence(self, agent):
        ctx = InvestigationContext(transaction_id="txn-g5")
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert graph_item.available is False
        assert "unavailable" in graph_item.description.lower()

    def test_graph_source_is_graph(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=0)
        ctx = InvestigationContext(transaction_id="txn-g6", graph=graph)
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert graph_item.source == "graph"

    def test_shared_entity_types_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            shared_entity_types=["device", "card"],
        )
        ctx = InvestigationContext(transaction_id="txn-g7", graph=graph)
        result = agent.analyze(ctx)
        graph_item = next(e for e in result.evidence if e.evidence_type == "graph")
        assert "device" in graph_item.details["shared_entity_types"]
        assert "card" in graph_item.details["shared_entity_types"]


# ---------------------------------------------------------------------------
# Network risk
# ---------------------------------------------------------------------------


class TestNetworkRisk:
    def test_network_risk_included_when_available(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        network = NetworkRiskEvidence(
            network_risk_score=65.0,
            network_risk_level="MEDIUM",
            combined_risk_score=72.0,
            combined_risk_level="MEDIUM",
            neighbor_count=2,
            suspicious_neighbor_count=1,
        )
        ctx = InvestigationContext(
            transaction_id="txn-nr1", graph=graph, network_risk=network
        )
        result = agent.analyze(ctx)
        nr_items = [e for e in result.evidence if e.evidence_type == "network_risk"]
        assert len(nr_items) == 1
        assert nr_items[0].available is True

    def test_network_risk_values_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=3)
        network = NetworkRiskEvidence(
            network_risk_score=80.0,
            network_risk_level="HIGH",
            combined_risk_score=85.0,
            combined_risk_level="HIGH",
            neighbor_count=3,
            suspicious_neighbor_count=2,
        )
        ctx = InvestigationContext(
            transaction_id="txn-nr2", graph=graph, network_risk=network
        )
        result = agent.analyze(ctx)
        nr_item = next(e for e in result.evidence if e.evidence_type == "network_risk")
        assert nr_item.details["network_risk_score"] == 80.0
        assert nr_item.details["combined_risk_score"] == 85.0

    def test_no_new_risk_formula(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        network = NetworkRiskEvidence(
            network_risk_score=55.0,
            network_risk_level="MEDIUM",
            combined_risk_score=62.0,
            combined_risk_level="MEDIUM",
            neighbor_count=2,
            suspicious_neighbor_count=0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-nr3", graph=graph, network_risk=network
        )
        result = agent.analyze(ctx)
        nr_item = next(e for e in result.evidence if e.evidence_type == "network_risk")
        assert nr_item.details["network_risk_score"] == 55.0
        assert nr_item.details["combined_risk_score"] == 62.0

    def test_missing_network_risk_with_graph(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-nr4", graph=graph, network_risk=None
        )
        result = agent.analyze(ctx)
        nr_item = next(e for e in result.evidence if e.evidence_type == "network_risk")
        assert nr_item.available is False
        assert "unavailable" in nr_item.description.lower()

    def test_no_network_evidence_without_graph(self, agent):
        ctx = InvestigationContext(transaction_id="txn-nr5")
        result = agent.analyze(ctx)
        nr_items = [e for e in result.evidence if e.evidence_type == "network_risk"]
        assert len(nr_items) == 0

    def test_network_risk_factors_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        network = NetworkRiskEvidence(
            network_risk_score=70.0,
            network_risk_level="HIGH",
            combined_risk_score=75.0,
            combined_risk_level="HIGH",
            neighbor_count=2,
            suspicious_neighbor_count=1,
            factors=[{"name": "suspicious_neighbors", "weight": 0.6}],
        )
        ctx = InvestigationContext(
            transaction_id="txn-nr6", graph=graph, network_risk=network
        )
        result = agent.analyze(ctx)
        nr_item = next(e for e in result.evidence if e.evidence_type == "network_risk")
        assert len(nr_item.details["factors"]) == 1


# ---------------------------------------------------------------------------
# Cluster evidence
# ---------------------------------------------------------------------------


class TestClusterEvidence:
    def test_cluster_included_when_found(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=3)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2", "t3"],
            total_transactions=3,
            entity_count=2,
            entity_types=["device", "card"],
            suspicious_transaction_count=2,
            suspicious_ratio=0.67,
            risk_level="HIGH",
            avg_risk_score=75.0,
            strong_entity_types=["device"],
            weak_entity_types=["card"],
        )
        ctx = InvestigationContext(
            transaction_id="txn-c1", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_items = [e for e in result.evidence if e.evidence_type == "cluster"]
        assert len(cl_items) == 1
        assert cl_items[0].available is True

    def test_cluster_transaction_ids_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-c2", "t1"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            risk_level="MEDIUM",
            avg_risk_score=55.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-c2", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert "txn-c2" in cl_item.details["transaction_ids"]
        assert "t1" in cl_item.details["transaction_ids"]

    def test_cluster_entity_info_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=3)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2", "t3"],
            total_transactions=3,
            entity_count=3,
            entity_types=["device", "card", "address"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.33,
            risk_level="MEDIUM",
            avg_risk_score=45.0,
            strong_entity_types=["device", "card"],
            weak_entity_types=["address"],
        )
        ctx = InvestigationContext(
            transaction_id="txn-c3", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.details["entity_count"] == 3
        assert "device" in cl_item.details["strong_entity_types"]
        assert "address" in cl_item.details["weak_entity_types"]

    def test_cluster_risk_info_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=2,
            suspicious_ratio=1.0,
            risk_level="HIGH",
            avg_risk_score=90.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-c4", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.details["risk_level"] == "HIGH"
        assert cl_item.details["avg_risk_score"] == 90.0
        assert cl_item.details["suspicious_ratio"] == 1.0

    def test_cluster_shared_identifiers_preserved(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            shared_identifiers={"device": ["dev-1", "dev-2"]},
            risk_level="MEDIUM",
            avg_risk_score=55.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-c5", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert "device" in cl_item.details["shared_identifiers"]
        assert "dev-1" in cl_item.details["shared_identifiers"]["device"]

    def test_missing_cluster_evidence(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-c6", graph=graph, cluster=None
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.available is False
        assert "unavailable" in cl_item.description.lower()

    def test_cluster_not_found(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        cluster = ClusterEvidence(found=False)
        ctx = InvestigationContext(
            transaction_id="txn-c7", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.available is False

    def test_no_cluster_evidence_without_graph(self, agent):
        ctx = InvestigationContext(transaction_id="txn-c8")
        result = agent.analyze(ctx)
        cl_items = [e for e in result.evidence if e.evidence_type == "cluster"]
        assert len(cl_items) == 0

    def test_cluster_source_is_cluster(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=1)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1"],
            total_transactions=1,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=0,
            suspicious_ratio=0.0,
            risk_level="LOW",
            avg_risk_score=10.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-c9", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.source == "cluster"


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


class TestTraceability:
    def test_all_evidence_items_have_valid_source(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.80,
            risk_score=80,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        shap = SHAPEvidence(
            risk_factors=[{"feature": "amount", "impact": 0.3}],
            risk_reducers=[],
        )
        graph = GraphEvidence(
            graph_available=True,
            total_connections=4,
            entity_count=3,
            suspicious_neighbor_count=2,
            shared_entity_types=["device"],
        )
        network = NetworkRiskEvidence(
            network_risk_score=70.0,
            network_risk_level="HIGH",
            combined_risk_score=78.0,
            combined_risk_level="HIGH",
            neighbor_count=4,
            suspicious_neighbor_count=2,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2", "t3"],
            total_transactions=3,
            entity_count=2,
            entity_types=["device"],
            suspicious_transaction_count=2,
            suspicious_ratio=0.67,
            risk_level="HIGH",
            avg_risk_score=75.0,
            strong_entity_types=["device"],
        )
        ctx = InvestigationContext(
            transaction_id="txn-trace",
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        valid_sources = {
            "transaction", "ml_prediction", "shap", "graph",
            "network_risk", "cluster",
        }
        for item in result.evidence:
            assert item.source in valid_sources

    def test_evidence_type_matches_source(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(
            transaction_id="txn-match", ml_prediction=ml
        )
        result = agent.analyze(ctx)
        for item in result.evidence:
            assert item.evidence_type == item.source


# ---------------------------------------------------------------------------
# No hallucination
# ---------------------------------------------------------------------------


class TestNoHallucination:
    def test_empty_context_no_fake_ids(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty")
        result = agent.analyze(ctx)
        for item in result.evidence:
            if item.available:
                for key, value in item.details.items():
                    if isinstance(value, str) and "ID" in key.upper():
                        assert value != "" or "id" not in key.lower()

    def test_empty_context_no_fake_scores(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty2")
        result = agent.analyze(ctx)
        for item in result.evidence:
            if item.available:
                assert "fraud_probability" not in item.details
                assert "risk_score" not in item.details or item.details.get("risk_score") == 0

    def test_empty_context_no_fake_entities(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty3")
        result = agent.analyze(ctx)
        for item in result.evidence:
            if item.available and item.evidence_type == "graph":
                assert item.details.get("entity_count", 0) == 0
                assert item.details.get("entities", []) == []

    def test_missing_shap_no_fake_factors(self, agent):
        ctx = InvestigationContext(transaction_id="txn-noshap", shap_explanation=None)
        result = agent.analyze(ctx)
        shap_item = next(e for e in result.evidence if e.evidence_type == "shap")
        assert shap_item.details == {}

    def test_missing_cluster_no_fake_members(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=1)
        ctx = InvestigationContext(
            transaction_id="txn-nocluster", graph=graph, cluster=None
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.details == {}

    def test_all_contexts_stable(self, agent):
        ctx = InvestigationContext(transaction_id="txn-stable")
        results = [agent.analyze(ctx) for _ in range(5)]
        for i in range(1, len(results)):
            assert results[0].model_dump() == results[i].model_dump()


# ---------------------------------------------------------------------------
# Score integrity
# ---------------------------------------------------------------------------


class TestScoreIntegrity:
    def test_fraud_probability_not_modified(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.87,
            risk_score=87,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        ctx = InvestigationContext(transaction_id="txn-si1", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["fraud_probability"] == 0.87
        assert ml.fraud_probability == 0.87

    def test_risk_score_not_modified(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.65,
            risk_score=65,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        ctx = InvestigationContext(transaction_id="txn-si2", ml_prediction=ml)
        result = agent.analyze(ctx)
        ml_item = next(e for e in result.evidence if e.evidence_type == "ml_prediction")
        assert ml_item.details["risk_score"] == 65
        assert ml.risk_score == 65

    def test_network_risk_score_not_modified(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        network = NetworkRiskEvidence(
            network_risk_score=55.0,
            network_risk_level="MEDIUM",
            combined_risk_score=62.0,
            combined_risk_level="MEDIUM",
            neighbor_count=2,
            suspicious_neighbor_count=0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-si3", graph=graph, network_risk=network
        )
        result = agent.analyze(ctx)
        nr_item = next(e for e in result.evidence if e.evidence_type == "network_risk")
        assert nr_item.details["network_risk_score"] == 55.0
        assert nr_item.details["combined_risk_score"] == 62.0
        assert network.network_risk_score == 55.0
        assert network.combined_risk_score == 62.0

    def test_cluster_scores_not_modified(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            risk_level="MEDIUM",
            avg_risk_score=55.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-si4", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_item = next(e for e in result.evidence if e.evidence_type == "cluster")
        assert cl_item.details["avg_risk_score"] == 55.0
        assert cl_item.details["suspicious_ratio"] == 0.50
        assert cluster.avg_risk_score == 55.0
        assert cluster.suspicious_ratio == 0.50


# ---------------------------------------------------------------------------
# Stable ordering
# ---------------------------------------------------------------------------


class TestStableOrdering:
    def test_evidence_ordering_stable(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.70,
            risk_score=70,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        shap = SHAPEvidence(
            risk_factors=[{"feature": "amount", "impact": 0.2}],
            risk_reducers=[],
        )
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            entity_count=2,
            suspicious_neighbor_count=1,
        )
        network = NetworkRiskEvidence(
            network_risk_score=60.0,
            network_risk_level="MEDIUM",
            combined_risk_score=68.0,
            combined_risk_level="MEDIUM",
            neighbor_count=3,
            suspicious_neighbor_count=1,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            risk_level="MEDIUM",
            avg_risk_score=50.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-order",
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )
        r1 = agent.analyze(ctx)
        r2 = agent.analyze(ctx)
        types1 = [e.evidence_type for e in r1.evidence]
        types2 = [e.evidence_type for e in r2.evidence]
        assert types1 == types2


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_availability_all_present(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        shap = SHAPEvidence(risk_factors=[], risk_reducers=[])
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            entity_count=1,
        )
        network = NetworkRiskEvidence(
            network_risk_score=40.0,
            network_risk_level="LOW",
            combined_risk_score=47.0,
            combined_risk_level="LOW",
            neighbor_count=2,
            suspicious_neighbor_count=0,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1"],
            total_transactions=1,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=0,
            suspicious_ratio=0.0,
            risk_level="LOW",
            avg_risk_score=10.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-avail",
            transaction={"amount": 100.0},
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert result.availability["transaction"] is True
        assert result.availability["ml_prediction"] is True
        assert result.availability["shap"] is True
        assert result.availability["graph"] is True
        assert result.availability["network_risk"] is True
        assert result.availability["cluster"] is True

    def test_availability_all_missing(self, agent):
        ctx = InvestigationContext(transaction_id="txn-avail2")
        result = agent.analyze(ctx)
        assert result.availability["transaction"] is False
        assert result.availability["ml_prediction"] is False
        assert result.availability["shap"] is False
        assert result.availability["graph"] is False
        assert result.availability["network_risk"] is False
        assert result.availability["cluster"] is False

    def test_availability_partial(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.30,
            risk_score=30,
            risk_level="LOW",
            recommended_action="ALLOW",
        )
        ctx = InvestigationContext(
            transaction_id="txn-avail3", ml_prediction=ml
        )
        result = agent.analyze(ctx)
        assert result.availability["transaction"] is False
        assert result.availability["ml_prediction"] is True
        assert result.availability["shap"] is False
        assert result.availability["graph"] is False
        assert result.availability["network_risk"] is False
        assert result.availability["cluster"] is False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_all_available(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        shap = SHAPEvidence(risk_factors=[], risk_reducers=[])
        graph = GraphEvidence(graph_available=True, total_connections=2)
        network = NetworkRiskEvidence(
            network_risk_score=40.0,
            network_risk_level="LOW",
            combined_risk_score=47.0,
            combined_risk_level="LOW",
            neighbor_count=2,
            suspicious_neighbor_count=0,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1"],
            total_transactions=1,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=0,
            suspicious_ratio=0.0,
            risk_level="LOW",
            avg_risk_score=10.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-sum1",
            transaction={"amount": 100.0},
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )
        result = agent.analyze(ctx)
        assert "6 evidence item(s)" in result.summary
        assert "6 available" in result.summary

    def test_summary_all_unavailable(self, agent):
        ctx = InvestigationContext(transaction_id="txn-sum2")
        result = agent.analyze(ctx)
        assert "4 evidence item(s)" in result.summary
        assert "4 unavailable" in result.summary

    def test_summary_mixed(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.50,
            risk_score=50,
            risk_level="MEDIUM",
            recommended_action="VERIFY",
        )
        graph = GraphEvidence(graph_available=True, total_connections=1)
        ctx = InvestigationContext(
            transaction_id="txn-sum3", ml_prediction=ml, graph=graph
        )
        result = agent.analyze(ctx)
        assert "6 evidence item(s)" in result.summary
        assert "2 available" in result.summary
        assert "4 unavailable" in result.summary


# ---------------------------------------------------------------------------
# Schema regression
# ---------------------------------------------------------------------------


class TestSchemaRegression:
    def test_evidence_item_valid_pydantic(self):
        item = EvidenceItem(
            evidence_type="test",
            source="test_source",
            description="Test description.",
            details={"key": "value"},
            available=True,
        )
        dumped = item.model_dump()
        assert dumped["evidence_type"] == "test"
        assert dumped["available"] is True

    def test_evidence_agent_result_valid_pydantic(self):
        result = EvidenceAgentResult(
            transaction_id="test",
            evidence=[],
            evidence_count=0,
            summary="No evidence.",
            availability={"graph": False},
        )
        dumped = result.model_dump()
        assert dumped["transaction_id"] == "test"
        assert dumped["evidence_count"] == 0
