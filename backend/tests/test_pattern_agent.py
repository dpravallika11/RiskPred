import pytest

from app.investigation.schemas import (
    InvestigationContext,
    MLPredictionEvidence,
    GraphEvidence,
    NetworkRiskEvidence,
    ClusterEvidence,
    DetectedPattern,
    PatternAgentResult,
)
from app.investigation.pattern_agent import PatternAgent


@pytest.fixture
def agent():
    return PatternAgent()


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_accepts_valid_context(self, agent):
        ctx = InvestigationContext(transaction_id="txn-001")
        result = agent.analyze(ctx)
        assert isinstance(result, PatternAgentResult)

    def test_returns_expected_fields(self, agent):
        ctx = InvestigationContext(transaction_id="txn-002")
        result = agent.analyze(ctx)
        assert result.transaction_id == "txn-002"
        assert isinstance(result.patterns, list)
        assert isinstance(result.pattern_count, int)
        assert isinstance(result.summary, str)
        assert isinstance(result.evidence_summary, dict)

    def test_output_is_deterministic(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=4,
            suspicious_neighbor_count=2,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.8, "risk_level": "HIGH"},
            ],
            shared_entity_types=["device"],
            entity_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-det", graph=graph)
        r1 = agent.analyze(ctx)
        r2 = agent.analyze(ctx)
        assert r1.model_dump() == r2.model_dump()

    def test_pattern_count_matches_patterns_list(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=5,
            suspicious_neighbor_count=3,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.9, "risk_level": "HIGH"},
                {"transaction_id": "t2", "fraud_probability": 0.7, "risk_level": "MEDIUM"},
                {"transaction_id": "t3", "fraud_probability": 0.6, "risk_level": "MEDIUM"},
            ],
            shared_entity_types=["device", "card"],
            entity_count=3,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-det", "t1", "t2"],
            total_transactions=3,
            entity_types=["device"],
            suspicious_transaction_count=2,
            suspicious_ratio=0.67,
            risk_level="HIGH",
            avg_risk_score=75.0,
            strong_entity_types=["device"],
        )
        ctx = InvestigationContext(
            transaction_id="txn-det", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        assert result.pattern_count == len(result.patterns)


# ---------------------------------------------------------------------------
# Suspicious neighbors
# ---------------------------------------------------------------------------


class TestSuspiciousNeighbors:
    def test_suspicious_neighbor_produces_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            suspicious_neighbor_count=1,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.8, "risk_level": "HIGH"},
            ],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-sn", graph=graph)
        result = agent.analyze(ctx)
        assert any(p.pattern_type == "suspicious_neighbors" for p in result.patterns)

    def test_suspicious_neighbor_count_propagated(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=4,
            suspicious_neighbor_count=3,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.9, "risk_level": "HIGH"},
                {"transaction_id": "t2", "fraud_probability": 0.7, "risk_level": "MEDIUM"},
                {"transaction_id": "t3", "fraud_probability": 0.6, "risk_level": "MEDIUM"},
            ],
            entity_count=3,
        )
        ctx = InvestigationContext(transaction_id="txn-sn3", graph=graph)
        result = agent.analyze(ctx)
        sn_pattern = next(p for p in result.patterns if p.pattern_type == "suspicious_neighbors")
        assert sn_pattern.evidence["suspicious_neighbor_count"] == 3
        assert sn_pattern.severity == "HIGH"

    def test_single_suspicious_neighbor_medium_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=1,
            suspicious_neighbor_count=1,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.6, "risk_level": "MEDIUM"},
            ],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-sn1", graph=graph)
        result = agent.analyze(ctx)
        sn_pattern = next(p for p in result.patterns if p.pattern_type == "suspicious_neighbors")
        assert sn_pattern.severity == "MEDIUM"

    def test_two_suspicious_neighbors_medium_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            suspicious_neighbor_count=2,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.8, "risk_level": "HIGH"},
                {"transaction_id": "t2", "fraud_probability": 0.7, "risk_level": "MEDIUM"},
            ],
            entity_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-sn2", graph=graph)
        result = agent.analyze(ctx)
        sn_pattern = next(p for p in result.patterns if p.pattern_type == "suspicious_neighbors")
        assert sn_pattern.severity == "MEDIUM"

    def test_zero_suspicious_neighbors_no_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            suspicious_neighbor_count=0,
            suspicious_neighbors=[],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-sn0", graph=graph)
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "suspicious_neighbors" for p in result.patterns)

    def test_missing_graph_no_suspicious_neighbor_pattern(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-graph")
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "suspicious_neighbors" for p in result.patterns)

    def test_neighbor_details_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=1,
            suspicious_neighbor_count=1,
            suspicious_neighbors=[
                {
                    "transaction_id": "t1",
                    "fraud_probability": 0.85,
                    "risk_level": "HIGH",
                },
            ],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-det-neigh", graph=graph)
        result = agent.analyze(ctx)
        sn_pattern = next(p for p in result.patterns if p.pattern_type == "suspicious_neighbors")
        details = sn_pattern.evidence.get("suspicious_neighbor_details", [])
        assert len(details) == 1
        assert details[0]["transaction_id"] == "t1"
        assert details[0]["fraud_probability"] == 0.85


# ---------------------------------------------------------------------------
# Shared entities
# ---------------------------------------------------------------------------


class TestSharedEntities:
    def test_shared_entity_produces_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=1,
            shared_entity_types=["device"],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-se", graph=graph)
        result = agent.analyze(ctx)
        assert any(p.pattern_type == "shared_entities" for p in result.patterns)

    def test_shared_entity_types_preserved(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            shared_entity_types=["device", "card"],
            entity_count=4,
        )
        ctx = InvestigationContext(transaction_id="txn-se2", graph=graph)
        result = agent.analyze(ctx)
        se_pattern = next(p for p in result.patterns if p.pattern_type == "shared_entities")
        assert "device" in se_pattern.evidence["shared_entity_types"]
        assert "card" in se_pattern.evidence["shared_entity_types"]
        assert se_pattern.evidence["entity_type_count"] == 2

    def test_single_shared_entity_low_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=1,
            shared_entity_types=["card"],
            entity_count=1,
        )
        ctx = InvestigationContext(transaction_id="txn-se1", graph=graph)
        result = agent.analyze(ctx)
        se_pattern = next(p for p in result.patterns if p.pattern_type == "shared_entities")
        assert se_pattern.severity == "LOW"

    def test_two_shared_entities_medium_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            shared_entity_types=["device", "card"],
            entity_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-se2med", graph=graph)
        result = agent.analyze(ctx)
        se_pattern = next(p for p in result.patterns if p.pattern_type == "shared_entities")
        assert se_pattern.severity == "MEDIUM"

    def test_three_shared_entities_high_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            shared_entity_types=["device", "card", "address"],
            entity_count=3,
        )
        ctx = InvestigationContext(transaction_id="txn-se3", graph=graph)
        result = agent.analyze(ctx)
        se_pattern = next(p for p in result.patterns if p.pattern_type == "shared_entities")
        assert se_pattern.severity == "HIGH"

    def test_empty_shared_entity_types_no_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            shared_entity_types=[],
            entity_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-se-empty", graph=graph)
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "shared_entities" for p in result.patterns)

    def test_missing_graph_no_shared_entity_pattern(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-graph-se")
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "shared_entities" for p in result.patterns)


# ---------------------------------------------------------------------------
# Cluster membership
# ---------------------------------------------------------------------------


class TestClusterMembership:
    def test_cluster_produces_pattern(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-cl", "t1", "t2"],
            total_transactions=3,
            entity_count=2,
            entity_types=["device"],
            suspicious_transaction_count=2,
            suspicious_ratio=0.67,
            risk_level="HIGH",
            avg_risk_score=70.0,
            strong_entity_types=["device"],
        )
        graph = GraphEvidence(graph_available=True, total_connections=3)
        ctx = InvestigationContext(
            transaction_id="txn-cl", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        assert any(p.pattern_type == "cluster_membership" for p in result.patterns)

    def test_cluster_transaction_count_preserved(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-cl2", "t1", "t2", "t3"],
            total_transactions=4,
            entity_count=3,
            entity_types=["device", "card"],
            suspicious_transaction_count=3,
            suspicious_ratio=0.75,
            risk_level="HIGH",
            avg_risk_score=80.0,
            strong_entity_types=["device", "card"],
        )
        graph = GraphEvidence(graph_available=True, total_connections=4)
        ctx = InvestigationContext(
            transaction_id="txn-cl2", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_pattern = next(p for p in result.patterns if p.pattern_type == "cluster_membership")
        assert cl_pattern.evidence["cluster_transaction_count"] == 4
        assert cl_pattern.evidence["cluster_risk_level"] == "HIGH"

    def test_cluster_shared_identifiers_preserved(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-cl3", "t1"],
            total_transactions=2,
            entity_count=1,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            shared_identifiers={"device": ["dev-1"]},
            risk_level="MEDIUM",
            avg_risk_score=55.0,
        )
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-cl3", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_pattern = next(p for p in result.patterns if p.pattern_type == "cluster_membership")
        assert "device" in cl_pattern.evidence["shared_identifiers"]
        assert "dev-1" in cl_pattern.evidence["shared_identifiers"]["device"]

    def test_cluster_not_found_no_pattern(self, agent):
        cluster = ClusterEvidence(found=False)
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-cl-nf", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "cluster_membership" for p in result.patterns)

    def test_missing_cluster_no_pattern(self, agent):
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-cl-missing", graph=graph, cluster=None
        )
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "cluster_membership" for p in result.patterns)

    def test_cluster_severity_matches_risk_level(self, agent):
        for level in ("HIGH", "MEDIUM", "LOW"):
            cluster = ClusterEvidence(
                found=True,
                transaction_ids=["txn-cl-sev", "t1"],
                total_transactions=2,
                entity_count=1,
                entity_types=["card"],
                suspicious_transaction_count=1,
                suspicious_ratio=0.50,
                risk_level=level,
                avg_risk_score=50.0,
            )
            graph = GraphEvidence(graph_available=True, total_connections=2)
            ctx = InvestigationContext(
                transaction_id=f"txn-cl-sev-{level}", graph=graph, cluster=cluster
            )
            result = agent.analyze(ctx)
            cl_pattern = next(p for p in result.patterns if p.pattern_type == "cluster_membership")
            assert cl_pattern.severity == level

    def test_cluster_strong_entity_types_preserved(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-cl-strong", "t1"],
            total_transactions=2,
            entity_count=2,
            entity_types=["device", "card"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            risk_level="MEDIUM",
            avg_risk_score=60.0,
            strong_entity_types=["device"],
            weak_entity_types=["card"],
        )
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-cl-strong", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        cl_pattern = next(p for p in result.patterns if p.pattern_type == "cluster_membership")
        assert "device" in cl_pattern.evidence["strong_entity_types"]


# ---------------------------------------------------------------------------
# Dense connections
# ---------------------------------------------------------------------------


class TestDenseConnections:
    def test_three_connections_medium_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            entity_count=3,
            neighborhood_nodes=[{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            neighborhood_edges=[{"source": "txn", "target": "n1"}],
        )
        ctx = InvestigationContext(transaction_id="txn-dc3", graph=graph)
        result = agent.analyze(ctx)
        dc_pattern = next(p for p in result.patterns if p.pattern_type == "dense_connections")
        assert dc_pattern.severity == "MEDIUM"
        assert dc_pattern.evidence["total_connections"] == 3

    def test_five_connections_high_severity(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=5,
            entity_count=5,
            neighborhood_nodes=[{"id": f"n{i}"} for i in range(5)],
            neighborhood_edges=[],
        )
        ctx = InvestigationContext(transaction_id="txn-dc5", graph=graph)
        result = agent.analyze(ctx)
        dc_pattern = next(p for p in result.patterns if p.pattern_type == "dense_connections")
        assert dc_pattern.severity == "HIGH"
        assert dc_pattern.evidence["total_connections"] == 5

    def test_two_connections_no_dense_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            entity_count=2,
        )
        ctx = InvestigationContext(transaction_id="txn-dc2", graph=graph)
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "dense_connections" for p in result.patterns)

    def test_zero_connections_no_dense_pattern(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=0,
            entity_count=0,
        )
        ctx = InvestigationContext(transaction_id="txn-dc0", graph=graph)
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "dense_connections" for p in result.patterns)

    def test_missing_graph_no_dense_pattern(self, agent):
        ctx = InvestigationContext(transaction_id="txn-dc-nograph")
        result = agent.analyze(ctx)
        assert not any(p.pattern_type == "dense_connections" for p in result.patterns)


# ---------------------------------------------------------------------------
# Missing evidence handling
# ---------------------------------------------------------------------------


class TestMissingEvidence:
    def test_graph_unavailable_summary_message(self, agent):
        ctx = InvestigationContext(transaction_id="txn-miss")
        result = agent.analyze(ctx)
        assert result.pattern_count == 0
        assert "Graph evidence unavailable" in result.summary

    def test_graph_available_no_patterns_summary(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=0,
            entity_count=0,
        )
        ctx = InvestigationContext(transaction_id="txn-miss2", graph=graph)
        result = agent.analyze(ctx)
        assert result.pattern_count == 0
        assert "No suspicious patterns" in result.summary

    def test_evidence_summary_graph_available(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=2,
            suspicious_neighbor_count=1,
            shared_entity_types=["device"],
            entity_count=3,
            neighborhood_nodes=[{"id": "n1"}],
            neighborhood_edges=[{"source": "txn", "target": "n1"}],
        )
        ctx = InvestigationContext(transaction_id="txn-es", graph=graph)
        result = agent.analyze(ctx)
        assert result.evidence_summary["graph_available"] is True
        assert result.evidence_summary["total_connections"] == 2
        assert result.evidence_summary["entity_count"] == 3
        assert result.evidence_summary["suspicious_neighbor_count"] == 1
        assert result.evidence_summary["shared_entity_types"] == ["device"]
        assert result.evidence_summary["neighborhood_node_count"] == 1
        assert result.evidence_summary["neighborhood_edge_count"] == 1

    def test_evidence_summary_graph_unavailable(self, agent):
        ctx = InvestigationContext(transaction_id="txn-es-na")
        result = agent.analyze(ctx)
        assert result.evidence_summary["graph_available"] is False
        assert result.evidence_summary["cluster_available"] is False

    def test_evidence_summary_cluster_available(self, agent):
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["t1", "t2"],
            total_transactions=2,
            entity_types=["device"],
            suspicious_transaction_count=1,
            suspicious_ratio=0.50,
            risk_level="MEDIUM",
            avg_risk_score=55.0,
        )
        graph = GraphEvidence(graph_available=True, total_connections=2)
        ctx = InvestigationContext(
            transaction_id="txn-es-cl", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["cluster_available"] is True
        assert result.evidence_summary["cluster_total_transactions"] == 2
        assert result.evidence_summary["cluster_risk_level"] == "MEDIUM"

    def test_evidence_summary_cluster_not_found(self, agent):
        cluster = ClusterEvidence(found=False)
        graph = GraphEvidence(graph_available=True, total_connections=1)
        ctx = InvestigationContext(
            transaction_id="txn-es-cl-nf", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        assert result.evidence_summary["cluster_available"] is False


# ---------------------------------------------------------------------------
# No hallucination
# ---------------------------------------------------------------------------


class TestNoHallucination:
    def test_empty_context_no_patterns(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty")
        result = agent.analyze(ctx)
        assert result.pattern_count == 0
        assert result.patterns == []

    def test_empty_context_no_fake_neighbors(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty2")
        result = agent.analyze(ctx)
        assert "suspicious_neighbor_count" not in result.evidence_summary

    def test_empty_context_no_fake_entities(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty3")
        result = agent.analyze(ctx)
        assert "shared_entity_types" not in result.evidence_summary

    def test_empty_context_no_fake_cluster(self, agent):
        ctx = InvestigationContext(transaction_id="txn-empty4")
        result = agent.analyze(ctx)
        assert result.evidence_summary["cluster_available"] is False
        assert "cluster_total_transactions" not in result.evidence_summary

    def test_no_pattern_without_graph_evidence(self, agent):
        ctx = InvestigationContext(transaction_id="txn-no-graph")
        result = agent.analyze(ctx)
        assert result.pattern_count == 0
        for pattern_type in ("suspicious_neighbors", "shared_entities", "cluster_membership", "dense_connections"):
            assert not any(p.pattern_type == pattern_type for p in result.patterns)


# ---------------------------------------------------------------------------
# No risk-score duplication
# ---------------------------------------------------------------------------


class TestNoRiskScoreDuplication:
    def test_no_risk_score_field_in_result(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=5,
            suspicious_neighbor_count=3,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.9, "risk_level": "HIGH"},
            ],
            shared_entity_types=["device", "card"],
            entity_count=3,
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-nr", "t1", "t2"],
            total_transactions=3,
            entity_types=["device"],
            suspicious_transaction_count=2,
            suspicious_ratio=0.67,
            risk_level="HIGH",
            avg_risk_score=80.0,
        )
        ctx = InvestigationContext(
            transaction_id="txn-nr", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        dumped = result.model_dump()
        assert "risk_score" not in dumped
        assert "risk_level" not in dumped
        assert "fraud_probability" not in dumped

    def test_patterns_do_not_mutate_context(self, agent):
        ml = MLPredictionEvidence(
            fraud_probability=0.85,
            risk_score=85,
            risk_level="HIGH",
            recommended_action="MANUAL_REVIEW",
        )
        graph = GraphEvidence(
            graph_available=True,
            total_connections=4,
            suspicious_neighbor_count=2,
            shared_entity_types=["device"],
            entity_count=2,
        )
        ctx = InvestigationContext(
            transaction_id="txn-nomut", ml_prediction=ml, graph=graph
        )
        _ = agent.analyze(ctx)
        assert ctx.ml_prediction.risk_score == 85
        assert ctx.ml_prediction.risk_level == "HIGH"
        assert ctx.graph.total_connections == 4
        assert ctx.graph.suspicious_neighbor_count == 2


# ---------------------------------------------------------------------------
# Multiple patterns
# ---------------------------------------------------------------------------


class TestMultiplePatterns:
    def test_all_pattern_types_detected(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=6,
            suspicious_neighbor_count=3,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.9, "risk_level": "HIGH"},
                {"transaction_id": "t2", "fraud_probability": 0.7, "risk_level": "MEDIUM"},
                {"transaction_id": "t3", "fraud_probability": 0.6, "risk_level": "MEDIUM"},
            ],
            shared_entity_types=["device", "card"],
            entity_count=4,
            neighborhood_nodes=[{"id": f"n{i}"} for i in range(6)],
            neighborhood_edges=[],
        )
        cluster = ClusterEvidence(
            found=True,
            transaction_ids=["txn-all", "t1", "t2", "t3"],
            total_transactions=4,
            entity_count=3,
            entity_types=["device", "card"],
            suspicious_transaction_count=3,
            suspicious_ratio=0.75,
            risk_level="HIGH",
            avg_risk_score=75.0,
            strong_entity_types=["device", "card"],
        )
        ctx = InvestigationContext(
            transaction_id="txn-all", graph=graph, cluster=cluster
        )
        result = agent.analyze(ctx)
        pattern_types = {p.pattern_type for p in result.patterns}
        assert "suspicious_neighbors" in pattern_types
        assert "shared_entities" in pattern_types
        assert "cluster_membership" in pattern_types
        assert "dense_connections" in pattern_types
        assert result.pattern_count == 4

    def test_summary_lists_severity_breakdown(self, agent):
        graph = GraphEvidence(
            graph_available=True,
            total_connections=6,
            suspicious_neighbor_count=3,
            suspicious_neighbors=[
                {"transaction_id": "t1", "fraud_probability": 0.9, "risk_level": "HIGH"},
            ],
            shared_entity_types=["device", "card"],
            entity_count=3,
        )
        ctx = InvestigationContext(transaction_id="txn-sum", graph=graph)
        result = agent.analyze(ctx)
        assert "HIGH" in result.summary
        assert "MEDIUM" in result.summary


# ---------------------------------------------------------------------------
# Schema regression
# ---------------------------------------------------------------------------


class TestSchemaRegression:
    def test_detected_pattern_valid_pydantic(self):
        p = DetectedPattern(
            pattern_type="test",
            description="test desc",
            evidence={"key": "val"},
            severity="LOW",
        )
        dumped = p.model_dump()
        assert dumped["pattern_type"] == "test"
        assert dumped["severity"] == "LOW"

    def test_pattern_agent_result_valid_pydantic(self):
        result = PatternAgentResult(
            transaction_id="test",
            patterns=[],
            pattern_count=0,
            summary="No patterns.",
            evidence_summary={"graph_available": False},
        )
        dumped = result.model_dump()
        assert dumped["transaction_id"] == "test"
        assert dumped["pattern_count"] == 0
