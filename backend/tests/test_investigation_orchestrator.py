import pytest
from unittest.mock import MagicMock, patch

from app.investigation.schemas import (
    InvestigationContext,
    MLPredictionEvidence,
    SHAPEvidence,
    GraphEvidence,
    NetworkRiskEvidence,
    ClusterEvidence,
    RiskAgentResult,
    PatternAgentResult,
    EvidenceAgentResult,
    EvidenceItem,
    DetectedPattern,
    AgentError,
    InvestigationResult,
)
from app.investigation.orchestrator import InvestigationOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_risk_result(txn_id="txn-001", risk_level="LOW", risk_score=10.0):
    return RiskAgentResult(
        transaction_id=txn_id,
        risk_level=risk_level,
        risk_score=risk_score,
        assessment=f"Transaction assessed as {risk_level} risk.",
        reasons=[],
        risk_factors=[],
        risk_reducers=[],
        evidence_summary={},
    )


def _make_pattern_result(txn_id="txn-001", patterns=None, pattern_count=0, summary="No patterns."):
    return PatternAgentResult(
        transaction_id=txn_id,
        patterns=patterns or [],
        pattern_count=pattern_count,
        summary=summary,
        evidence_summary={},
    )


def _make_evidence_result(txn_id="txn-001", evidence=None, evidence_count=0, summary="No evidence."):
    return EvidenceAgentResult(
        transaction_id=txn_id,
        evidence=evidence or [],
        evidence_count=evidence_count,
        summary=summary,
        availability={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def risk_agent_mock():
    mock = MagicMock()
    mock.analyze.return_value = _make_risk_result()
    return mock


@pytest.fixture
def pattern_agent_mock():
    mock = MagicMock()
    mock.analyze.return_value = _make_pattern_result()
    return mock


@pytest.fixture
def evidence_agent_mock():
    mock = MagicMock()
    mock.analyze.return_value = _make_evidence_result()
    return mock


@pytest.fixture
def orchestrator(risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
    return InvestigationOrchestrator(
        risk_agent_instance=risk_agent_mock,
        pattern_agent_instance=pattern_agent_mock,
        evidence_agent_instance=evidence_agent_mock,
    )


# ---------------------------------------------------------------------------
# 1. Basic orchestration
# ---------------------------------------------------------------------------


class TestBasicOrchestration:
    def test_accepts_valid_context(self, orchestrator):
        ctx = InvestigationContext(transaction_id="txn-001")
        result = orchestrator.investigate(ctx)
        assert isinstance(result, InvestigationResult)

    def test_returns_expected_fields(self, orchestrator):
        ctx = InvestigationContext(transaction_id="txn-002")
        result = orchestrator.investigate(ctx)
        assert result.transaction_id == "txn-002"
        assert hasattr(result, "risk_assessment")
        assert hasattr(result, "pattern_analysis")
        assert hasattr(result, "evidence")
        assert hasattr(result, "agent_errors")
        assert hasattr(result, "metadata")

    def test_invokes_risk_agent(self, orchestrator, risk_agent_mock):
        ctx = InvestigationContext(transaction_id="txn-003")
        orchestrator.investigate(ctx)
        risk_agent_mock.analyze.assert_called_once_with(ctx)

    def test_invokes_pattern_agent(self, orchestrator, pattern_agent_mock):
        ctx = InvestigationContext(transaction_id="txn-004")
        orchestrator.investigate(ctx)
        pattern_agent_mock.analyze.assert_called_once_with(ctx)

    def test_invokes_evidence_agent(self, orchestrator, evidence_agent_mock):
        ctx = InvestigationContext(transaction_id="txn-005")
        orchestrator.investigate(ctx)
        evidence_agent_mock.analyze.assert_called_once_with(ctx)

    def test_output_is_deterministic(self, orchestrator):
        ctx = InvestigationContext(transaction_id="txn-det")
        r1 = orchestrator.investigate(ctx)
        r2 = orchestrator.investigate(ctx)
        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# 2. Result preservation
# ---------------------------------------------------------------------------


class TestResultPreservation:
    def test_risk_result_preserved(self, orchestrator, risk_agent_mock):
        risk_result = _make_risk_result(
            txn_id="txn-preserve", risk_level="HIGH", risk_score=85.0
        )
        risk_agent_mock.analyze.return_value = risk_result

        ctx = InvestigationContext(transaction_id="txn-preserve")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is risk_result
        assert result.risk_assessment.risk_level == "HIGH"
        assert result.risk_assessment.risk_score == 85.0

    def test_pattern_result_preserved(self, orchestrator, pattern_agent_mock):
        pattern = DetectedPattern(
            pattern_type="suspicious_neighbors",
            description="2 suspicious neighbors found.",
            evidence={"count": 2},
            severity="MEDIUM",
        )
        pattern_result = _make_pattern_result(
            txn_id="txn-preserve2",
            patterns=[pattern],
            pattern_count=1,
            summary="1 pattern detected.",
        )
        pattern_agent_mock.analyze.return_value = pattern_result

        ctx = InvestigationContext(transaction_id="txn-preserve2")
        result = orchestrator.investigate(ctx)

        assert result.pattern_analysis is pattern_result
        assert result.pattern_analysis.pattern_count == 1
        assert result.pattern_analysis.patterns[0].pattern_type == "suspicious_neighbors"

    def test_evidence_result_preserved(self, orchestrator, evidence_agent_mock):
        evidence_item = EvidenceItem(
            evidence_type="ml_prediction",
            source="ml_prediction",
            description="ML prediction available.",
            details={"fraud_probability": 0.85},
            available=True,
        )
        evidence_result = _make_evidence_result(
            txn_id="txn-preserve3",
            evidence=[evidence_item],
            evidence_count=1,
            summary="1 evidence item.",
        )
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-preserve3")
        result = orchestrator.investigate(ctx)

        assert result.evidence is evidence_result
        assert result.evidence.evidence_count == 1
        assert result.evidence.evidence[0].evidence_type == "ml_prediction"

    def test_all_results_preserved_together(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(txn_id="txn-all", risk_level="MEDIUM", risk_score=55.0)
        pattern_result = _make_pattern_result(txn_id="txn-all", pattern_count=2)
        evidence_result = _make_evidence_result(txn_id="txn-all", evidence_count=3)

        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = pattern_result
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-all")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is risk_result
        assert result.pattern_analysis is pattern_result
        assert result.evidence is evidence_result

    def test_transaction_id_propagated(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(txn_id="txn-prop")
        pattern_result = _make_pattern_result(txn_id="txn-prop")
        evidence_result = _make_evidence_result(txn_id="txn-prop")

        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = pattern_result
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-prop")
        result = orchestrator.investigate(ctx)

        assert result.transaction_id == "txn-prop"
        assert result.risk_assessment.transaction_id == "txn-prop"
        assert result.pattern_analysis.transaction_id == "txn-prop"
        assert result.evidence.transaction_id == "txn-prop"


# ---------------------------------------------------------------------------
# 3. Agent independence
# ---------------------------------------------------------------------------


class TestAgentIndependence:
    def test_each_agent_called_exactly_once(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        ctx = InvestigationContext(transaction_id="txn-once")
        orchestrator.investigate(ctx)

        assert risk_agent_mock.analyze.call_count == 1
        assert pattern_agent_mock.analyze.call_count == 1
        assert evidence_agent_mock.analyze.call_count == 1

    def test_agent_calls_are_sequential(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        call_order = []

        def risk_side_effect(ctx):
            call_order.append("risk")
            return _make_risk_result()

        def pattern_side_effect(ctx):
            call_order.append("pattern")
            return _make_pattern_result()

        def evidence_side_effect(ctx):
            call_order.append("evidence")
            return _make_evidence_result()

        risk_agent_mock.analyze.side_effect = risk_side_effect
        pattern_agent_mock.analyze.side_effect = pattern_side_effect
        evidence_agent_mock.analyze.side_effect = evidence_side_effect

        ctx = InvestigationContext(transaction_id="txn-order")
        orchestrator.investigate(ctx)

        assert call_order == ["risk", "pattern", "evidence"]


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_context_same_result(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result(risk_level="HIGH", risk_score=80.0)
        pattern_agent_mock.analyze.return_value = _make_pattern_result(pattern_count=2)
        evidence_agent_mock.analyze.return_value = _make_evidence_result(evidence_count=4)

        ctx = InvestigationContext(transaction_id="txn-det")
        r1 = orchestrator.investigate(ctx)
        r2 = orchestrator.investigate(ctx)

        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# 5. Missing evidence
# ---------------------------------------------------------------------------


class TestMissingEvidence:
    def test_missing_ml_preserved(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result(risk_level="UNKNOWN")
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-noml")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.risk_level == "UNKNOWN"

    def test_missing_graph_preserved(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(risk_level="LOW")
        risk_result.evidence_summary = {"graph_available": False}
        risk_agent_mock.analyze.return_value = risk_result

        pattern_result = _make_pattern_result()
        pattern_result.evidence_summary = {"graph_available": False}
        pattern_agent_mock.analyze.return_value = pattern_result

        evidence_result = _make_evidence_result()
        evidence_result.availability = {"graph": False}
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-nograph")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.evidence_summary["graph_available"] is False
        assert result.pattern_analysis.evidence_summary["graph_available"] is False
        assert result.evidence.availability["graph"] is False

    def test_missing_cluster_preserved(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result()
        risk_result.evidence_summary = {"cluster_available": False}
        risk_agent_mock.analyze.return_value = risk_result

        pattern_result = _make_pattern_result()
        pattern_result.evidence_summary = {"cluster_available": False}
        pattern_agent_mock.analyze.return_value = pattern_result

        evidence_result = _make_evidence_result()
        evidence_result.availability = {"cluster": False}
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-nocluster")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.evidence_summary["cluster_available"] is False
        assert result.pattern_analysis.evidence_summary["cluster_available"] is False
        assert result.evidence.availability["cluster"] is False

    def test_full_context_preserved(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        ml = MLPredictionEvidence(
            fraud_probability=0.85, risk_score=85, risk_level="HIGH", recommended_action="MANUAL_REVIEW"
        )
        shap = SHAPEvidence(
            risk_factors=[{"feature": "amount", "value": 5000}],
            risk_reducers=[],
        )
        graph = GraphEvidence(
            graph_available=True,
            total_connections=3,
            entity_count=2,
            suspicious_neighbor_count=1,
            suspicious_neighbors=[{"transaction_id": "txn-nbr1", "fraud_probability": 0.9}],
            shared_entity_types=["device_id"],
        )
        network = NetworkRiskEvidence(
            network_risk_score=75.0, network_risk_level="HIGH",
            combined_risk_score=82.0, combined_risk_level="HIGH",
            neighbor_count=3, suspicious_neighbor_count=1,
        )
        cluster = ClusterEvidence(
            found=True, transaction_ids=["txn-001", "txn-nbr1"],
            total_transactions=2, entity_count=1,
            entity_types=["device_id"], suspicious_transaction_count=1,
            suspicious_ratio=0.5, risk_level="HIGH", avg_risk_score=80.0,
        )

        ctx = InvestigationContext(
            transaction_id="txn-full",
            transaction={"amount": 5000, "currency": "USD"},
            ml_prediction=ml,
            shap_explanation=shap,
            graph=graph,
            network_risk=network,
            cluster=cluster,
        )

        risk_result = _make_risk_result(txn_id="txn-full", risk_level="HIGH", risk_score=82.0)
        pattern_result = _make_pattern_result(txn_id="txn-full", pattern_count=2)
        evidence_result = _make_evidence_result(txn_id="txn-full", evidence_count=6)

        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = pattern_result
        evidence_agent_mock.analyze.return_value = evidence_result

        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.risk_level == "HIGH"
        assert result.risk_assessment.risk_score == 82.0
        assert result.pattern_analysis.pattern_count == 2
        assert result.evidence.evidence_count == 6


# ---------------------------------------------------------------------------
# 6. No hallucination
# ---------------------------------------------------------------------------


class TestNoHallucination:
    def test_no_fake_transaction_ids(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result(txn_id="txn-real")
        pattern_agent_mock.analyze.return_value = _make_pattern_result(txn_id="txn-real")
        evidence_agent_mock.analyze.return_value = _make_evidence_result(txn_id="txn-real")

        ctx = InvestigationContext(transaction_id="txn-real")
        result = orchestrator.investigate(ctx)

        assert result.transaction_id == "txn-real"
        assert result.risk_assessment.transaction_id == "txn-real"
        assert result.pattern_analysis.transaction_id == "txn-real"
        assert result.evidence.transaction_id == "txn-real"

    def test_no_fake_scores(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(risk_level="LOW", risk_score=10.0)
        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-nofake")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.risk_score == 10.0
        assert result.risk_assessment.risk_level == "LOW"

    def test_no_fake_patterns(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        pattern_result = _make_pattern_result(pattern_count=0, summary="No patterns found.")
        risk_agent_mock.analyze.return_value = _make_risk_result()
        pattern_agent_mock.analyze.return_value = pattern_result
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-nopatterns")
        result = orchestrator.investigate(ctx)

        assert result.pattern_analysis.pattern_count == 0
        assert len(result.pattern_analysis.patterns) == 0

    def test_no_fake_evidence(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        evidence_result = _make_evidence_result(evidence_count=0, summary="No evidence found.")
        risk_agent_mock.analyze.return_value = _make_risk_result()
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-noevidence")
        result = orchestrator.investigate(ctx)

        assert result.evidence.evidence_count == 0
        assert len(result.evidence.evidence) == 0


# ---------------------------------------------------------------------------
# 7. Score integrity
# ---------------------------------------------------------------------------


class TestScoreIntegrity:
    def test_fraud_probability_not_modified(self, orchestrator, risk_agent_mock):
        risk_result = _make_risk_result(risk_level="HIGH", risk_score=85.0)
        risk_result.evidence_summary = {"fraud_probability": 0.85}
        risk_agent_mock.analyze.return_value = risk_result
        orchestrator._pattern_agent = MagicMock()
        orchestrator._evidence_agent = MagicMock()
        orchestrator._pattern_agent.analyze.return_value = _make_pattern_result()
        orchestrator._evidence_agent.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-score")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.evidence_summary["fraud_probability"] == 0.85
        assert result.risk_assessment.risk_score == 85.0

    def test_risk_score_not_modified(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(risk_level="MEDIUM", risk_score=55.0)
        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-score2")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.risk_score == 55.0
        assert result.risk_assessment.risk_level == "MEDIUM"

    def test_network_risk_score_not_modified(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(risk_level="HIGH", risk_score=82.0)
        risk_result.evidence_summary = {
            "network_risk_score": 75.0,
            "combined_risk_score": 82.0,
        }
        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-net")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.evidence_summary["network_risk_score"] == 75.0
        assert result.risk_assessment.evidence_summary["combined_risk_score"] == 82.0

    def test_combined_risk_score_not_modified(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_result = _make_risk_result(risk_level="MEDIUM", risk_score=62.0)
        risk_result.evidence_summary = {"combined_risk_score": 62.0}
        risk_agent_mock.analyze.return_value = risk_result
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-combined")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment.evidence_summary["combined_risk_score"] == 62.0


# ---------------------------------------------------------------------------
# 8. Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_risk_agent_failure(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.side_effect = RuntimeError("Risk agent crashed")
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-fail-risk")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is None
        assert result.pattern_analysis is not None
        assert result.evidence is not None
        assert len(result.agent_errors) == 1
        assert result.agent_errors[0].agent_name == "RiskAgent"
        assert "Risk agent crashed" in result.agent_errors[0].error_message

    def test_pattern_agent_failure(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result()
        pattern_agent_mock.analyze.side_effect = RuntimeError("Pattern agent crashed")
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-fail-pattern")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is not None
        assert result.pattern_analysis is None
        assert result.evidence is not None
        assert len(result.agent_errors) == 1
        assert result.agent_errors[0].agent_name == "PatternAgent"
        assert "Pattern agent crashed" in result.agent_errors[0].error_message

    def test_evidence_agent_failure(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result()
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.side_effect = RuntimeError("Evidence agent crashed")

        ctx = InvestigationContext(transaction_id="txn-fail-evidence")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is not None
        assert result.pattern_analysis is not None
        assert result.evidence is None
        assert len(result.agent_errors) == 1
        assert result.agent_errors[0].agent_name == "EvidenceAgent"
        assert "Evidence agent crashed" in result.agent_errors[0].error_message

    def test_all_agents_fail(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.side_effect = RuntimeError("Risk crashed")
        pattern_agent_mock.analyze.side_effect = RuntimeError("Pattern crashed")
        evidence_agent_mock.analyze.side_effect = RuntimeError("Evidence crashed")

        ctx = InvestigationContext(transaction_id="txn-fail-all")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is None
        assert result.pattern_analysis is None
        assert result.evidence is None
        assert len(result.agent_errors) == 3

    def test_failure_does_not_swallow_other_results(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.side_effect = RuntimeError("Risk failed")
        pattern_result = _make_pattern_result(txn_id="txn-partial", pattern_count=1)
        evidence_result = _make_evidence_result(txn_id="txn-partial", evidence_count=2)

        pattern_agent_mock.analyze.return_value = pattern_result
        evidence_agent_mock.analyze.return_value = evidence_result

        ctx = InvestigationContext(transaction_id="txn-partial")
        result = orchestrator.investigate(ctx)

        assert result.risk_assessment is None
        assert result.pattern_analysis is pattern_result
        assert result.evidence is evidence_result
        assert len(result.agent_errors) == 1

    def test_failure_metadata_reflects_failure(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.side_effect = RuntimeError("crash")
        pattern_agent_mock.analyze.return_value = _make_pattern_result()
        evidence_agent_mock.analyze.return_value = _make_evidence_result()

        ctx = InvestigationContext(transaction_id="txn-meta-fail")
        result = orchestrator.investigate(ctx)

        assert result.metadata["risk_agent_invoked"] is False
        assert result.metadata["pattern_agent_invoked"] is True
        assert result.metadata["evidence_agent_invoked"] is True


# ---------------------------------------------------------------------------
# 9. Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_all_success(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        risk_agent_mock.analyze.return_value = _make_risk_result(risk_level="HIGH", risk_score=85.0)
        pattern_agent_mock.analyze.return_value = _make_pattern_result(pattern_count=3)
        evidence_agent_mock.analyze.return_value = _make_evidence_result(evidence_count=5)

        ctx = InvestigationContext(transaction_id="txn-meta")
        result = orchestrator.investigate(ctx)

        assert result.metadata["transaction_id"] == "txn-meta"
        assert result.metadata["risk_agent_invoked"] is True
        assert result.metadata["pattern_agent_invoked"] is True
        assert result.metadata["evidence_agent_invoked"] is True
        assert result.metadata["risk_level"] == "HIGH"
        assert result.metadata["risk_score"] == 85.0
        assert result.metadata["pattern_count"] == 3
        assert result.metadata["evidence_count"] == 5

    def test_metadata_no_errors(self, orchestrator):
        ctx = InvestigationContext(transaction_id="txn-meta-ok")
        result = orchestrator.investigate(ctx)
        assert result.agent_errors == []


# ---------------------------------------------------------------------------
# 10. No circular dependencies
# ---------------------------------------------------------------------------


class TestNoCircularDependencies:
    def test_agents_do_not_call_orchestrator(self, orchestrator, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        ctx = InvestigationContext(transaction_id="txn-nocirc")
        orchestrator.investigate(ctx)

        risk_agent_mock.analyze.assert_called_once()
        pattern_agent_mock.analyze.assert_called_once()
        evidence_agent_mock.analyze.assert_called_once()

        for mock in [risk_agent_mock, pattern_agent_mock, evidence_agent_mock]:
            mock.analyze.assert_called_once_with(ctx)


# ---------------------------------------------------------------------------
# 11. Constructor injection / defaults
# ---------------------------------------------------------------------------


class TestConstructorInjection:
    def test_default_agents_used_when_none_provided(self):
        from app.investigation.risk_agent import risk_agent
        from app.investigation.pattern_agent import pattern_agent
        from app.investigation.evidence_agent import evidence_agent

        orch = InvestigationOrchestrator()
        assert orch._risk_agent is risk_agent
        assert orch._pattern_agent is pattern_agent
        assert orch._evidence_agent is evidence_agent

    def test_custom_agents_injected(self, risk_agent_mock, pattern_agent_mock, evidence_agent_mock):
        orch = InvestigationOrchestrator(
            risk_agent_instance=risk_agent_mock,
            pattern_agent_instance=pattern_agent_mock,
            evidence_agent_instance=evidence_agent_mock,
        )
        assert orch._risk_agent is risk_agent_mock
        assert orch._pattern_agent is pattern_agent_mock
        assert orch._evidence_agent is evidence_agent_mock


# ---------------------------------------------------------------------------
# 12. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_investigation_result_valid_pydantic(self):
        result = InvestigationResult(
            transaction_id="test",
            risk_assessment=_make_risk_result(txn_id="test"),
            pattern_analysis=_make_pattern_result(txn_id="test"),
            evidence=_make_evidence_result(txn_id="test"),
            agent_errors=[],
            metadata={"key": "value"},
        )
        dumped = result.model_dump()
        assert dumped["transaction_id"] == "test"
        assert dumped["risk_assessment"]["risk_level"] == "LOW"
        assert dumped["agent_errors"] == []

    def test_agent_error_valid_pydantic(self):
        err = AgentError(agent_name="RiskAgent", error_message="Something went wrong")
        dumped = err.model_dump()
        assert dumped["agent_name"] == "RiskAgent"
        assert dumped["error_message"] == "Something went wrong"

    def test_investigation_result_with_none_agents(self):
        result = InvestigationResult(
            transaction_id="test-none",
            risk_assessment=None,
            pattern_analysis=None,
            evidence=None,
            agent_errors=[],
            metadata={},
        )
        dumped = result.model_dump()
        assert dumped["risk_assessment"] is None
        assert dumped["pattern_analysis"] is None
        assert dumped["evidence"] is None
