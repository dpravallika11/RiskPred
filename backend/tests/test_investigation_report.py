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
    InvestigationReport,
)
from app.investigation.report import InvestigationReportGenerator, report_generator
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


def _make_evidence_result(txn_id="txn-001", evidence=None, evidence_count=0, summary="No evidence.", availability=None):
    return EvidenceAgentResult(
        transaction_id=txn_id,
        evidence=evidence or [],
        evidence_count=evidence_count,
        summary=summary,
        availability=availability or {},
    )


def _make_result(
    txn_id="txn-001",
    risk_level="LOW",
    risk_score=10.0,
    pattern_count=0,
    evidence_count=0,
    agent_errors=None,
):
    return InvestigationResult(
        transaction_id=txn_id,
        risk_assessment=_make_risk_result(txn_id, risk_level, risk_score),
        pattern_analysis=_make_pattern_result(txn_id, pattern_count=pattern_count),
        evidence=_make_evidence_result(txn_id, evidence_count=evidence_count),
        agent_errors=agent_errors or [],
        metadata={},
    )


# ---------------------------------------------------------------------------
# 1. Basic report generation
# ---------------------------------------------------------------------------


class TestBasicReportGeneration:
    def test_valid_result_produces_report(self):
        result = _make_result()
        report = report_generator.generate(result)
        assert isinstance(report, InvestigationReport)

    def test_transaction_id_preserved(self):
        result = _make_result(txn_id="txn-42")
        report = report_generator.generate(result)
        assert report.transaction_id == "txn-42"

    def test_report_schema_valid(self):
        result = _make_result()
        report = report_generator.generate(result)
        dumped = report.model_dump()
        assert "transaction_id" in dumped
        assert "risk_assessment" in dumped
        assert "detected_patterns" in dumped
        assert "evidence" in dumped
        assert "conclusion" in dumped
        assert "recommended_action" in dumped
        assert "agent_errors" in dumped
        assert "metadata" in dumped

    def test_deterministic_output(self):
        result = _make_result()
        r1 = report_generator.generate(result)
        r2 = report_generator.generate(result)
        assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# 2. Risk assessment preservation
# ---------------------------------------------------------------------------


class TestRiskPreservation:
    def test_risk_level_preserved(self):
        result = _make_result(risk_level="HIGH", risk_score=85.0)
        report = report_generator.generate(result)
        assert report.risk_assessment.risk_level == "HIGH"
        assert report.risk_assessment.risk_score == 85.0

    def test_risk_score_not_modified(self):
        result = _make_result(risk_level="MEDIUM", risk_score=55.0)
        report = report_generator.generate(result)
        assert report.risk_assessment.risk_score == 55.0
        assert report.risk_assessment.risk_level == "MEDIUM"

    def test_recommended_action_preserved(self):
        risk = _make_risk_result(risk_level="HIGH", risk_score=85.0)
        risk.evidence_summary = {"recommended_action": "MANUAL_REVIEW"}
        result = InvestigationResult(
            transaction_id="txn-act",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-act"),
            evidence=_make_evidence_result(txn_id="txn-act"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.recommended_action == "MANUAL_REVIEW"

    def test_risk_factors_preserved(self):
        risk = _make_risk_result()
        risk.risk_factors = [{"feature": "amount", "value": 5000}]
        risk.risk_reducers = [{"feature": "card_age", "value": 5}]
        result = InvestigationResult(
            transaction_id="txn-factors",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-factors"),
            evidence=_make_evidence_result(txn_id="txn-factors"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment.risk_factors == [{"feature": "amount", "value": 5000}]
        assert report.risk_assessment.risk_reducers == [{"feature": "card_age", "value": 5}]


# ---------------------------------------------------------------------------
# 3. Pattern preservation
# ---------------------------------------------------------------------------


class TestPatternPreservation:
    def test_patterns_preserved(self):
        pattern = DetectedPattern(
            pattern_type="suspicious_neighbors",
            description="2 suspicious neighbors found.",
            evidence={"count": 2},
            severity="MEDIUM",
        )
        result = _make_result(pattern_count=2)
        result.pattern_analysis = _make_pattern_result(
            txn_id="txn-001",
            patterns=[pattern],
            pattern_count=2,
            summary="2 patterns.",
        )
        report = report_generator.generate(result)
        assert report.detected_patterns.pattern_count == 2
        assert report.detected_patterns.patterns[0].pattern_type == "suspicious_neighbors"

    def test_pattern_severity_preserved(self):
        pattern = DetectedPattern(
            pattern_type="high_amount",
            description="High transaction amount.",
            evidence={"amount": 10000},
            severity="HIGH",
        )
        result = _make_result(pattern_count=1)
        result.pattern_analysis = _make_pattern_result(
            txn_id="txn-001",
            patterns=[pattern],
            pattern_count=1,
        )
        report = report_generator.generate(result)
        assert report.detected_patterns.patterns[0].severity == "HIGH"

    def test_pattern_count_preserved(self):
        result = _make_result(pattern_count=5)
        result.pattern_analysis = _make_pattern_result(
            txn_id="txn-001", pattern_count=5
        )
        report = report_generator.generate(result)
        assert report.detected_patterns.pattern_count == 5


# ---------------------------------------------------------------------------
# 4. Evidence preservation
# ---------------------------------------------------------------------------


class TestEvidencePreservation:
    def test_evidence_items_preserved(self):
        item = EvidenceItem(
            evidence_type="ml_prediction",
            source="ml_prediction",
            description="ML prediction available.",
            details={"fraud_probability": 0.85},
            available=True,
        )
        result = _make_result(evidence_count=1)
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            evidence=[item],
            evidence_count=1,
        )
        report = report_generator.generate(result)
        assert report.evidence.evidence_count == 1
        assert report.evidence.evidence[0].evidence_type == "ml_prediction"

    def test_evidence_source_preserved(self):
        item = EvidenceItem(
            evidence_type="graph_neighbor",
            source="graph_intelligence",
            description="Suspicious neighbor.",
            details={"transaction_id": "txn-nbr1"},
            available=True,
        )
        result = _make_result(evidence_count=1)
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            evidence=[item],
            evidence_count=1,
        )
        report = report_generator.generate(result)
        assert report.evidence.evidence[0].source == "graph_intelligence"

    def test_evidence_availability_preserved(self):
        result = _make_result(evidence_count=0)
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            availability={"graph": False, "cluster": False},
        )
        report = report_generator.generate(result)
        assert report.evidence.availability["graph"] is False
        assert report.evidence.availability["cluster"] is False

    def test_evidence_details_preserved(self):
        item = EvidenceItem(
            evidence_type="shap_factor",
            source="shap",
            description="Top risk factor.",
            details={"feature": "amount", "importance": 0.35},
            available=True,
        )
        result = _make_result(evidence_count=1)
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            evidence=[item],
            evidence_count=1,
        )
        report = report_generator.generate(result)
        assert report.evidence.evidence[0].details["feature"] == "amount"
        assert report.evidence.evidence[0].details["importance"] == 0.35


# ---------------------------------------------------------------------------
# 5. Conclusion
# ---------------------------------------------------------------------------


class TestConclusion:
    def test_conclusion_deterministic(self):
        result = _make_result()
        r1 = report_generator.generate(result)
        r2 = report_generator.generate(result)
        assert r1.conclusion == r2.conclusion

    def test_conclusion_uses_only_available_results(self):
        result = _make_result(risk_level="HIGH", risk_score=85.0)
        report = report_generator.generate(result)
        assert "HIGH" in report.conclusion
        assert "85.0" in report.conclusion

    def test_conclusion_no_unsupported_facts(self):
        result = _make_result()
        report = report_generator.generate(result)
        unsupported = [
            "criminal", "money laundering", "stolen identity",
            "fraud ring", "confirmed", "proven",
        ]
        for term in unsupported:
            assert term.lower() not in report.conclusion.lower()

    def test_conclusion_with_patterns(self):
        pattern = DetectedPattern(
            pattern_type="velocity_spike",
            description="High velocity.",
            evidence={},
            severity="HIGH",
        )
        result = _make_result(pattern_count=1)
        result.pattern_analysis = _make_pattern_result(
            txn_id="txn-001",
            patterns=[pattern],
            pattern_count=1,
        )
        report = report_generator.generate(result)
        assert "1 pattern(s) detected" in report.conclusion
        assert "velocity_spike" in report.conclusion

    def test_conclusion_with_no_patterns(self):
        result = _make_result(pattern_count=0)
        report = report_generator.generate(result)
        assert "No suspicious patterns detected" in report.conclusion


# ---------------------------------------------------------------------------
# 6. Missing evidence
# ---------------------------------------------------------------------------


class TestMissingEvidence:
    def test_missing_ml(self):
        result = InvestigationResult(
            transaction_id="txn-noml",
            risk_assessment=None,
            pattern_analysis=_make_pattern_result(txn_id="txn-noml"),
            evidence=_make_evidence_result(txn_id="txn-noml"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment is None
        assert "Risk assessment is unavailable" in report.conclusion

    def test_missing_graph(self):
        result = _make_result()
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            availability={"graph": False},
        )
        report = report_generator.generate(result)
        assert "Graph evidence unavailable" in report.conclusion

    def test_missing_shap(self):
        result = _make_result()
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            availability={"shap": False},
        )
        report = report_generator.generate(result)
        assert "SHAP evidence unavailable" in report.conclusion

    def test_missing_cluster(self):
        result = _make_result()
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            availability={"cluster": False},
        )
        report = report_generator.generate(result)
        assert "Cluster evidence unavailable" in report.conclusion

    def test_missing_network(self):
        result = _make_result()
        result.evidence = _make_evidence_result(
            txn_id="txn-001",
            availability={"network": False},
        )
        report = report_generator.generate(result)
        assert "Network risk evidence unavailable" in report.conclusion


# ---------------------------------------------------------------------------
# 7. Agent failures
# ---------------------------------------------------------------------------


class TestAgentFailures:
    def test_agent_error_exposed(self):
        errors = [AgentError(agent_name="RiskAgent", error_message="Risk agent crashed")]
        result = _make_result(agent_errors=errors)
        report = report_generator.generate(result)
        assert len(report.agent_errors) == 1
        assert report.agent_errors[0].agent_name == "RiskAgent"
        assert "Risk agent crashed" in report.agent_errors[0].error_message

    def test_agent_error_in_conclusion(self):
        errors = [AgentError(agent_name="PatternAgent", error_message="Pattern agent failed")]
        result = _make_result(agent_errors=errors)
        report = report_generator.generate(result)
        assert "Agent failure(s): PatternAgent" in report.conclusion

    def test_multiple_agent_errors(self):
        errors = [
            AgentError(agent_name="RiskAgent", error_message="crash"),
            AgentError(agent_name="EvidenceAgent", error_message="timeout"),
        ]
        result = _make_result(agent_errors=errors)
        report = report_generator.generate(result)
        assert len(report.agent_errors) == 2
        assert "Agent failure(s): RiskAgent, EvidenceAgent" in report.conclusion

    def test_agent_failure_does_not_fabricate_output(self):
        errors = [AgentError(agent_name="RiskAgent", error_message="crash")]
        result = InvestigationResult(
            transaction_id="txn-fail",
            risk_assessment=None,
            pattern_analysis=_make_pattern_result(txn_id="txn-fail"),
            evidence=_make_evidence_result(txn_id="txn-fail"),
            agent_errors=errors,
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment is None
        assert "Risk assessment is unavailable" in report.conclusion


# ---------------------------------------------------------------------------
# 8. Score integrity
# ---------------------------------------------------------------------------


class TestScoreIntegrity:
    def test_fraud_probability_not_modified(self):
        risk = _make_risk_result(risk_level="HIGH", risk_score=85.0)
        risk.evidence_summary = {"fraud_probability": 0.85}
        result = InvestigationResult(
            transaction_id="txn-score",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-score"),
            evidence=_make_evidence_result(txn_id="txn-score"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment.evidence_summary["fraud_probability"] == 0.85
        assert report.risk_assessment.risk_score == 85.0

    def test_risk_score_not_modified(self):
        risk = _make_risk_result(risk_level="MEDIUM", risk_score=55.0)
        result = InvestigationResult(
            transaction_id="txn-score2",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-score2"),
            evidence=_make_evidence_result(txn_id="txn-score2"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment.risk_score == 55.0
        assert report.risk_assessment.risk_level == "MEDIUM"

    def test_network_risk_score_not_modified(self):
        risk = _make_risk_result(risk_level="HIGH", risk_score=82.0)
        risk.evidence_summary = {
            "network_risk_score": 75.0,
            "combined_risk_score": 82.0,
        }
        result = InvestigationResult(
            transaction_id="txn-net",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-net"),
            evidence=_make_evidence_result(txn_id="txn-net"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment.evidence_summary["network_risk_score"] == 75.0
        assert report.risk_assessment.evidence_summary["combined_risk_score"] == 82.0

    def test_combined_risk_score_not_modified(self):
        risk = _make_risk_result(risk_level="MEDIUM", risk_score=62.0)
        risk.evidence_summary = {"combined_risk_score": 62.0}
        result = InvestigationResult(
            transaction_id="txn-combined",
            risk_assessment=risk,
            pattern_analysis=_make_pattern_result(txn_id="txn-combined"),
            evidence=_make_evidence_result(txn_id="txn-combined"),
            agent_errors=[],
            metadata={},
        )
        report = report_generator.generate(result)
        assert report.risk_assessment.evidence_summary["combined_risk_score"] == 62.0


# ---------------------------------------------------------------------------
# 9. Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    def test_metadata_preserved(self):
        result = _make_result()
        result.metadata = {"transaction_id": "txn-meta", "risk_agent_invoked": True}
        report = report_generator.generate(result)
        assert report.metadata["transaction_id"] == "txn-meta"
        assert report.metadata["risk_agent_invoked"] is True


# ---------------------------------------------------------------------------
# 10. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_investigation_report_valid_pydantic(self):
        report = InvestigationReport(
            transaction_id="test",
            risk_assessment=_make_risk_result(txn_id="test"),
            detected_patterns=_make_pattern_result(txn_id="test"),
            evidence=_make_evidence_result(txn_id="test"),
            conclusion="Test conclusion.",
            recommended_action="ALLOW",
            agent_errors=[],
            metadata={"key": "value"},
        )
        dumped = report.model_dump()
        assert dumped["transaction_id"] == "test"
        assert dumped["conclusion"] == "Test conclusion."
        assert dumped["recommended_action"] == "ALLOW"

    def test_investigation_report_with_none_agents(self):
        report = InvestigationReport(
            transaction_id="test-none",
            risk_assessment=None,
            detected_patterns=None,
            evidence=None,
            conclusion="Agent results unavailable.",
            recommended_action=None,
            agent_errors=[],
            metadata={},
        )
        dumped = report.model_dump()
        assert dumped["risk_assessment"] is None
        assert dumped["detected_patterns"] is None
        assert dumped["evidence"] is None
