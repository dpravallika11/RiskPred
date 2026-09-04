from typing import Any, Dict, List, Optional

from app.investigation.schemas import (
    InvestigationReport,
    InvestigationResult,
    RiskAgentResult,
    PatternAgentResult,
    EvidenceAgentResult,
    AgentError,
)


class InvestigationReportGenerator:
    """Generates a deterministic InvestigationReport from an InvestigationResult.

    This generator does NOT:
    - Calculate risk scores
    - Detect patterns
    - Collect evidence
    - Call external services
    - Use LLMs or generative text

    It produces a deterministic conclusion based solely on the structured
    results already present in the InvestigationResult.
    """

    def generate(self, result: InvestigationResult) -> InvestigationReport:
        conclusion = self._build_conclusion(
            result.risk_assessment,
            result.pattern_analysis,
            result.evidence,
            result.agent_errors,
        )

        recommended_action = self._extract_recommended_action(
            result.risk_assessment,
        )

        return InvestigationReport(
            transaction_id=result.transaction_id,
            risk_assessment=result.risk_assessment,
            detected_patterns=result.pattern_analysis,
            evidence=result.evidence,
            conclusion=conclusion,
            recommended_action=recommended_action,
            agent_errors=result.agent_errors,
            metadata=result.metadata,
        )

    # ------------------------------------------------------------------
    # Conclusion generation (deterministic, evidence-based)
    # ------------------------------------------------------------------

    def _build_conclusion(
        self,
        risk: Optional[RiskAgentResult],
        patterns: Optional[PatternAgentResult],
        evidence: Optional[EvidenceAgentResult],
        agent_errors: List[AgentError],
    ) -> str:
        parts: List[str] = []

        # Risk summary
        if risk is not None:
            parts.append(
                f"The transaction was assessed as {risk.risk_level} risk "
                f"(score {risk.risk_score:.1f})."
            )
        else:
            parts.append("Risk assessment is unavailable.")

        # Pattern summary
        if patterns is not None:
            if patterns.pattern_count > 0:
                pattern_types = [
                    p.pattern_type for p in patterns.patterns[:3]
                ]
                parts.append(
                    f"{patterns.pattern_count} pattern(s) detected: "
                    f"{', '.join(pattern_types)}."
                )
            else:
                parts.append("No suspicious patterns detected.")
        else:
            parts.append("Pattern analysis is unavailable.")

        # Evidence summary
        if evidence is not None:
            if evidence.evidence_count > 0:
                parts.append(
                    f"{evidence.evidence_count} evidence item(s) collected."
                )
            else:
                parts.append("No evidence items collected.")

            # Missing evidence notes
            availability_notes = self._build_availability_notes(evidence)
            if availability_notes:
                parts.append(availability_notes)
        else:
            parts.append("Evidence collection is unavailable.")

        # Agent errors
        if agent_errors:
            failed_agents = [e.agent_name for e in agent_errors]
            parts.append(
                f"Agent failure(s): {', '.join(failed_agents)}."
            )

        return " ".join(parts)

    def _build_availability_notes(self, evidence: EvidenceAgentResult) -> str:
        notes: List[str] = []
        availability = evidence.availability

        if not availability:
            return ""

        if availability.get("graph") is False:
            notes.append("Graph evidence unavailable.")
        if availability.get("cluster") is False:
            notes.append("Cluster evidence unavailable.")
        if availability.get("shap") is False:
            notes.append("SHAP evidence unavailable.")
        if availability.get("network") is False:
            notes.append("Network risk evidence unavailable.")

        return " ".join(notes)

    # ------------------------------------------------------------------
    # Recommended action extraction
    # ------------------------------------------------------------------

    def _extract_recommended_action(
        self, risk: Optional[RiskAgentResult]
    ) -> Optional[str]:
        if risk is None:
            return None
        return risk.evidence_summary.get("recommended_action")


report_generator = InvestigationReportGenerator()
