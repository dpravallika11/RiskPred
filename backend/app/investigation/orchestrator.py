from typing import Any, Dict, Optional

from app.investigation.schemas import (
    AgentError,
    InvestigationContext,
    InvestigationResult,
    RiskAgentResult,
    PatternAgentResult,
    EvidenceAgentResult,
)
from app.investigation.risk_agent import RiskAgent, risk_agent
from app.investigation.pattern_agent import PatternAgent, pattern_agent
from app.investigation.evidence_agent import EvidenceAgent, evidence_agent


class InvestigationOrchestrator:
    """Coordinates risk, pattern, and evidence agents for a single investigation.

    This orchestrator is responsible for:
    - Invoking the Risk Agent, Pattern Agent, and Evidence Agent
    - Collecting their structured results
    - Aggregating them into a combined InvestigationResult
    - Handling agent failures without fabricating results

    The orchestrator does NOT calculate risk scores, detect patterns, or
    collect evidence independently. It delegates entirely to the agents.
    """

    def __init__(
        self,
        risk_agent_instance: Optional[RiskAgent] = None,
        pattern_agent_instance: Optional[PatternAgent] = None,
        evidence_agent_instance: Optional[EvidenceAgent] = None,
    ):
        self._risk_agent = risk_agent_instance or risk_agent
        self._pattern_agent = pattern_agent_instance or pattern_agent
        self._evidence_agent = evidence_agent_instance or evidence_agent

    def investigate(self, context: InvestigationContext) -> InvestigationResult:
        """Run the full investigation pipeline on a given context.

        Each agent is invoked exactly once. Agent failures are captured
        as structured errors rather than silently swallowed.
        """
        risk_result: Optional[RiskAgentResult] = None
        pattern_result: Optional[PatternAgentResult] = None
        evidence_result: Optional[EvidenceAgentResult] = None
        agent_errors: list[AgentError] = []

        risk_result = self._invoke_agent(
            "RiskAgent", self._risk_agent.analyze, context, agent_errors
        )

        pattern_result = self._invoke_agent(
            "PatternAgent", self._pattern_agent.analyze, context, agent_errors
        )

        evidence_result = self._invoke_agent(
            "EvidenceAgent", self._evidence_agent.analyze, context, agent_errors
        )

        metadata = self._build_metadata(context, risk_result, pattern_result, evidence_result)

        return InvestigationResult(
            transaction_id=context.transaction_id,
            risk_assessment=risk_result,
            pattern_analysis=pattern_result,
            evidence=evidence_result,
            agent_errors=agent_errors,
            metadata=metadata,
        )

    def _invoke_agent(
        self,
        agent_name: str,
        agent_fn,
        context: InvestigationContext,
        agent_errors: list[AgentError],
    ) -> Any:
        """Invoke a single agent, capturing any exceptions as structured errors."""
        try:
            return agent_fn(context)
        except Exception as exc:
            agent_errors.append(AgentError(
                agent_name=agent_name,
                error_message=str(exc),
            ))
            return None

    def _build_metadata(
        self,
        context: InvestigationContext,
        risk_result: Optional[RiskAgentResult],
        pattern_result: Optional[PatternAgentResult],
        evidence_result: Optional[EvidenceAgentResult],
    ) -> Dict[str, Any]:
        """Build metadata summarizing the orchestration outcome."""
        metadata: Dict[str, Any] = {
            "transaction_id": context.transaction_id,
            "risk_agent_invoked": risk_result is not None,
            "pattern_agent_invoked": pattern_result is not None,
            "evidence_agent_invoked": evidence_result is not None,
        }

        if risk_result is not None:
            metadata["risk_level"] = risk_result.risk_level
            metadata["risk_score"] = risk_result.risk_score

        if pattern_result is not None:
            metadata["pattern_count"] = pattern_result.pattern_count

        if evidence_result is not None:
            metadata["evidence_count"] = evidence_result.evidence_count

        return metadata


orchestrator = InvestigationOrchestrator()
