from typing import Any, Dict, List, Optional

from app.investigation.schemas import (
    EvidenceAgentResult,
    EvidenceItem,
    InvestigationContext,
)


class EvidenceAgent:
    """Deterministic evidence collection agent.

    Analyzes an InvestigationContext and produces a structured collection of
    traceable evidence supporting the investigation. This agent organizes
    evidence already present in the context — it does not invent evidence,
    create competing risk scores, or call external services.
    """

    def analyze(self, context: InvestigationContext) -> EvidenceAgentResult:
        evidence: List[EvidenceItem] = []

        self._collect_transaction_evidence(context, evidence)
        self._collect_ml_evidence(context, evidence)
        self._collect_shap_evidence(context, evidence)
        self._collect_graph_evidence(context, evidence)
        self._collect_network_risk_evidence(context, evidence)
        self._collect_cluster_evidence(context, evidence)

        availability = self._build_availability(context)
        summary = self._build_summary(evidence, context)

        return EvidenceAgentResult(
            transaction_id=context.transaction_id,
            evidence=evidence,
            evidence_count=len(evidence),
            summary=summary,
            availability=availability,
        )

    # ------------------------------------------------------------------
    # Transaction evidence
    # ------------------------------------------------------------------

    def _collect_transaction_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        txn = context.transaction

        if txn is None:
            evidence.append(EvidenceItem(
                evidence_type="transaction",
                source="transaction",
                description="Transaction data is unavailable.",
                details={},
                available=False,
            ))
            return

        details: Dict[str, Any] = {}
        for key, value in txn.items():
            details[key] = value

        evidence.append(EvidenceItem(
            evidence_type="transaction",
            source="transaction",
            description=f"Transaction {context.transaction_id} data.",
            details=details,
            available=True,
        ))

    # ------------------------------------------------------------------
    # ML prediction evidence
    # ------------------------------------------------------------------

    def _collect_ml_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        ml = context.ml_prediction

        if ml is None:
            evidence.append(EvidenceItem(
                evidence_type="ml_prediction",
                source="ml_prediction",
                description="ML prediction evidence is unavailable.",
                details={},
                available=False,
            ))
            return

        evidence.append(EvidenceItem(
            evidence_type="ml_prediction",
            source="ml_prediction",
            description=(
                f"ML model predicts fraud probability {ml.fraud_probability:.4f} "
                f"with risk score {ml.risk_score} (level: {ml.risk_level}). "
                f"Recommended action: {ml.recommended_action}."
            ),
            details={
                "fraud_probability": ml.fraud_probability,
                "risk_score": ml.risk_score,
                "risk_level": ml.risk_level,
                "recommended_action": ml.recommended_action,
            },
            available=True,
        ))

    # ------------------------------------------------------------------
    # SHAP explanation evidence
    # ------------------------------------------------------------------

    def _collect_shap_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        shap = context.shap_explanation

        if shap is None:
            evidence.append(EvidenceItem(
                evidence_type="shap",
                source="shap",
                description="SHAP explainability evidence is unavailable.",
                details={},
                available=False,
            ))
            return

        factor_count = len(shap.risk_factors)
        reducer_count = len(shap.risk_reducers)

        details: Dict[str, Any] = {
            "risk_factors": list(shap.risk_factors),
            "risk_reducers": list(shap.risk_reducers),
            "risk_factor_count": factor_count,
            "risk_reducer_count": reducer_count,
        }

        description_parts: List[str] = []
        if factor_count > 0:
            names = [
                f.get("feature", f.get("name", "unknown"))
                for f in shap.risk_factors[:3]
            ]
            description_parts.append(
                f"{factor_count} risk factor(s) identified ({', '.join(names)})"
            )
        if reducer_count > 0:
            names = [
                f.get("feature", f.get("name", "unknown"))
                for f in shap.risk_reducers[:3]
            ]
            description_parts.append(
                f"{reducer_count} risk reducer(s) identified ({', '.join(names)})"
            )

        if not description_parts:
            description = "SHAP analysis available but no significant factors or reducers found."
        else:
            description = "SHAP analysis: " + "; ".join(description_parts) + "."

        evidence.append(EvidenceItem(
            evidence_type="shap",
            source="shap",
            description=description,
            details=details,
            available=True,
        ))

    # ------------------------------------------------------------------
    # Graph evidence
    # ------------------------------------------------------------------

    def _collect_graph_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        graph = context.graph

        if not graph.graph_available:
            evidence.append(EvidenceItem(
                evidence_type="graph",
                source="graph",
                description="Graph evidence is unavailable — transaction graph has not been built.",
                details={},
                available=False,
            ))
            return

        details: Dict[str, Any] = {
            "total_connections": graph.total_connections,
            "entity_count": graph.entity_count,
            "suspicious_neighbor_count": graph.suspicious_neighbor_count,
            "shared_entity_types": list(graph.shared_entity_types),
            "neighborhood_node_count": len(graph.neighborhood_nodes),
            "neighborhood_edge_count": len(graph.neighborhood_edges),
        }

        if graph.connected_transactions:
            details["connected_transactions"] = list(graph.connected_transactions)
        if graph.entities:
            details["entities"] = list(graph.entities)
        if graph.suspicious_neighbors:
            details["suspicious_neighbors"] = list(graph.suspicious_neighbors)

        description = (
            f"Graph has {graph.total_connections} connection(s), "
            f"{graph.entity_count} entity/entities, "
            f"{graph.suspicious_neighbor_count} suspicious neighbor(s)."
        )

        evidence.append(EvidenceItem(
            evidence_type="graph",
            source="graph",
            description=description,
            details=details,
            available=True,
        ))

    # ------------------------------------------------------------------
    # Network risk evidence
    # ------------------------------------------------------------------

    def _collect_network_risk_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        graph = context.graph
        network = context.network_risk

        if not graph.graph_available:
            return

        if network is None:
            evidence.append(EvidenceItem(
                evidence_type="network_risk",
                source="network_risk",
                description="Network risk calculation is unavailable despite graph evidence being present.",
                details={},
                available=False,
            ))
            return

        details: Dict[str, Any] = {
            "network_risk_score": network.network_risk_score,
            "network_risk_level": network.network_risk_level,
            "combined_risk_score": network.combined_risk_score,
            "combined_risk_level": network.combined_risk_level,
            "neighbor_count": network.neighbor_count,
            "suspicious_neighbor_count": network.suspicious_neighbor_count,
        }

        if network.factors:
            details["factors"] = list(network.factors)

        description = (
            f"Network risk score: {network.network_risk_score:.1f} "
            f"(level: {network.network_risk_level}). "
            f"Combined risk score: {network.combined_risk_score:.1f} "
            f"(level: {network.combined_risk_level})."
        )

        evidence.append(EvidenceItem(
            evidence_type="network_risk",
            source="network_risk",
            description=description,
            details=details,
            available=True,
        ))

    # ------------------------------------------------------------------
    # Cluster evidence
    # ------------------------------------------------------------------

    def _collect_cluster_evidence(
        self, context: InvestigationContext, evidence: List[EvidenceItem]
    ) -> None:
        graph = context.graph
        cluster = context.cluster

        if not graph.graph_available:
            return

        if cluster is None or not cluster.found:
            evidence.append(EvidenceItem(
                evidence_type="cluster",
                source="cluster",
                description="Cluster evidence is unavailable for this transaction.",
                details={},
                available=False,
            ))
            return

        details: Dict[str, Any] = {
            "total_transactions": cluster.total_transactions,
            "entity_count": cluster.entity_count,
            "entity_types": list(cluster.entity_types),
            "suspicious_transaction_count": cluster.suspicious_transaction_count,
            "suspicious_ratio": cluster.suspicious_ratio,
            "risk_level": cluster.risk_level,
            "avg_risk_score": cluster.avg_risk_score,
        }

        if cluster.transaction_ids:
            details["transaction_ids"] = list(cluster.transaction_ids)
        if cluster.shared_identifiers:
            details["shared_identifiers"] = {
                k: list(v) for k, v in cluster.shared_identifiers.items()
            }
        if cluster.strong_entity_types:
            details["strong_entity_types"] = list(cluster.strong_entity_types)
        if cluster.weak_entity_types:
            details["weak_entity_types"] = list(cluster.weak_entity_types)

        description = (
            f"Cluster of {cluster.total_transactions} transaction(s) "
            f"with risk level {cluster.risk_level}. "
            f"{cluster.suspicious_transaction_count}/{cluster.total_transactions} "
            f"suspicious (ratio: {cluster.suspicious_ratio:.2f})."
        )

        evidence.append(EvidenceItem(
            evidence_type="cluster",
            source="cluster",
            description=description,
            details=details,
            available=True,
        ))

    # ------------------------------------------------------------------
    # Availability summary
    # ------------------------------------------------------------------

    def _build_availability(self, context: InvestigationContext) -> Dict[str, bool]:
        graph = context.graph
        cluster = context.cluster

        availability: Dict[str, bool] = {
            "transaction": context.transaction is not None,
            "ml_prediction": context.ml_prediction is not None,
            "shap": context.shap_explanation is not None,
            "graph": graph.graph_available,
            "network_risk": graph.graph_available and context.network_risk is not None,
            "cluster": graph.graph_available and cluster is not None and cluster.found,
        }

        return availability

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self, evidence: List[EvidenceItem], context: InvestigationContext
    ) -> str:
        available_count = sum(1 for e in evidence if e.available)
        unavailable_count = sum(1 for e in evidence if not e.available)

        parts: List[str] = [
            f"{len(evidence)} evidence item(s) collected for transaction "
            f"{context.transaction_id}."
        ]

        if available_count > 0:
            parts.append(f"{available_count} available.")
        if unavailable_count > 0:
            parts.append(f"{unavailable_count} unavailable.")

        return " ".join(parts)


evidence_agent = EvidenceAgent()
