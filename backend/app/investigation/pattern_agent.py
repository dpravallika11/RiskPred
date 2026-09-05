from typing import Any, Dict, List, Optional

from app.investigation.schemas import (
    DetectedPattern,
    InvestigationContext,
    PatternAgentResult,
)


class PatternAgent:
    """Deterministic pattern detection agent.

    Analyzes an InvestigationContext and identifies suspicious transaction/network
    patterns supported by available evidence. This agent performs pattern detection
    and description only — it does not make the final risk decision.

    All patterns are traceable to fields in InvestigationContext. No evidence is
    fabricated or hallucinated.
    """

    def analyze(self, context: InvestigationContext) -> PatternAgentResult:
        patterns: List[DetectedPattern] = []

        self._detect_suspicious_neighbor_patterns(context, patterns)
        self._detect_shared_entity_patterns(context, patterns)
        self._detect_cluster_patterns(context, patterns)
        self._detect_dense_connection_patterns(context, patterns)

        summary = self._build_summary(patterns, context)
        evidence_summary = self._build_evidence_summary(context)

        return PatternAgentResult(
            transaction_id=context.transaction_id,
            patterns=patterns,
            pattern_count=len(patterns),
            summary=summary,
            evidence_summary=evidence_summary,
        )

    # ------------------------------------------------------------------
    # Suspicious neighbor pattern
    # ------------------------------------------------------------------

    def _detect_suspicious_neighbor_patterns(
        self, context: InvestigationContext, patterns: List[DetectedPattern]
    ) -> None:
        graph = context.graph
        if not graph.graph_available:
            return

        network = context.network_risk
        suspicious_count = graph.suspicious_neighbor_count
        if network is not None:
            suspicious_count = max(suspicious_count, network.suspicious_neighbor_count)

        if suspicious_count <= 0:
            return

        if suspicious_count >= 3:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        evidence: Dict[str, Any] = {
            "suspicious_neighbor_count": suspicious_count,
            "total_connections": graph.total_connections,
        }

        if graph.suspicious_neighbors:
            neighbor_details = []
            for neighbor in graph.suspicious_neighbors:
                detail: Dict[str, Any] = {}
                if isinstance(neighbor, dict):
                    if "transaction_id" in neighbor:
                        detail["transaction_id"] = neighbor["transaction_id"]
                    if "fraud_probability" in neighbor:
                        detail["fraud_probability"] = neighbor["fraud_probability"]
                    if "risk_level" in neighbor:
                        detail["risk_level"] = neighbor["risk_level"]
                if detail:
                    neighbor_details.append(detail)
            if neighbor_details:
                evidence["suspicious_neighbor_details"] = neighbor_details

        description = (
            f"Transaction has {suspicious_count} suspicious neighboring "
            f"transaction(s) sharing entities in the transaction graph."
        )

        patterns.append(DetectedPattern(
            pattern_type="suspicious_neighbors",
            description=description,
            evidence=evidence,
            severity=severity,
        ))

    # ------------------------------------------------------------------
    # Shared entity pattern
    # ------------------------------------------------------------------

    def _detect_shared_entity_patterns(
        self, context: InvestigationContext, patterns: List[DetectedPattern]
    ) -> None:
        graph = context.graph
        if not graph.graph_available:
            return

        shared_types = list(graph.shared_entity_types)
        if not shared_types:
            return

        type_count = len(shared_types)

        if type_count >= 3:
            severity = "HIGH"
        elif type_count >= 2:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        evidence: Dict[str, Any] = {
            "shared_entity_types": shared_types,
            "entity_type_count": type_count,
            "entity_count": graph.entity_count,
            "connected_transaction_count": graph.total_connections,
        }

        description = (
            f"Transaction shares {type_count} entity type(s) "
            f"({', '.join(shared_types)}) with connected transactions."
        )

        patterns.append(DetectedPattern(
            pattern_type="shared_entities",
            description=description,
            evidence=evidence,
            severity=severity,
        ))

    # ------------------------------------------------------------------
    # Cluster pattern
    # ------------------------------------------------------------------

    def _detect_cluster_patterns(
        self, context: InvestigationContext, patterns: List[DetectedPattern]
    ) -> None:
        graph = context.graph
        if not graph.graph_available:
            return

        cluster = context.cluster
        if cluster is None or not cluster.found:
            return

        severity = cluster.risk_level if cluster.risk_level in ("HIGH", "MEDIUM", "LOW") else "UNKNOWN"

        evidence: Dict[str, Any] = {
            "cluster_transaction_count": cluster.total_transactions,
            "cluster_entity_count": cluster.entity_count,
            "cluster_entity_types": list(cluster.entity_types),
            "suspicious_transaction_count": cluster.suspicious_transaction_count,
            "suspicious_ratio": cluster.suspicious_ratio,
            "cluster_risk_level": cluster.risk_level,
            "avg_risk_score": cluster.avg_risk_score,
        }

        if cluster.shared_identifiers:
            evidence["shared_identifiers"] = {
                k: list(v) for k, v in cluster.shared_identifiers.items()
            }

        if cluster.strong_entity_types:
            evidence["strong_entity_types"] = list(cluster.strong_entity_types)

        description = (
            f"Transaction belongs to a cluster of {cluster.total_transactions} "
            f"transaction(s) connected through shared entities "
            f"({', '.join(cluster.entity_types)}). "
            f"{cluster.suspicious_transaction_count}/{cluster.total_transactions} "
            f"cluster transactions are flagged as suspicious."
        )

        patterns.append(DetectedPattern(
            pattern_type="cluster_membership",
            description=description,
            evidence=evidence,
            severity=severity,
        ))

    # ------------------------------------------------------------------
    # Dense connection pattern
    # ------------------------------------------------------------------

    def _detect_dense_connection_patterns(
        self, context: InvestigationContext, patterns: List[DetectedPattern]
    ) -> None:
        graph = context.graph
        if not graph.graph_available:
            return

        total = graph.total_connections
        if total < 3:
            return

        if total >= 5:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        evidence: Dict[str, Any] = {
            "total_connections": total,
            "entity_count": graph.entity_count,
            "neighborhood_node_count": len(graph.neighborhood_nodes),
            "neighborhood_edge_count": len(graph.neighborhood_edges),
        }

        description = (
            f"Transaction has {total} connected transaction(s) in the "
            f"graph, indicating dense connectivity."
        )

        patterns.append(DetectedPattern(
            pattern_type="dense_connections",
            description=description,
            evidence=evidence,
            severity=severity,
        ))

    # ------------------------------------------------------------------
    # Summary and evidence
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        patterns: List[DetectedPattern],
        context: InvestigationContext,
    ) -> str:
        if not patterns:
            graph = context.graph
            if not graph.graph_available:
                return "Graph evidence unavailable. No patterns can be assessed."
            return (
                f"No suspicious patterns detected for transaction "
                f"{context.transaction_id} based on available evidence."
            )

        high_count = sum(1 for p in patterns if p.severity == "HIGH")
        medium_count = sum(1 for p in patterns if p.severity == "MEDIUM")
        low_count = sum(1 for p in patterns if p.severity == "LOW")

        parts: List[str] = [
            f"{len(patterns)} suspicious pattern(s) detected for transaction "
            f"{context.transaction_id}."
        ]

        severity_parts: List[str] = []
        if high_count:
            severity_parts.append(f"{high_count} HIGH")
        if medium_count:
            severity_parts.append(f"{medium_count} MEDIUM")
        if low_count:
            severity_parts.append(f"{low_count} LOW")

        if severity_parts:
            parts.append("Severity breakdown: " + ", ".join(severity_parts) + ".")

        return " ".join(parts)

    def _build_evidence_summary(
        self, context: InvestigationContext
    ) -> Dict[str, Any]:
        graph = context.graph
        cluster = context.cluster

        summary: Dict[str, Any] = {
            "graph_available": graph.graph_available,
        }

        if graph.graph_available:
            summary["total_connections"] = graph.total_connections
            summary["entity_count"] = graph.entity_count
            summary["suspicious_neighbor_count"] = graph.suspicious_neighbor_count
            summary["shared_entity_types"] = list(graph.shared_entity_types)
            summary["neighborhood_node_count"] = len(graph.neighborhood_nodes)
            summary["neighborhood_edge_count"] = len(graph.neighborhood_edges)

        if cluster is not None and cluster.found:
            summary["cluster_available"] = True
            summary["cluster_total_transactions"] = cluster.total_transactions
            summary["cluster_risk_level"] = cluster.risk_level
            summary["cluster_suspicious_ratio"] = cluster.suspicious_ratio
        else:
            summary["cluster_available"] = False

        return summary


pattern_agent = PatternAgent()
