from typing import Any, Dict, List, Optional

from app.investigation.schemas import (
    InvestigationContext,
    RiskAgentResult,
)


class RiskAgent:
    """Deterministic risk assessment agent.

    Analyzes an InvestigationContext and produces a structured risk assessment.
    This agent interprets existing computed risk scores and evidence — it does
    not create competing scoring formulas, train models, or call external services.
    """

    def analyze(self, context: InvestigationContext) -> RiskAgentResult:
        ml = context.ml_prediction
        shap = context.shap_explanation
        graph = context.graph
        network = context.network_risk
        cluster = context.cluster

        risk_score = self._resolve_risk_score(ml, network)
        risk_level = self._resolve_risk_level(ml, network)
        assessment = self._build_assessment(risk_level, risk_score, ml, network, cluster)
        reasons = self._build_reasons(ml, shap, graph, network, cluster)
        risk_factors = self._get_risk_factors(shap)
        risk_reducers = self._get_risk_reducers(shap)
        evidence_summary = self._build_evidence_summary(ml, shap, graph, network, cluster)

        return RiskAgentResult(
            transaction_id=context.transaction_id,
            risk_level=risk_level,
            risk_score=risk_score,
            assessment=assessment,
            reasons=reasons,
            risk_factors=risk_factors,
            risk_reducers=risk_reducers,
            evidence_summary=evidence_summary,
        )

    # -- Risk score / level resolution ---------------------------------------

    def _resolve_risk_score(
        self,
        ml: Optional[Any],
        network: Optional[Any],
    ) -> float:
        if network is not None and network.combined_risk_score > 0:
            return network.combined_risk_score
        if ml is not None:
            return float(ml.risk_score)
        return 0.0

    def _resolve_risk_level(
        self,
        ml: Optional[Any],
        network: Optional[Any],
    ) -> str:
        if network is not None and network.combined_risk_level not in ("UNKNOWN", ""):
            return network.combined_risk_level
        if ml is not None:
            return ml.risk_level
        return "UNKNOWN"

    # -- Assessment text -----------------------------------------------------

    def _build_assessment(
        self,
        risk_level: str,
        risk_score: float,
        ml: Optional[Any],
        network: Optional[Any],
        cluster: Optional[Any],
    ) -> str:
        parts: List[str] = []

        if risk_level == "UNKNOWN":
            parts.append("Insufficient evidence to determine risk level.")
        elif risk_level == "HIGH":
            parts.append(f"Transaction is assessed as HIGH risk (score {risk_score:.1f}).")
        elif risk_level == "MEDIUM":
            parts.append(f"Transaction is assessed as MEDIUM risk (score {risk_score:.1f}).")
        else:
            parts.append(f"Transaction is assessed as LOW risk (score {risk_score:.1f}).")

        if ml is not None:
            parts.append(
                f"ML model recommends: {ml.recommended_action}."
            )

        if network is not None and network.suspicious_neighbor_count > 0:
            parts.append(
                f"{network.suspicious_neighbor_count} suspicious neighbor(s) identified."
            )

        if cluster is not None and cluster.found:
            parts.append(
                f"Transaction belongs to a cluster of {cluster.total_transactions} "
                f"transaction(s) with cluster risk level {cluster.risk_level}."
            )

        return " ".join(parts)

    # -- Reason generation ---------------------------------------------------

    def _build_reasons(
        self,
        ml: Optional[Any],
        shap: Optional[Any],
        graph: Any,
        network: Optional[Any],
        cluster: Optional[Any],
    ) -> List[str]:
        reasons: List[str] = []

        self._add_ml_reasons(reasons, ml)
        self._add_shap_reasons(reasons, shap)
        self._add_network_reasons(reasons, graph, network)
        self._add_cluster_reasons(reasons, cluster)
        self._add_missing_evidence_notes(reasons, ml, shap, graph, network, cluster)

        return reasons

    def _add_ml_reasons(self, reasons: List[str], ml: Optional[Any]) -> None:
        if ml is None:
            return

        if ml.risk_level == "HIGH":
            reasons.append(
                f"High ML risk score ({ml.risk_score}) indicates elevated transaction risk."
            )
        elif ml.risk_level == "MEDIUM":
            reasons.append(
                f"Moderate ML risk score ({ml.risk_score}) suggests some risk indicators."
            )
        elif ml.risk_level == "LOW":
            reasons.append(
                f"Low ML risk score ({ml.risk_score}) indicates minimal transaction risk."
            )

    def _add_shap_reasons(self, reasons: List[str], shap: Optional[Any]) -> None:
        if shap is None:
            return

        factor_count = len(shap.risk_factors)
        reducer_count = len(shap.risk_reducers)

        if factor_count > 0:
            names = [
                f.get("feature", f.get("name", "unknown"))
                for f in shap.risk_factors[:3]
            ]
            reasons.append(
                f"SHAP analysis identified {factor_count} risk factor(s): "
                f"{', '.join(names)}."
            )

        if reducer_count > 0:
            names = [
                f.get("feature", f.get("name", "unknown"))
                for f in shap.risk_reducers[:3]
            ]
            reasons.append(
                f"SHAP analysis identified {reducer_count} risk reducer(s): "
                f"{', '.join(names)}."
            )

    def _add_network_reasons(
        self,
        reasons: List[str],
        graph: Any,
        network: Optional[Any],
    ) -> None:
        if not graph.graph_available:
            return

        if network is None:
            return

        if network.suspicious_neighbor_count > 0:
            reasons.append(
                f"{network.suspicious_neighbor_count} suspicious neighboring "
                f"transaction(s) strengthen the network-risk assessment."
            )

        if network.network_risk_level == "HIGH":
            reasons.append(
                f"Network risk level is HIGH (score {network.network_risk_score:.1f})."
            )
        elif network.network_risk_level == "MEDIUM":
            reasons.append(
                f"Network risk level is MEDIUM (score {network.network_risk_score:.1f})."
            )

    def _add_cluster_reasons(self, reasons: List[str], cluster: Optional[Any]) -> None:
        if cluster is None or not cluster.found:
            return

        if cluster.risk_level == "HIGH":
            reasons.append(
                f"Transaction belongs to a HIGH-risk fraud cluster "
                f"({cluster.suspicious_transaction_count}/{cluster.total_transactions} "
                f"suspicious, ratio {cluster.suspicious_ratio:.2f})."
            )
        elif cluster.risk_level == "MEDIUM":
            reasons.append(
                f"Transaction belongs to a MEDIUM-risk cluster "
                f"({cluster.suspicious_transaction_count}/{cluster.total_transactions} "
                f"suspicious, ratio {cluster.suspicious_ratio:.2f})."
            )

    def _add_missing_evidence_notes(
        self,
        reasons: List[str],
        ml: Optional[Any],
        shap: Optional[Any],
        graph: Any,
        network: Optional[Any],
        cluster: Optional[Any],
    ) -> None:
        if ml is None:
            reasons.append("ML prediction evidence is unavailable.")

        if shap is None:
            reasons.append("SHAP explainability evidence is unavailable.")

        if not graph.graph_available:
            reasons.append("Graph/network evidence is unavailable — transaction graph has not been built.")

        if graph.graph_available and network is None:
            reasons.append("Network risk calculation is unavailable despite graph evidence being present.")

        if cluster is None and graph.graph_available:
            reasons.append("Cluster evidence is unavailable for this transaction.")

    # -- Risk factors / reducers pass-through --------------------------------

    def _get_risk_factors(self, shap: Optional[Any]) -> List[Dict[str, Any]]:
        if shap is None:
            return []
        return list(shap.risk_factors)

    def _get_risk_reducers(self, shap: Optional[Any]) -> List[Dict[str, Any]]:
        if shap is None:
            return []
        return list(shap.risk_reducers)

    # -- Evidence summary ----------------------------------------------------

    def _build_evidence_summary(
        self,
        ml: Optional[Any],
        shap: Optional[Any],
        graph: Any,
        network: Optional[Any],
        cluster: Optional[Any],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "ml_available": ml is not None,
            "shap_available": shap is not None,
            "graph_available": graph.graph_available,
            "network_risk_available": network is not None,
            "cluster_available": cluster is not None and cluster.found,
        }

        if ml is not None:
            summary["ml_risk_score"] = ml.risk_score
            summary["ml_risk_level"] = ml.risk_level
            summary["fraud_probability"] = ml.fraud_probability
            summary["recommended_action"] = ml.recommended_action

        if shap is not None:
            summary["risk_factor_count"] = len(shap.risk_factors)
            summary["risk_reducer_count"] = len(shap.risk_reducers)

        if graph.graph_available:
            summary["total_connections"] = graph.total_connections
            summary["entity_count"] = graph.entity_count
            summary["suspicious_neighbor_count"] = graph.suspicious_neighbor_count

        if network is not None:
            summary["network_risk_score"] = network.network_risk_score
            summary["network_risk_level"] = network.network_risk_level
            summary["combined_risk_score"] = network.combined_risk_score
            summary["combined_risk_level"] = network.combined_risk_level

        if cluster is not None and cluster.found:
            summary["cluster_total_transactions"] = cluster.total_transactions
            summary["cluster_suspicious_count"] = cluster.suspicious_transaction_count
            summary["cluster_suspicious_ratio"] = cluster.suspicious_ratio
            summary["cluster_risk_level"] = cluster.risk_level
            summary["cluster_avg_risk_score"] = cluster.avg_risk_score

        return summary


risk_agent = RiskAgent()
