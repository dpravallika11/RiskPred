from typing import Any, Dict, Optional

from app.investigation.schemas import (
    InvestigationContext,
    MLPredictionEvidence,
    SHAPEvidence,
    GraphEvidence,
    NetworkRiskEvidence,
    ClusterEvidence,
)
from app.services.prediction_service import prediction_service
from app.graph.graph_service import graph_service


class InvestigationContextService:
    """Assembles an InvestigationContext by coordinating existing services.

    This service does NOT reimplement ML prediction, SHAP, graph construction,
    network risk calculation, or cluster detection. It delegates to the existing
    services and assembles their outputs into a single structured context.
    """

    def build_context(
        self,
        transaction_id: str,
        transaction: Optional[Dict[str, Any]] = None,
    ) -> InvestigationContext:
        ml_prediction = self._get_ml_prediction(transaction)
        shap_explanation = self._get_shap_evidence(transaction)
        graph_evidence = self._get_graph_evidence(transaction_id)
        network_risk = self._get_network_risk(transaction_id, ml_prediction)
        cluster = self._get_cluster_evidence(transaction_id)

        return InvestigationContext(
            transaction_id=transaction_id,
            transaction=transaction,
            ml_prediction=ml_prediction,
            shap_explanation=shap_explanation,
            graph=graph_evidence,
            network_risk=network_risk,
            cluster=cluster,
        )

    def _get_ml_prediction(
        self, transaction: Optional[Dict[str, Any]]
    ) -> Optional[MLPredictionEvidence]:
        if transaction is None or not prediction_service.is_ready:
            return None
        try:
            result = prediction_service.predict(transaction)
            return MLPredictionEvidence(
                fraud_probability=result["fraud_probability"],
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                recommended_action=result["recommended_action"],
            )
        except Exception:
            return None

    def _get_shap_evidence(
        self, transaction: Optional[Dict[str, Any]]
    ) -> Optional[SHAPEvidence]:
        if transaction is None or not prediction_service.is_ready:
            return None
        try:
            result = prediction_service.predict(transaction)
            return SHAPEvidence(
                risk_factors=result.get("top_risk_factors", []),
                risk_reducers=result.get("top_risk_reducers", []),
            )
        except Exception:
            return None

    def _get_graph_evidence(self, transaction_id: str) -> GraphEvidence:
        if not graph_service.is_ready:
            return GraphEvidence(graph_available=False)

        try:
            connections = graph_service.get_connected_transactions(transaction_id)
        except Exception:
            connections = {"connected_transactions": [], "total_connections": 0}

        try:
            entities = graph_service.get_transaction_entities(transaction_id)
        except Exception:
            entities = []

        try:
            neighborhood = graph_service.get_neighborhood(transaction_id)
        except Exception:
            neighborhood = {"nodes": [], "edges": []}

        try:
            neighborhood_risk = graph_service.get_neighborhood_risk(transaction_id)
        except Exception:
            neighborhood_risk = {
                "suspicious_neighbors": [],
                "suspicious_neighbor_count": 0,
                "shared_entity_types": [],
            }

        return GraphEvidence(
            connected_transactions=connections.get("connected_transactions", []),
            total_connections=connections.get("total_connections", 0),
            entities=entities,
            entity_count=len(entities),
            neighborhood_nodes=neighborhood.get("nodes", []),
            neighborhood_edges=neighborhood.get("edges", []),
            suspicious_neighbors=neighborhood_risk.get("suspicious_neighbors", []),
            suspicious_neighbor_count=neighborhood_risk.get("suspicious_neighbor_count", 0),
            shared_entity_types=neighborhood_risk.get("shared_entity_types", []),
            graph_available=True,
        )

    def _get_network_risk(
        self,
        transaction_id: str,
        ml_prediction: Optional[MLPredictionEvidence],
    ) -> Optional[NetworkRiskEvidence]:
        if not graph_service.is_ready:
            return None

        ml_risk_score = ml_prediction.risk_score if ml_prediction else 0
        ml_risk_level = ml_prediction.risk_level if ml_prediction else "UNKNOWN"

        try:
            result = graph_service.get_network_risk(
                transaction_id, ml_risk_score, ml_risk_level
            )
            return NetworkRiskEvidence(
                network_risk_score=result.get("network_risk_score", 0),
                network_risk_level=result.get("network_risk_level", "UNKNOWN"),
                combined_risk_score=result.get("combined_risk_score", 0),
                combined_risk_level=result.get("combined_risk_level", "UNKNOWN"),
                factors=result.get("factors", []),
                neighbor_count=result.get("neighbor_count", 0),
                suspicious_neighbor_count=result.get("suspicious_neighbor_count", 0),
            )
        except Exception:
            return None

    def _get_cluster_evidence(
        self, transaction_id: str
    ) -> Optional[ClusterEvidence]:
        if not graph_service.is_ready:
            return None

        try:
            result = graph_service.get_cluster_for_transaction(transaction_id)
        except Exception:
            result = None

        if result is None:
            return None

        return ClusterEvidence(
            found=True,
            transaction_ids=result.get("transaction_ids", []),
            total_transactions=result.get("total_transactions", 0),
            entity_count=result.get("entity_count", 0),
            entity_types=result.get("entity_types", []),
            suspicious_transaction_count=result.get("suspicious_transaction_count", 0),
            suspicious_ratio=result.get("suspicious_ratio", 0.0),
            shared_identifiers=result.get("shared_identifiers", {}),
            risk_level=result.get("risk_level", "UNKNOWN"),
            avg_risk_score=result.get("avg_risk_score", 0.0),
            strong_entity_types=result.get("strong_entity_types", []),
            weak_entity_types=result.get("weak_entity_types", []),
        )


investigation_service = InvestigationContextService()
