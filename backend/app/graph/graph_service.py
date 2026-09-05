import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from app.graph.entity_extractor import EntityExtractor
from app.graph.graph_builder import GraphBuilder
from app.graph.graph_queries import GraphQuerier
from app.graph.cluster_detector import ClusterDetector
from app.graph.network_risk import NetworkRiskCalculator, SUSPICIOUS_THRESHOLD

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(self):
        self._builder = GraphBuilder()
        self._querier: Optional[GraphQuerier] = None
        self._cluster_detector: Optional[ClusterDetector] = None
        self._network_risk: Optional[NetworkRiskCalculator] = None
        self._last_built: Optional[datetime] = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def last_built(self) -> Optional[datetime]:
        return self._last_built

    @property
    def transaction_count(self) -> int:
        return self._builder.transaction_count

    @property
    def entity_count(self) -> int:
        return self._builder.entity_count

    @property
    def edge_count(self) -> int:
        return self._builder.edge_count

    def build(self, transactions: List[Dict[str, Any]], risk_results: Optional[Dict[str, Dict[str, Any]]] = None, persist: bool = False):
        self._builder.build(transactions, risk_results)
        self._querier = GraphQuerier(self._builder.graph, self._builder.resolver)
        self._cluster_detector = ClusterDetector(self._builder.graph)
        self._network_risk = NetworkRiskCalculator(self._builder.graph)
        self._last_built = datetime.now(timezone.utc)
        self._is_ready = True

        if persist:
            self._persist_graph_to_db()

    def _persist_graph_to_db(self):
        """Persist the in-memory graph to Supabase. Failures are logged but do not affect the in-memory graph."""
        try:
            from app.db.repositories import entity_repo, graph_edge_repo
            self._builder.persist_graph(entity_repo, graph_edge_repo)
        except Exception as exc:
            logger.error("Graph persistence to DB failed: %s", exc)

    def add_transaction_risk(self, txn_id: str, risk: Dict[str, Any]):
        self._builder.add_risk_to_transaction(txn_id, risk)
        if self._network_risk is None:
            self._network_risk = NetworkRiskCalculator(self._builder.graph)

    def get_transaction_risk(self, txn_id: str) -> Optional[Dict[str, Any]]:
        """Public method to retrieve stored ML risk for a transaction."""
        self._ensure_ready()
        return self._builder.get_transaction_risk(txn_id)

    def get_connected_transactions(self, txn_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        connected = self._querier.get_connected_transactions(txn_id)
        return {
            "transaction_id": txn_id,
            "connected_transactions": connected,
            "total_connections": len(connected),
        }

    def get_transaction_entities(self, txn_id: str) -> List[Dict[str, str]]:
        self._ensure_ready()
        return self._querier.get_transaction_entities(txn_id)

    def get_neighborhood(self, txn_id: str, max_hops: int = 2) -> Dict[str, Any]:
        self._ensure_ready()
        return self._querier.get_neighborhood(txn_id, max_hops)

    def get_neighborhood_risk(self, txn_id: str) -> Dict[str, Any]:
        """Correctly identify suspicious neighbors based on their fraud probability."""
        self._ensure_ready()
        txn_risk = self._builder.get_transaction_risk(txn_id)
        if txn_risk:
            ml_risk_score = txn_risk.get("risk_score", 0)
            ml_risk_level = txn_risk.get("risk_level", "UNKNOWN")
        else:
            ml_risk_score = 0
            ml_risk_level = "UNKNOWN"

        connected = self._querier.get_connected_transactions(txn_id)

        suspicious_neighbors = []
        shared_entity_types_set = set()
        for conn in connected:
            neighbor_id = conn["transaction_id"]
            neighbor_risk = self._builder.get_transaction_risk(neighbor_id)
            neighbor_fp = neighbor_risk.get("fraud_probability", 0) if neighbor_risk else 0
            if neighbor_fp is not None and neighbor_fp >= SUSPICIOUS_THRESHOLD:
                suspicious_neighbors.append({
                    "transaction_id": neighbor_id,
                    "fraud_probability": neighbor_fp,
                    "risk_level": neighbor_risk.get("risk_level", "UNKNOWN") if neighbor_risk else "UNKNOWN",
                })
            for ent in conn.get("shared_entities", []):
                shared_entity_types_set.add(ent.get("type", "unknown"))

        return {
            "transaction_id": txn_id,
            "ml_risk_score": ml_risk_score,
            "ml_risk_level": ml_risk_level,
            "neighbor_count": len(connected),
            "suspicious_neighbor_count": len(suspicious_neighbors),
            "suspicious_neighbors": suspicious_neighbors,
            "shared_entity_types": list(shared_entity_types_set),
            "network_context_added": len(connected) > 0,
        }

    def get_clusters(self) -> Dict[str, Any]:
        self._ensure_ready()
        clusters = self._cluster_detector.detect_clusters()
        total_txn_in_clusters = sum(c["total_transactions"] for c in clusters)
        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "total_transactions_in_clusters": total_txn_in_clusters,
        }

    def get_cluster_for_transaction(self, txn_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_ready()
        return self._cluster_detector.get_cluster_for_transaction(txn_id)

    def get_network_risk(self, txn_id: str, ml_risk_score: float, ml_risk_level: str) -> Dict[str, Any]:
        self._ensure_ready()
        return self._network_risk.compute_combined_risk(txn_id, ml_risk_score, ml_risk_level)

    def get_suspicious_transactions(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        self._ensure_ready()
        return self._querier.get_suspicious_transactions(threshold)

    def clear(self):
        self._builder.clear()
        self._querier = None
        self._cluster_detector = None
        self._network_risk = None
        self._last_built = None
        self._is_ready = False

    def _ensure_ready(self):
        if not self._is_ready:
            raise RuntimeError("Graph not built. Call build() first.")


graph_service = GraphService()
