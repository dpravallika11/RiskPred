import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from app.graph.entity_extractor import EntityExtractor
from app.graph.graph_builder import GraphBuilder
from app.graph.graph_queries import GraphQuerier
from app.graph.cluster_detector import ClusterDetector
from app.graph.network_risk import NetworkRiskCalculator, SUSPICIOUS_THRESHOLD
from app.db.repositories import entity_repo, graph_edge_repo, prediction_repo

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

    def load_from_db(self) -> bool:
        """Load the persisted graph from Supabase and reconstruct the NetworkX graph.

        Returns True if the graph was successfully loaded, False otherwise.
        On failure, the graph remains in its current state (not-ready if freshly created).
        """
        try:
            # 1. Load persisted entities and graph edges from Supabase
            entities = entity_repo.get_all()
            graph_edges = graph_edge_repo.get_all()

            if not entities:
                logger.info("No persisted graph data found in Supabase.")
                return False

            # 2. Clear current in-memory state
            self._builder.clear()

            # 3. Build entity_id -> entity record lookup
            entity_by_id = {e["id"]: e for e in entities}

            # 4. Collect all unique transaction IDs from graph edges
            txn_ids = set()
            for edge in graph_edges:
                txn_ids.add(edge["transaction_id"])

            # 5. Add transaction nodes
            for txn_id in txn_ids:
                self._builder._graph.add_node(txn_id, node_type="transaction")

            # 6. Add entity nodes and edges
            for edge in graph_edges:
                txn_id = edge["transaction_id"]
                entity_id = edge["entity_id"]
                relationship = edge.get("relationship", "unknown")

                entity_record = entity_by_id.get(entity_id)
                if not entity_record:
                    continue

                node_key = entity_record["node_key"]
                entity_type = entity_record["entity_type"]
                entity_value = entity_record["entity_value"]

                if node_key not in self._builder._graph:
                    self._builder._graph.add_node(
                        node_key,
                        node_type=entity_type,
                        value=entity_value,
                    )

                self._builder._graph.add_edge(txn_id, node_key, relationship=relationship)

            # 7. Reconstruct EntityResolver internal maps
            for edge in graph_edges:
                txn_id = edge["transaction_id"]
                entity_id = edge["entity_id"]
                relationship = edge.get("relationship", "unknown")

                entity_record = entity_by_id.get(entity_id)
                if not entity_record:
                    continue

                node_key = entity_record["node_key"]

                # _reverse_map: node_key -> set of transaction_ids
                if node_key not in self._builder._resolver._reverse_map:
                    self._builder._resolver._reverse_map[node_key] = set()
                self._builder._resolver._reverse_map[node_key].add(txn_id)

                # _entity_map: transaction_id -> {entity_type: node_key}
                if txn_id not in self._builder._resolver._entity_map:
                    self._builder._resolver._entity_map[txn_id] = {}
                self._builder._resolver._entity_map[txn_id][relationship] = node_key

            # 8. Load risk data from predictions table
            try:
                recent_predictions = prediction_repo.get_recent(limit=500)
                for pred in recent_predictions:
                    txn_id = pred.get("transaction_id")
                    if txn_id and txn_id in self._builder._graph:
                        risk = {
                            "fraud_probability": pred.get("fraud_probability", 0),
                            "risk_score": pred.get("risk_score", 0),
                            "risk_level": pred.get("risk_level", "UNKNOWN"),
                        }
                        self._builder._transaction_risk[txn_id] = risk
                        self._builder._graph.nodes[txn_id]["fraud_probability"] = risk["fraud_probability"]
                        self._builder._graph.nodes[txn_id]["risk_score"] = risk["risk_score"]
                        self._builder._graph.nodes[txn_id]["risk_level"] = risk["risk_level"]
            except Exception as exc:
                logger.warning("Could not load prediction risk data: %s", exc)

            # 9. Wire up querier, cluster detector, and network risk calculator
            self._querier = GraphQuerier(self._builder.graph, self._builder.resolver)
            self._cluster_detector = ClusterDetector(self._builder.graph)
            self._network_risk = NetworkRiskCalculator(self._builder.graph)
            self._last_built = datetime.now(timezone.utc)
            self._is_ready = True

            logger.info(
                "Graph loaded from Supabase: %d transactions, %d entities, %d edges",
                self.transaction_count,
                self.entity_count,
                self.edge_count,
            )
            return True

        except Exception as exc:
            logger.error("Failed to load graph from Supabase: %s", exc)
            return False

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
