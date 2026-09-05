import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Set
from app.graph.entity_extractor import EntityExtractor
from app.graph.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self):
        self._graph = nx.Graph()
        self._extractor = EntityExtractor()
        self._resolver = EntityResolver()
        self._transaction_risk: Dict[str, Dict[str, Any]] = {}

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def resolver(self) -> EntityResolver:
        return self._resolver

    def build(self, transactions: List[Dict[str, Any]], risk_results: Optional[Dict[str, Dict[str, Any]]] = None):
        self._graph.clear()
        self._resolver.clear()
        self._transaction_risk.clear()

        for txn in transactions:
            txn_id = txn.get("transaction_id")
            if not txn_id:
                continue
            self._add_transaction_node(txn_id, txn)

            entities = self._extractor.extract(txn)
            for entity_type, values in entities.items():
                for value in values:
                    entity_key = self._resolver.link(entity_type, value, txn_id)
                    self._add_entity_node(entity_key, entity_type, value)
                    self._graph.add_edge(txn_id, entity_key, relationship=entity_type)

        if risk_results:
            for txn_id, risk in risk_results.items():
                self._transaction_risk[txn_id] = risk
                if txn_id in self._graph:
                    self._graph.nodes[txn_id]["fraud_probability"] = risk.get("fraud_probability", 0)
                    self._graph.nodes[txn_id]["risk_score"] = risk.get("risk_score", 0)
                    self._graph.nodes[txn_id]["risk_level"] = risk.get("risk_level", "UNKNOWN")

    def _add_transaction_node(self, txn_id: str, txn_dict: Dict[str, Any]):
        self._graph.add_node(
            txn_id,
            node_type="transaction",
            amount=txn_dict.get("amount", 0),
            merchant_id=txn_dict.get("merchant_id"),
            customer_id=txn_dict.get("customer_id"),
        )

    def _add_entity_node(self, entity_key: str, entity_type: str, value: str):
        if entity_key not in self._graph:
            self._graph.add_node(entity_key, node_type=entity_type, value=value)

    def add_risk_to_transaction(self, txn_id: str, risk: Dict[str, Any]):
        self._transaction_risk[txn_id] = risk
        if txn_id in self._graph:
            self._graph.nodes[txn_id]["fraud_probability"] = risk.get("fraud_probability", 0)
            self._graph.nodes[txn_id]["risk_score"] = risk.get("risk_score", 0)
            self._graph.nodes[txn_id]["risk_level"] = risk.get("risk_level", "UNKNOWN")

    def get_transaction_risk(self, txn_id: str) -> Optional[Dict[str, Any]]:
        return self._transaction_risk.get(txn_id)

    def clear(self):
        self._graph.clear()
        self._resolver.clear()
        self._transaction_risk.clear()
        self._extractor = EntityExtractor()

    def persist_graph(self, entity_repo, graph_edge_repo) -> bool:
        """Persist the in-memory graph to Supabase.

        Args:
            entity_repo: EntityRepository instance for upserting entities and creating links.
            graph_edge_repo: GraphEdgeRepository instance for creating graph edges.

        Returns:
            True if all persistence operations succeeded, False otherwise.
        """
        try:
            # 1. Clean previous graph data
            graph_edge_repo.delete_all()
            entity_repo.delete_all()

            # 2. Upsert entities
            node_key_to_entity_id: Dict[str, str] = {}
            for entity_key, data in self._graph.nodes(data=True):
                if data.get("node_type") == "transaction":
                    continue
                entity_type = data.get("node_type", "unknown")
                entity_value = data.get("value", entity_key.split(":", 1)[-1] if ":" in entity_key else entity_key)
                entity_record = entity_repo.upsert(
                    entity_type=entity_type,
                    entity_value=entity_value,
                    normalized_value=entity_value,
                    node_key=entity_key,
                )
                if entity_record:
                    node_key_to_entity_id[entity_key] = entity_record["id"]

            # 3. Link transactions to entities and create graph edges
            graph_edges = []
            for node1, node2, edge_data in self._graph.edges(data=True):
                data1 = self._graph.nodes.get(node1, {})
                data2 = self._graph.nodes.get(node2, {})
                if data1.get("node_type") == "transaction":
                    txn_id, entity_key = node1, node2
                elif data2.get("node_type") == "transaction":
                    txn_id, entity_key = node2, node1
                else:
                    continue

                entity_id = node_key_to_entity_id.get(entity_key)
                if entity_id is None:
                    continue
                relationship = edge_data.get("relationship", "unknown")
                entity_repo.link_to_transaction(txn_id, entity_id, relationship)
                graph_edges.append({
                    "transaction_id": txn_id,
                    "entity_id": entity_id,
                    "relationship": relationship,
                    "weight": 1.0,
                })

            # 4. Batch insert graph edges
            if graph_edges:
                graph_edge_repo.create_many(graph_edges)

            return True
        except Exception:
            logger.error("Graph persistence failed", exc_info=True)
            return False

    @property
    def transaction_count(self) -> int:
        return sum(1 for n, d in self._graph.nodes(data=True) if d.get("node_type") == "transaction")

    @property
    def entity_count(self) -> int:
        return sum(1 for n, d in self._graph.nodes(data=True) if d.get("node_type") != "transaction")

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
