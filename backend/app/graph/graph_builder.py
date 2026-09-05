import networkx as nx
from typing import Dict, List, Any, Optional, Set
from app.graph.entity_extractor import EntityExtractor
from app.graph.entity_resolver import EntityResolver


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

    @property
    def transaction_count(self) -> int:
        return sum(1 for n, d in self._graph.nodes(data=True) if d.get("node_type") == "transaction")

    @property
    def entity_count(self) -> int:
        return sum(1 for n, d in self._graph.nodes(data=True) if d.get("node_type") != "transaction")

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
