from typing import Dict, List, Any, Optional, Set
import networkx as nx
from app.graph.entity_resolver import EntityResolver


class GraphQuerier:
    def __init__(self, graph: nx.Graph, resolver: EntityResolver):
        self._graph = graph
        self._resolver = resolver

    def get_connected_transactions(self, txn_id: str) -> List[Dict[str, Any]]:
        if txn_id not in self._graph:
            return []
        connected = []
        for neighbor in self._graph.neighbors(txn_id):
            node_data = self._graph.nodes[neighbor]
            if node_data.get("node_type") == "transaction":
                shared = self._get_shared_entities(txn_id, neighbor)
                connected.append({
                    "transaction_id": neighbor,
                    "shared_entities": shared,
                })
            else:
                for other in self._graph.neighbors(neighbor):
                    if other != txn_id:
                        other_data = self._graph.nodes.get(other, {})
                        if other_data.get("node_type") == "transaction":
                            entity_type = node_data.get("node_type", "unknown")
                            entity_value = node_data.get("value", neighbor)
                            already = any(c["transaction_id"] == other for c in connected)
                            if not already:
                                connected.append({
                                    "transaction_id": other,
                                    "shared_entities": [{"type": entity_type, "value": entity_value}],
                                })
                            else:
                                for c in connected:
                                    if c["transaction_id"] == other:
                                        c["shared_entities"].append({"type": entity_type, "value": entity_value})
        return connected

    def _get_shared_entities(self, txn_a: str, txn_b: str) -> List[Dict[str, str]]:
        shared = []
        neighbors_a = set(self._graph.neighbors(txn_a))
        neighbors_b = set(self._graph.neighbors(txn_b))
        common = neighbors_a & neighbors_b
        for entity_key in common:
            node_data = self._graph.nodes[entity_key]
            shared.append({
                "type": node_data.get("node_type", "unknown"),
                "value": node_data.get("value", entity_key),
            })
        return shared

    def get_entity_usage_count(self, entity_key: str) -> int:
        if entity_key not in self._graph:
            return 0
        return sum(
            1 for n in self._graph.neighbors(entity_key)
            if self._graph.nodes.get(n, {}).get("node_type") == "transaction"
        )

    def get_transaction_entities(self, txn_id: str) -> List[Dict[str, str]]:
        if txn_id not in self._graph:
            return []
        entities = []
        for neighbor in self._graph.neighbors(txn_id):
            node_data = self._graph.nodes[neighbor]
            if node_data.get("node_type") != "transaction":
                entities.append({
                    "type": node_data.get("node_type", "unknown"),
                    "value": node_data.get("value", neighbor),
                    "node_key": neighbor,
                })
        return entities

    def get_neighborhood(self, txn_id: str, max_hops: int = 2) -> Dict[str, Any]:
        if txn_id not in self._graph:
            return {"transaction_id": txn_id, "nodes": [], "edges": []}
        visited = set()
        queue = [(txn_id, 0)]
        nodes = []
        edges = []
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            node_data = self._graph.nodes[current]
            nodes.append({"id": current, "type": node_data.get("node_type", "unknown"), "depth": depth})
            for neighbor in self._graph.neighbors(current):
                if neighbor not in visited:
                    edge_data = self._graph.edges[current, neighbor]
                    edges.append({
                        "source": current,
                        "target": neighbor,
                        "relationship": edge_data.get("relationship", "unknown"),
                    })
                    queue.append((neighbor, depth + 1))
        return {"transaction_id": txn_id, "nodes": nodes, "edges": edges}

    def get_suspicious_transactions(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        suspicious = []
        for node, data in self._graph.nodes(data=True):
            if data.get("node_type") == "transaction":
                fp = data.get("fraud_probability", 0)
                if fp >= threshold:
                    suspicious.append({
                        "transaction_id": node,
                        "fraud_probability": fp,
                        "risk_level": data.get("risk_level", "UNKNOWN"),
                    })
        return suspicious
