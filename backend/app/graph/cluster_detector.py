from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx


class ClusterDetector:
    SUSPICIOUS_THRESHOLD = 0.5

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def detect_clusters(self) -> List[Dict[str, Any]]:
        clusters = []
        for idx, component in enumerate(nx.connected_components(self._graph)):
            subgraph = self._graph.subgraph(component)
            txn_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "transaction"]
            entity_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") != "transaction"]
            if not txn_nodes:
                continue
            entity_types = list(set(self._graph.nodes[n].get("node_type", "unknown") for n in entity_nodes))
            suspicious = [
                n for n in txn_nodes
                if self._graph.nodes[n].get("fraud_probability", 0) >= self.SUSPICIOUS_THRESHOLD
            ]
            total_risk = sum(self._graph.nodes[n].get("risk_score", 0) for n in txn_nodes)
            avg_risk = total_risk / len(txn_nodes) if txn_nodes else 0
            shared_ids = {}
            for e_node in entity_nodes:
                e_type = self._graph.nodes[e_node].get("node_type", "unknown")
                e_value = self._graph.nodes[e_node].get("value", e_node)
                if e_type not in shared_ids:
                    shared_ids[e_type] = []
                shared_ids[e_type].append(e_value)
            risk_level = self._compute_cluster_risk(suspicious, txn_nodes, avg_risk)
            clusters.append({
                "cluster_id": idx,
                "transaction_ids": txn_nodes,
                "entity_count": len(entity_nodes),
                "entity_types": entity_types,
                "total_transactions": len(txn_nodes),
                "suspicious_transaction_count": len(suspicious),
                "suspicious_ratio": len(suspicious) / len(txn_nodes) if txn_nodes else 0,
                "shared_identifiers": shared_ids,
                "risk_level": risk_level,
                "avg_risk_score": avg_risk,
            })
        clusters.sort(key=lambda c: c["suspicious_transaction_count"], reverse=True)
        return clusters

    def get_cluster_for_transaction(self, txn_id: str) -> Optional[Dict[str, Any]]:
        if txn_id not in self._graph:
            return None
        for component in nx.connected_components(self._graph):
            if txn_id in component:
                subgraph = self._graph.subgraph(component)
                txn_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "transaction"]
                entity_nodes = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") != "transaction"]
                entity_types = list(set(self._graph.nodes[n].get("node_type", "unknown") for n in entity_nodes))
                suspicious = [
                    n for n in txn_nodes
                    if self._graph.nodes[n].get("fraud_probability", 0) >= self.SUSPICIOUS_THRESHOLD
                ]
                shared_ids = {}
                for e_node in entity_nodes:
                    e_type = self._graph.nodes[e_node].get("node_type", "unknown")
                    e_value = self._graph.nodes[e_node].get("value", e_node)
                    if e_type not in shared_ids:
                        shared_ids[e_type] = []
                    shared_ids[e_type].append(e_value)
                return {
                    "transaction_ids": txn_nodes,
                    "entity_count": len(entity_nodes),
                    "entity_types": entity_types,
                    "total_transactions": len(txn_nodes),
                    "suspicious_transaction_count": len(suspicious),
                    "suspicious_ratio": len(suspicious) / len(txn_nodes) if txn_nodes else 0,
                    "shared_identifiers": shared_ids,
                    "risk_level": self._compute_cluster_risk(suspicious, txn_nodes, 0),
                }
        return None

    def _compute_cluster_risk(self, suspicious: List[str], all_txns: List[str], avg_risk: float) -> str:
        if not all_txns:
            return "UNKNOWN"
        ratio = len(suspicious) / len(all_txns)
        if ratio >= 0.5 or avg_risk >= 70:
            return "HIGH"
        elif ratio >= 0.2 or avg_risk >= 30:
            return "MEDIUM"
        else:
            return "LOW"
