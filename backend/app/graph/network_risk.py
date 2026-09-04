from typing import Dict, List, Any, Optional, Set
import networkx as nx


class NetworkRiskCalculator:
    NEIGHBOR_WEIGHT = 0.15
    SUSPICIOUS_NEIGHBOR_WEIGHT = 0.25
    CLUSTER_WEIGHT = 0.10
    MAX_NETWORK_BOOST = 30
    SUSPICIOUS_THRESHOLD = 0.5

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def _get_connected_transactions(self, txn_id: str) -> Set[str]:
        connected = set()
        for entity_neighbor in self._graph.neighbors(txn_id):
            if self._graph.nodes.get(entity_neighbor, {}).get("node_type") == "transaction":
                continue
            for other in self._graph.neighbors(entity_neighbor):
                if other != txn_id and self._graph.nodes.get(other, {}).get("node_type") == "transaction":
                    connected.add(other)
        return connected

    def _get_shared_entity_types(self, txn_id: str) -> Set[str]:
        types = set()
        for entity_neighbor in self._graph.neighbors(txn_id):
            if self._graph.nodes.get(entity_neighbor, {}).get("node_type") != "transaction":
                types.add(self._graph.nodes[entity_neighbor].get("node_type", "unknown"))
        return types

    def compute_network_risk(self, txn_id: str, ml_risk_score: float, ml_risk_level: str) -> Dict[str, Any]:
        if txn_id not in self._graph:
            return {
                "network_risk_score": 0,
                "network_risk_level": "UNKNOWN",
                "factors": [],
                "neighbor_count": 0,
                "suspicious_neighbor_count": 0,
                "graph_available": False,
            }

        connected_txns = self._get_connected_transactions(txn_id)
        shared_entity_types = self._get_shared_entity_types(txn_id)

        suspicious_neighbors = []
        for txn in connected_txns:
            fp = self._graph.nodes[txn].get("fraud_probability", 0)
            if fp >= self.SUSPICIOUS_THRESHOLD:
                suspicious_neighbors.append(txn)

        factors = []
        network_score = ml_risk_score

        if suspicious_neighbors:
            boost = min(len(suspicious_neighbors) * self.SUSPICIOUS_NEIGHBOR_WEIGHT * 100, self.MAX_NETWORK_BOOST)
            network_score += boost
            factors.append({
                "factor": "suspicious_neighbors",
                "count": len(suspicious_neighbors),
                "boost": round(boost, 2),
                "description": f"{len(suspicious_neighbors)} suspicious transaction(s) in network",
            })

        if len(connected_txns) > 0:
            connection_boost = min(len(connected_txns) * self.NEIGHBOR_WEIGHT * 10, 10)
            network_score += connection_boost
            factors.append({
                "factor": "network_connections",
                "count": len(connected_txns),
                "boost": round(connection_boost, 2),
                "description": f"Connected to {len(connected_txns)} other transactions",
            })

        if len(shared_entity_types) >= 3:
            entity_boost = 5.0
            network_score += entity_boost
            factors.append({
                "factor": "shared_entities",
                "types": list(shared_entity_types),
                "boost": entity_boost,
                "description": f"Shares {len(shared_entity_types)} entity types with network",
            })

        network_score = max(0, min(100, network_score))

        if network_score >= 71:
            network_risk_level = "HIGH"
        elif network_score >= 31:
            network_risk_level = "MEDIUM"
        else:
            network_risk_level = "LOW"

        return {
            "network_risk_score": round(network_score, 2),
            "network_risk_level": network_risk_level,
            "factors": factors,
            "neighbor_count": len(connected_txns),
            "suspicious_neighbor_count": len(suspicious_neighbors),
            "graph_available": True,
        }

    def compute_combined_risk(self, txn_id: str, ml_risk_score: float, ml_risk_level: str) -> Dict[str, Any]:
        network_result = self.compute_network_risk(txn_id, ml_risk_score, ml_risk_level)
        ml_weight = 0.70
        network_weight = 0.30
        if network_result["graph_available"]:
            combined_score = ml_risk_score * ml_weight + network_result["network_risk_score"] * network_weight
        else:
            combined_score = ml_risk_score
        combined_score = max(0, min(100, combined_score))
        if combined_score >= 71:
            combined_level = "HIGH"
        elif combined_score >= 31:
            combined_level = "MEDIUM"
        else:
            combined_level = "LOW"
        return {
            "ml_risk_score": ml_risk_score,
            "ml_risk_level": ml_risk_level,
            "network_risk_score": network_result["network_risk_score"],
            "network_risk_level": network_result["network_risk_level"],
            "combined_risk_score": round(combined_score, 2),
            "combined_risk_level": combined_level,
            "factors": network_result["factors"],
            "neighbor_count": network_result["neighbor_count"],
            "suspicious_neighbor_count": network_result["suspicious_neighbor_count"],
            "graph_available": network_result["graph_available"],
        }
