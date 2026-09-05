from typing import Dict, List, Any, Optional, Set
import networkx as nx


# Entity type weights for graph risk scoring.
# These are engineering heuristics, NOT empirically validated fraud scores.
# They reflect the evidentiary strength of shared identifiers.
ENTITY_WEIGHTS = {
    "device": 10.0,
    "card": 10.0,
    "address": 5.0,
    "email_domain": 1.0,
    "merchant": 1.0,
    "customer": 0.5,
}

ML_WEIGHT = 0.70
NETWORK_WEIGHT = 0.30
SUSPICIOUS_THRESHOLD = 0.5
MAX_NETWORK_SCORE = 100


class NetworkRiskCalculator:
    """Calculates network-level risk from graph topology and shared entities.

    The network risk score is computed INDEPENDENTLY of the ML risk score.
    It is based solely on graph-derived signals:
      - suspicious connected transactions
      - connection count
      - shared entity diversity (weighted by entity type)
      - entity usage concentration (weak entities used by many transactions
        are de-emphasized)

    The combined risk is then:
        combined = ML_WEIGHT * ml_score + NETWORK_WEIGHT * graph_score
    """

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

    def _get_shared_entities(self, txn_id: str) -> List[Dict[str, Any]]:
        entities = []
        for entity_neighbor in self._graph.neighbors(txn_id):
            node_data = self._graph.nodes.get(entity_neighbor, {})
            if node_data.get("node_type") != "transaction":
                entity_type = node_data.get("node_type", "unknown")
                usage_count = self._get_entity_usage_count(entity_neighbor)
                if usage_count >= 2:
                    entities.append({
                        "node_key": entity_neighbor,
                        "type": entity_type,
                        "value": node_data.get("value", entity_neighbor),
                        "usage_count": usage_count,
                        "weight": ENTITY_WEIGHTS.get(entity_type, 1.0),
                    })
        return entities

    def _get_entity_usage_count(self, entity_key: str) -> int:
        return sum(
            1 for n in self._graph.neighbors(entity_key)
            if self._graph.nodes.get(n, {}).get("node_type") == "transaction"
        )

    def _compute_graph_score(self, txn_id: str) -> Dict[str, Any]:
        """Compute a graph-derived risk score (0-100) independent of ML risk."""
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
        shared_entities = self._get_shared_entities(txn_id)

        suspicious_neighbors = []
        for txn in connected_txns:
            fp = self._graph.nodes[txn].get("fraud_probability", 0)
            if fp is not None and fp >= SUSPICIOUS_THRESHOLD:
                suspicious_neighbors.append(txn)

        graph_score = 0.0
        factors: List[Dict[str, Any]] = []

        # Factor 1: Suspicious neighbors
        if suspicious_neighbors:
            # Each suspicious neighbor adds 20 points, capped at 50
            suspicious_boost = min(len(suspicious_neighbors) * 20.0, 50.0)
            graph_score += suspicious_boost
            factors.append({
                "factor": "suspicious_neighbors",
                "count": len(suspicious_neighbors),
                "boost": round(suspicious_boost, 2),
                "description": f"{len(suspicious_neighbors)} suspicious transaction(s) connected via shared entities",
            })

        # Factor 2: Connection density
        if len(connected_txns) > 0:
            density_boost = min(len(connected_txns) * 2.0, 15.0)
            graph_score += density_boost
            factors.append({
                "factor": "connection_density",
                "count": len(connected_txns),
                "boost": round(density_boost, 2),
                "description": f"Connected to {len(connected_txns)} other transactions",
            })

        # Factor 3: Shared entity diversity (weighted by entity type)
        weighted_entity_score = 0.0
        entity_type_set = set()
        for ent in shared_entities:
            entity_type_set.add(ent["type"])
            # Weight is de-emphasized when entity is used by many transactions
            usage_ratio = ent["usage_count"] / max(len(connected_txns) + 1, 2)
            effective_weight = ent["weight"] * (1.0 - min(usage_ratio, 0.9))
            weighted_entity_score += effective_weight

        if weighted_entity_score > 0:
            entity_boost = min(weighted_entity_score, 25.0)
            graph_score += entity_boost
            factors.append({
                "factor": "shared_entities",
                "types": list(entity_type_set),
                "weighted_score": round(weighted_entity_score, 2),
                "boost": round(entity_boost, 2),
                "description": f"Shares {len(entity_type_set)} entity type(s) with network (weighted by strength)",
            })

        # Factor 4: High entity usage (penalize when common entities dominate)
        common_weak_entities = [
            e for e in shared_entities
            if e["usage_count"] > 10 and ENTITY_WEIGHTS.get(e["type"], 1.0) <= 1.0
        ]
        if common_weak_entities and not suspicious_neighbors:
            # Reduce score when only weak common entities are present
            dampen = min(len(common_weak_entities) * 2.0, 10.0)
            graph_score = max(0, graph_score - dampen)
            factors.append({
                "factor": "common_weak_entities",
                "count": len(common_weak_entities),
                "dampen": round(dampen, 2),
                "description": f"{len(common_weak_entities)} common weak entity type(s) reduce network risk",
            })

        graph_score = max(0, min(MAX_NETWORK_SCORE, graph_score))

        if graph_score >= 71:
            network_risk_level = "HIGH"
        elif graph_score >= 31:
            network_risk_level = "MEDIUM"
        else:
            network_risk_level = "LOW"

        return {
            "network_risk_score": round(graph_score, 2),
            "network_risk_level": network_risk_level,
            "factors": factors,
            "neighbor_count": len(connected_txns),
            "suspicious_neighbor_count": len(suspicious_neighbors),
            "graph_available": True,
        }

    def compute_network_risk(self, txn_id: str, ml_risk_score: float = 0, ml_risk_level: str = "UNKNOWN") -> Dict[str, Any]:
        """Compute graph-derived network risk. ml_risk_score is accepted for
        signature compatibility but is NOT used in the graph score calculation.
        The graph score is purely graph-derived."""
        return self._compute_graph_score(txn_id)

    def compute_combined_risk(self, txn_id: str, ml_risk_score: float, ml_risk_level: str) -> Dict[str, Any]:
        """Compute combined risk: 70% ML + 30% graph-derived network risk."""
        network_result = self._compute_graph_score(txn_id)

        if network_result["graph_available"]:
            combined_score = ML_WEIGHT * ml_risk_score + NETWORK_WEIGHT * network_result["network_risk_score"]
        else:
            combined_score = ml_risk_score

        combined_score = max(0, min(MAX_NETWORK_SCORE, combined_score))

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
