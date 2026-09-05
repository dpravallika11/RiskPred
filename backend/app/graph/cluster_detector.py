from typing import Dict, List, Any, Optional, Set
import networkx as nx


# Entity types that are considered "weak" for fraud cluster evaluation.
# Common merchants and email domains shared by many transactions do not
# inherently indicate fraud. These are used to determine whether a cluster
# is suspicious based on meaningful entity types rather than sheer size.
WEAK_ENTITY_TYPES = {"merchant", "email_domain", "customer"}

# Strong entity types that suggest meaningful shared identifiers.
STRONG_ENTITY_TYPES = {"device", "card", "address"}

SUSPICIOUS_THRESHOLD = 0.5


class ClusterDetector:
    """Detects connected components in the transaction graph and evaluates
    their suspiciousness.

    Connected components are NOT automatically labeled as fraud clusters.
    Risk classification is based on:
      - suspicious transaction ratio
      - presence of strong shared entities (device, card, address)
      - average risk score
    """

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

            entity_type_set = set(self._graph.nodes[n].get("node_type", "unknown") for n in entity_nodes)
            strong_types = entity_type_set & STRONG_ENTITY_TYPES
            weak_types = entity_type_set & WEAK_ENTITY_TYPES

            suspicious = [
                n for n in txn_nodes
                if self._graph.nodes[n].get("fraud_probability", 0) is not None
                and self._graph.nodes[n].get("fraud_probability", 0) >= SUSPICIOUS_THRESHOLD
            ]
            total_risk = sum(self._graph.nodes[n].get("risk_score", 0) or 0 for n in txn_nodes)
            avg_risk = total_risk / len(txn_nodes) if txn_nodes else 0

            shared_ids: Dict[str, List[str]] = {}
            for e_node in entity_nodes:
                e_type = self._graph.nodes[e_node].get("node_type", "unknown")
                e_value = self._graph.nodes[e_node].get("value", e_node)
                if e_type not in shared_ids:
                    shared_ids[e_type] = []
                shared_ids[e_type].append(e_value)

            risk_level = self._compute_cluster_risk(
                suspicious, txn_nodes, avg_risk, strong_types, weak_types
            )
            clusters.append({
                "cluster_id": idx,
                "transaction_ids": txn_nodes,
                "entity_count": len(entity_nodes),
                "entity_types": list(entity_type_set),
                "total_transactions": len(txn_nodes),
                "suspicious_transaction_count": len(suspicious),
                "suspicious_ratio": len(suspicious) / len(txn_nodes) if txn_nodes else 0,
                "shared_identifiers": shared_ids,
                "risk_level": risk_level,
                "avg_risk_score": round(avg_risk, 2),
                "strong_entity_types": list(strong_types),
                "weak_entity_types": list(weak_types),
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
                entity_type_set = set(self._graph.nodes[n].get("node_type", "unknown") for n in entity_nodes)
                strong_types = entity_type_set & STRONG_ENTITY_TYPES
                weak_types = entity_type_set & WEAK_ENTITY_TYPES
                suspicious = [
                    n for n in txn_nodes
                    if self._graph.nodes[n].get("fraud_probability", 0) is not None
                    and self._graph.nodes[n].get("fraud_probability", 0) >= SUSPICIOUS_THRESHOLD
                ]
                total_risk = sum(self._graph.nodes[n].get("risk_score", 0) or 0 for n in txn_nodes)
                avg_risk = total_risk / len(txn_nodes) if txn_nodes else 0

                shared_ids: Dict[str, List[str]] = {}
                for e_node in entity_nodes:
                    e_type = self._graph.nodes[e_node].get("node_type", "unknown")
                    e_value = self._graph.nodes[e_node].get("value", e_node)
                    if e_type not in shared_ids:
                        shared_ids[e_type] = []
                    shared_ids[e_type].append(e_value)

                return {
                    "transaction_ids": txn_nodes,
                    "entity_count": len(entity_nodes),
                    "entity_types": list(entity_type_set),
                    "total_transactions": len(txn_nodes),
                    "suspicious_transaction_count": len(suspicious),
                    "suspicious_ratio": len(suspicious) / len(txn_nodes) if txn_nodes else 0,
                    "shared_identifiers": shared_ids,
                    "risk_level": self._compute_cluster_risk(
                        suspicious, txn_nodes, avg_risk, strong_types, weak_types
                    ),
                    "avg_risk_score": round(avg_risk, 2),
                    "strong_entity_types": list(strong_types),
                    "weak_entity_types": list(weak_types),
                }
        return None

    def _compute_cluster_risk(
        self,
        suspicious: List[str],
        all_txns: List[str],
        avg_risk: float,
        strong_types: set,
        weak_types: set,
    ) -> str:
        """Compute cluster risk level.

        A cluster is only HIGH if it has both:
        - a high suspicious ratio OR high average risk, AND
        - at least one strong entity type (or high suspicious ratio alone)

        Clusters connected only by weak entities (merchant, email_domain)
        are limited to MEDIUM at most unless suspicious ratio is very high.
        """
        if not all_txns:
            return "UNKNOWN"
        ratio = len(suspicious) / len(all_txns)

        # Very high suspicious ratio overrides everything
        if ratio >= 0.7:
            return "HIGH"
        # High ratio with strong entities = HIGH
        if ratio >= 0.5 and strong_types:
            return "HIGH"
        # High ratio but only weak entities = MEDIUM max
        if ratio >= 0.5 and not strong_types:
            return "MEDIUM"
        # High average risk with strong entities = HIGH
        if avg_risk >= 70 and strong_types:
            return "HIGH"
        # Moderate suspicious ratio with strong entities
        if ratio >= 0.2 and strong_types:
            return "MEDIUM"
        # High average risk but weak entities only = MEDIUM max
        if avg_risk >= 30:
            return "MEDIUM" if weak_types and not strong_types else "MEDIUM"
        # Low suspicious ratio with only weak entities = LOW
        if ratio >= 0.2 and not strong_types:
            return "LOW"
        return "LOW"
