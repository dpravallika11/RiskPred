from typing import Dict, List, Set, Optional
from app.graph.entity_extractor import _normalize_identifier


class EntityResolver:
    def __init__(self):
        self._entity_map: Dict[str, Dict[str, str]] = {}
        self._reverse_map: Dict[str, Set[str]] = {}

    def _node_key(self, entity_type: str, value: str) -> str:
        return f"{entity_type}:{value}"

    def resolve(self, entity_type: str, value: str) -> str:
        normalized = _normalize_identifier(value)
        if normalized is None:
            raise ValueError(f"Cannot resolve invalid identifier: {value}")
        key = self._node_key(entity_type, normalized)
        if key not in self._reverse_map:
            self._reverse_map[key] = set()
        return key

    def link(self, entity_type: str, value: str, transaction_id: str) -> str:
        node_key = self.resolve(entity_type, value)
        self._reverse_map[node_key].add(transaction_id)
        if transaction_id not in self._entity_map:
            self._entity_map[transaction_id] = {}
        if entity_type not in self._entity_map[transaction_id]:
            self._entity_map[transaction_id][entity_type] = node_key
        return node_key

    def get_transactions_for_entity(self, entity_node_key: str) -> Set[str]:
        return self._reverse_map.get(entity_node_key, set())

    def get_entities_for_transaction(self, transaction_id: str) -> Dict[str, str]:
        return self._entity_map.get(transaction_id, {})

    def get_all_entity_keys(self) -> Set[str]:
        return set(self._reverse_map.keys())

    def get_all_transactions(self) -> Set[str]:
        return set(self._entity_map.keys())

    def clear(self):
        self._entity_map.clear()
        self._reverse_map.clear()

    @property
    def entity_count(self) -> int:
        return len(self._reverse_map)

    @property
    def transaction_count(self) -> int:
        return len(self._entity_map)
