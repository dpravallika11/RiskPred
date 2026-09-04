import math
from typing import Dict, List, Any, Optional


INVALID标识符标识符标识符 = frozenset({
    None, "UNKNOWN", "unknown", "", "N/A", "n/a", "null", "NULL",
    "NaN", "nan", "None", "none", "-1", "-1.0",
})


def _is_valid_identifier(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    s = str(value).strip()
    if s == "":
        return False
    if s in INVALID标识符标识符标识符:
        return False
    return True


def _normalize_identifier(value: Any) -> Optional[str]:
    if not _is_valid_identifier(value):
        return None
    s = str(value).strip()
    if isinstance(value, float) and value == int(value):
        s = str(int(value))
    return s


class EntityExtractor:
    ENTITY_FIELD_MAP = {
        "card": ["card1", "card2", "card3", "card4", "card5", "card6"],
        "device": ["device_id"],
        "email": ["P_emaildomain", "R_emaildomain"],
        "address": ["addr1", "addr2"],
        "merchant": ["merchant_id"],
        "customer": ["customer_id"],
    }

    def extract(self, txn_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        for entity_type, fields in self.ENTITY_FIELD_MAP.items():
            values = []
            for field in fields:
                raw = txn_dict.get(field)
                normalized = _normalize_identifier(raw)
                if normalized is not None:
                    values.append(normalized)
            if values:
                entities[entity_type] = values
        return entities

    def extract_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
        return [self.extract(txn) for txn in transactions]
