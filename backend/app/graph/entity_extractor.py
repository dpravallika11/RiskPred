import math
from typing import Dict, List, Any, Optional


INVALID_IDENTIFIERS = frozenset({
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
    if s in INVALID_IDENTIFIERS:
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
    # Maps entity types to the transaction fields that represent them.
    # "email" fields are P_emaildomain/R_emaildomain which are EMAIL DOMAINS,
    # not individual email addresses. They represent weak shared attributes
    # because multiple unrelated users legitimately share the same domain.
    ENTITY_FIELD_MAP = {
        "card": ["card1", "card2", "card3", "card4", "card5", "card6"],
        "device": ["device_id"],
        "email_domain": ["p_emaildomain", "r_emaildomain"],
        "address": ["addr1", "addr2"],
        "merchant": ["merchant_id"],
        "customer": ["customer_id"],
    }

    def extract(self, txn_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        norm = {k.lower(): v for k, v in txn_dict.items()}
        entities: Dict[str, List[str]] = {}
        for entity_type, fields in self.ENTITY_FIELD_MAP.items():
            values = []
            for field in fields:
                raw = norm.get(field)
                normalized = _normalize_identifier(raw)
                if normalized is not None:
                    values.append(normalized)
            if values:
                entities[entity_type] = values
        return entities

    def extract_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
        return [self.extract(txn) for txn in transactions]
