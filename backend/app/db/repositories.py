from typing import Any, Dict, List, Optional
from datetime import datetime

from app.db.supabase_client import get_supabase


class TransactionRepository:
    """CRUD operations for transactions table via Supabase."""

    def __init__(self):
        self.table = "transactions"

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sb = get_supabase()
        result = sb.table(self.table).insert(data).execute()
        return result.data[0] if result.data else None

    def get_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("transaction_id", transaction_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = sb.table(self.table).select("*").eq("id", id).execute()
        return result.data[0] if result.data else None

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .range(offset, offset + limit - 1)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    def update(self, transaction_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .update(data)
            .eq("transaction_id", transaction_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def delete(self, transaction_id: str) -> bool:
        sb = get_supabase()
        result = sb.table(self.table).delete().eq("transaction_id", transaction_id).execute()
        return True

    def count(self) -> int:
        sb = get_supabase()
        result = sb.table(self.table).select("*", count="exact").execute()
        return result.count or 0


class PredictionRepository:
    """CRUD operations for predictions table via Supabase."""

    def __init__(self):
        self.table = "predictions"

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sb = get_supabase()
        result = sb.table(self.table).insert(data).execute()
        return result.data[0] if result.data else None

    def get_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("transaction_id", transaction_id)
            .order("prediction_timestamp", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_all_by_transaction_id(self, transaction_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("transaction_id", transaction_id)
            .order("prediction_timestamp", desc=True)
            .execute()
        )
        return result.data or []

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .order("prediction_timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def get_by_risk_level(self, risk_level: str, limit: int = 100) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("risk_level", risk_level)
            .order("prediction_timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def delete(self, id: str) -> bool:
        sb = get_supabase()
        sb.table(self.table).delete().eq("id", id).execute()
        return True


class RiskFactorRepository:
    """CRUD operations for risk_factors table via Supabase."""

    def __init__(self):
        self.table = "risk_factors"

    def create_many(self, prediction_id: str, factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sb = get_supabase()
        records = [{**f, "prediction_id": prediction_id} for f in factors]
        result = sb.table(self.table).insert(records).execute()
        return result.data or []

    def get_by_prediction_id(self, prediction_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("prediction_id", prediction_id)
            .execute()
        )
        return result.data or []

    def delete_by_prediction_id(self, prediction_id: str) -> bool:
        sb = get_supabase()
        sb.table(self.table).delete().eq("prediction_id", prediction_id).execute()
        return True


class InvestigationRepository:
    """CRUD operations for investigations table via Supabase."""

    def __init__(self):
        self.table = "investigations"

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sb = get_supabase()
        result = sb.table(self.table).insert(data).execute()
        return result.data[0] if result.data else None

    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = sb.table(self.table).select("*").eq("id", id).execute()
        return result.data[0] if result.data else None

    def get_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("transaction_id", transaction_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .range(offset, offset + limit - 1)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    def update(self, id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sb = get_supabase()
        result = sb.table(self.table).update(data).eq("id", id).execute()
        return result.data[0] if result.data else None

    def delete(self, id: str) -> bool:
        sb = get_supabase()
        sb.table(self.table).delete().eq("id", id).execute()
        return True


class EvidenceRepository:
    """CRUD operations for investigation_evidence table via Supabase."""

    def __init__(self):
        self.table = "investigation_evidence"

    def create_many(self, investigation_id: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sb = get_supabase()
        records = [{**item, "investigation_id": investigation_id} for item in items]
        result = sb.table(self.table).insert(records).execute()
        return result.data or []

    def get_by_investigation_id(self, investigation_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("investigation_id", investigation_id)
            .execute()
        )
        return result.data or []

    def delete_by_investigation_id(self, investigation_id: str) -> bool:
        sb = get_supabase()
        sb.table(self.table).delete().eq("investigation_id", investigation_id).execute()
        return True


class PatternRepository:
    """CRUD operations for detected_patterns table via Supabase."""

    def __init__(self):
        self.table = "detected_patterns"

    def create_many(self, investigation_id: str, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sb = get_supabase()
        records = [{**p, "investigation_id": investigation_id} for p in patterns]
        result = sb.table(self.table).insert(records).execute()
        return result.data or []

    def get_by_investigation_id(self, investigation_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("investigation_id", investigation_id)
            .execute()
        )
        return result.data or []


class AgentResultRepository:
    """CRUD operations for agent_results table via Supabase."""

    def __init__(self):
        self.table = "agent_results"

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sb = get_supabase()
        result = sb.table(self.table).insert(data).execute()
        return result.data[0] if result.data else None

    def get_by_investigation_id(self, investigation_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .eq("investigation_id", investigation_id)
            .execute()
        )
        return result.data or []


class GraphEdgeRepository:
    """CRUD operations for graph_edges table via Supabase."""

    def __init__(self):
        self.table = "graph_edges"

    def create_many(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = sb.table(self.table).insert(edges).execute()
        return result.data or []

    def get_by_transaction_id(self, transaction_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.table)
            .select("*")
            .or_(f"source_transaction_id.eq.{transaction_id},target_transaction_id.eq.{transaction_id}")
            .execute()
        )
        return result.data or []

    def delete_all(self) -> bool:
        sb = get_supabase()
        sb.table(self.table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        return True


class EntityRepository:
    """CRUD operations for entities and transaction_entities tables via Supabase."""

    def __init__(self):
        self.table = "entities"
        self.junction_table = "transaction_entities"

    def upsert(self, entity_type: str, entity_value: str, normalized_value: str = None) -> Dict[str, Any]:
        sb = get_supabase()
        existing = (
            sb.table(self.table)
            .select("*")
            .eq("entity_type", entity_type)
            .eq("entity_value", entity_value)
            .execute()
        )
        if existing.data:
            sb.table(self.table).update({"last_seen_at": datetime.utcnow().isoformat()}).eq("id", existing.data[0]["id"]).execute()
            return existing.data[0]
        result = sb.table(self.table).insert({
            "entity_type": entity_type,
            "entity_value": entity_value,
            "normalized_value": normalized_value,
        }).execute()
        return result.data[0] if result.data else None

    def link_to_transaction(self, transaction_id: str, entity_id: str) -> Dict[str, Any]:
        sb = get_supabase()
        result = sb.table(self.junction_table).insert({
            "transaction_id": transaction_id,
            "entity_id": entity_id,
        }).execute()
        return result.data[0] if result.data else None

    def get_entities_for_transaction(self, transaction_id: str) -> List[Dict[str, Any]]:
        sb = get_supabase()
        result = (
            sb.table(self.junction_table)
            .select("*, entities(*)")
            .eq("transaction_id", transaction_id)
            .execute()
        )
        return result.data or []


# Singleton instances
transaction_repo = TransactionRepository()
prediction_repo = PredictionRepository()
risk_factor_repo = RiskFactorRepository()
investigation_repo = InvestigationRepository()
evidence_repo = EvidenceRepository()
pattern_repo = PatternRepository()
agent_result_repo = AgentResultRepository()
graph_edge_repo = GraphEdgeRepository()
entity_repo = EntityRepository()
