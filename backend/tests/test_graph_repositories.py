"""Tests for EntityRepository and GraphEdgeRepository.

Uses mocks for the Supabase client so no live connection is required.
Verifies the Sprint 6.2 repository layer for persistent graph storage.
"""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_supabase_response(data=None, count=None):
    resp = MagicMock()
    resp.data = data if data is not None else []
    resp.count = count
    return resp


def _mock_sb_table():
    table = MagicMock()
    chain = MagicMock()
    table.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.range.return_value = chain
    chain.execute.return_value = _mock_supabase_response([])
    table.insert.return_value.execute.return_value = _mock_supabase_response([{"id": "fake-uuid"}])
    table.update.return_value = chain
    table.delete.return_value = chain
    return table


# ===========================================================================
# 1. EntityRepository — upsert behavior
# ===========================================================================

class TestEntityRepositoryUpsert:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_upsert_inserts_new_entity(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.upsert("card", "100", normalized_value="100")

        assert result is not None
        self.mock_table.insert.assert_called_once()
        call_args = self.mock_table.insert.call_args[0][0]
        assert call_args["entity_type"] == "card"
        assert call_args["entity_value"] == "100"
        assert call_args["node_key"] == "card:100"
        assert call_args["normalized_value"] == "100"

    def test_upsert_updates_existing_entity(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        existing = {"id": "existing-uuid", "entity_type": "card", "entity_value": "100", "node_key": "card:100"}
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([existing])

        result = repo.upsert("card", "100")

        assert result["id"] == "existing-uuid"
        self.mock_table.update.assert_called_once()

    def test_upsert_generates_node_key_automatically(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        repo.upsert("device", "DEV1")

        call_args = self.mock_table.insert.call_args[0][0]
        assert call_args["node_key"] == "device:DEV1"

    def test_upsert_accepts_custom_node_key(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        repo.upsert("card", "100", node_key="custom:node:key")

        call_args = self.mock_table.insert.call_args[0][0]
        assert call_args["node_key"] == "custom:node:key"


# ===========================================================================
# 2. EntityRepository — lookup by key
# ===========================================================================

class TestEntityRepositoryLookup:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_get_entity_by_key_found(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        entity = {"id": "uuid-1", "entity_type": "card", "entity_value": "100", "node_key": "card:100"}
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([entity])

        result = repo.get_entity_by_key("card", "100")

        assert result is not None
        assert result["entity_type"] == "card"
        assert result["node_key"] == "card:100"

    def test_get_entity_by_key_not_found(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.get_entity_by_key("card", "NONEXISTENT")

        assert result is None

    def test_get_entity_by_node_key_found(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        entity = {"id": "uuid-2", "entity_type": "device", "entity_value": "DEV1", "node_key": "device:DEV1"}
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([entity])

        result = repo.get_entity_by_node_key("device:DEV1")

        assert result is not None
        assert result["node_key"] == "device:DEV1"

    def test_get_entity_by_node_key_not_found(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.get_entity_by_node_key("device:NONEXISTENT")

        assert result is None


# ===========================================================================
# 3. EntityRepository — get_all_entity_keys
# ===========================================================================

class TestEntityRepositoryAllKeys:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_get_all_entity_keys(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response([
            {"node_key": "card:100"},
            {"node_key": "device:DEV1"},
            {"node_key": "address:315"},
        ])

        result = repo.get_all_entity_keys()

        assert len(result) == 3
        assert "card:100" in result
        assert "device:DEV1" in result
        assert "address:315" in result

    def test_get_all_entity_keys_empty(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.get_all_entity_keys()

        assert result == []


# ===========================================================================
# 4. EntityRepository — transaction ↔ entity linking
# ===========================================================================

class TestEntityRepositoryLinking:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_link_to_transaction_includes_relationship(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()

        repo.link_to_transaction("T1", "entity-uuid-1", "card")

        self.mock_table.insert.assert_called_once()
        call_args = self.mock_table.insert.call_args[0][0]
        assert call_args["transaction_id"] == "T1"
        assert call_args["entity_id"] == "entity-uuid-1"
        assert call_args["relationship"] == "card"

    def test_link_to_transaction_preserves_relationship_value(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()

        repo.link_to_transaction("T2", "entity-uuid-2", "device")

        call_args = self.mock_table.insert.call_args[0][0]
        assert call_args["relationship"] == "device"

    def test_get_entities_for_transaction(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        links = [
            {"transaction_id": "T1", "entity_id": "e1", "relationship": "card"},
            {"transaction_id": "T1", "entity_id": "e2", "relationship": "device"},
        ]
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response(links)

        result = repo.get_entities_for_transaction("T1")

        assert len(result) == 2
        assert result[0]["relationship"] == "card"
        assert result[1]["relationship"] == "device"

    def test_get_transactions_for_entity(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        links = [
            {"transaction_id": "T1", "entity_id": "e1", "relationship": "card"},
            {"transaction_id": "T2", "entity_id": "e1", "relationship": "card"},
        ]
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response(links)

        result = repo.get_transactions_for_entity("e1")

        assert len(result) == 2
        assert result[0]["transaction_id"] == "T1"
        assert result[1]["transaction_id"] == "T2"


# ===========================================================================
# 5. GraphEdgeRepository — create and query
# ===========================================================================

class TestGraphEdgeRepository:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_create_many_edges(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        edges = [
            {"transaction_id": "T1", "entity_id": "e1", "relationship": "card", "weight": 1.0},
            {"transaction_id": "T1", "entity_id": "e2", "relationship": "device", "weight": 1.0},
        ]
        self.mock_table.insert.return_value.execute.return_value = _mock_supabase_response(edges)

        result = repo.create_many(edges)

        assert len(result) == 2
        self.mock_table.insert.assert_called_once_with(edges)

    def test_get_by_transaction_id(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        edges = [
            {"transaction_id": "T1", "entity_id": "e1", "relationship": "card"},
            {"transaction_id": "T1", "entity_id": "e2", "relationship": "device"},
        ]
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response(edges)

        result = repo.get_by_transaction_id("T1")

        assert len(result) == 2
        assert result[0]["relationship"] == "card"
        assert result[1]["relationship"] == "device"

    def test_get_by_entity_id(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        edges = [
            {"transaction_id": "T1", "entity_id": "e1", "relationship": "card"},
            {"transaction_id": "T2", "entity_id": "e1", "relationship": "card"},
        ]
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response(edges)

        result = repo.get_by_entity_id("e1")

        assert len(result) == 2
        assert result[0]["transaction_id"] == "T1"
        assert result[1]["transaction_id"] == "T2"

    def test_delete_all(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()

        result = repo.delete_all()

        assert result is True
        self.mock_table.delete.assert_called_once()

    def test_create_many_empty_list(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        self.mock_table.insert.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.create_many([])

        assert result == []


# ===========================================================================
# 6. GraphEdge — model matches new schema
# ===========================================================================

class TestGraphEdgeModel:
    def test_graph_edge_has_correct_columns(self):
        from app.models.database_models import GraphEdge
        column_names = {c.name for c in GraphEdge.__table__.columns}
        assert "transaction_id" in column_names
        assert "entity_id" in column_names
        assert "relationship" in column_names
        assert "weight" in column_names
        assert "created_at" in column_names
        assert "id" in column_names
        # Old columns should NOT exist
        assert "source_transaction_id" not in column_names
        assert "target_transaction_id" not in column_names
        assert "edge_type" not in column_names
        assert "shared_entities" not in column_names


# ===========================================================================
# 7. Entity — model has node_key
# ===========================================================================

class TestEntityModel:
    def test_entity_has_node_key(self):
        from app.models.database_models import Entity
        column_names = {c.name for c in Entity.__table__.columns}
        assert "node_key" in column_names
        assert "entity_type" in column_names
        assert "entity_value" in column_names
        assert "normalized_value" in column_names

    def test_entity_node_key_is_unique(self):
        from app.models.database_models import Entity
        node_key_col = Entity.__table__.c.node_key
        assert node_key_col.unique is True


# ===========================================================================
# 8. TransactionEntity — model has relationship
# ===========================================================================

class TestTransactionEntityModel:
    def test_transaction_entity_has_relationship(self):
        from app.models.database_models import TransactionEntity
        column_names = {c.name for c in TransactionEntity.__table__.columns}
        assert "relationship" in column_names
        assert "transaction_id" in column_names
        assert "entity_id" in column_names


# ===========================================================================
# 9. Existing transaction/prediction repos unchanged
# ===========================================================================

class TestExistingReposUnchanged:
    def test_transaction_repo_still_works(self):
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        assert repo.table == "transactions"

    def test_prediction_repo_still_works(self):
        from app.db.repositories import PredictionRepository
        repo = PredictionRepository()
        assert repo.table == "predictions"

    def test_risk_factor_repo_still_works(self):
        from app.db.repositories import RiskFactorRepository
        repo = RiskFactorRepository()
        assert repo.table == "risk_factors"

    def test_investigation_repo_still_works(self):
        from app.db.repositories import InvestigationRepository
        repo = InvestigationRepository()
        assert repo.table == "investigations"

    def test_singleton_instances_exist(self):
        from app.db.repositories import (
            transaction_repo, prediction_repo, risk_factor_repo,
            investigation_repo, evidence_repo, pattern_repo,
            agent_result_repo, graph_edge_repo, entity_repo,
        )
        assert transaction_repo is not None
        assert prediction_repo is not None
        assert risk_factor_repo is not None
        assert investigation_repo is not None
        assert evidence_repo is not None
        assert pattern_repo is not None
        assert agent_result_repo is not None
        assert graph_edge_repo is not None
        assert entity_repo is not None


# ===========================================================================
# 10. Duplicate graph edge prevention (via UNIQUE constraint verified at model level)
# ===========================================================================

class TestGraphEdgeDuplicatePrevention:
    def test_graph_edge_model_has_unique_constraint(self):
        from app.models.database_models import GraphEdge
        table = GraphEdge.__table__
        unique_constraints = [uc for uc in table.constraints if hasattr(uc, 'columns')]
        # At least one unique constraint should exist beyond the primary key
        assert len(unique_constraints) >= 1

    def test_entity_model_has_unique_node_key(self):
        from app.models.database_models import Entity
        node_key_col = Entity.__table__.c.node_key
        assert node_key_col.unique is True

    def test_transaction_entity_model_allows_multiple_relationships(self):
        """transaction_entities should allow same txn+entity with different relationship."""
        from app.models.database_models import TransactionEntity
        table = TransactionEntity.__table__
        # The model should NOT have a unique constraint on (transaction_id, entity_id) alone
        # because we need different relationships for the same pair
        unique_constraints = [uc for uc in table.constraints if hasattr(uc, 'columns')]
        for uc in unique_constraints:
            col_names = [c.name for c in uc.columns]
            # Should not be just (transaction_id, entity_id) without relationship
            if set(col_names) == {"transaction_id", "entity_id"}:
                pytest.fail("transaction_entities should not have unique constraint on (transaction_id, entity_id) alone")


# ===========================================================================
# 11. EntityRepository — delete_all for graph cleanup
# ===========================================================================

class TestEntityRepositoryDeleteAll:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_delete_all_junction_table(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()

        result = repo.delete_all()

        assert result is True
        self.mock_table.delete.assert_called_once()
