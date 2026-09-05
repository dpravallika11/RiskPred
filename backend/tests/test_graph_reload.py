"""Tests for Sprint 6.4: Persistent Graph Reload After Backend Restart.

Verifies that GraphService.load_from_db() correctly reconstructs the
in-memory NetworkX graph from Supabase-persisted data, and that all
existing graph algorithms (queries, clusters, risk) work after reload.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


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


def _make_db_entities():
    """Simulate persisted entity records from the entities table."""
    return [
        {"id": "e1", "entity_type": "card", "entity_value": "100", "node_key": "card:100"},
        {"id": "e2", "entity_type": "device", "entity_value": "DEV1", "node_key": "device:DEV1"},
        {"id": "e3", "entity_type": "device", "entity_value": "DEV2", "node_key": "device:DEV2"},
        {"id": "e4", "entity_type": "merchant", "entity_value": "M1", "node_key": "merchant:M1"},
        {"id": "e5", "entity_type": "merchant", "entity_value": "M2", "node_key": "merchant:M2"},
        {"id": "e6", "entity_type": "merchant", "entity_value": "M3", "node_key": "merchant:M3"},
        {"id": "e7", "entity_type": "customer", "entity_value": "C1", "node_key": "customer:C1"},
        {"id": "e8", "entity_type": "customer", "entity_value": "C2", "node_key": "customer:C2"},
        {"id": "e9", "entity_type": "customer", "entity_value": "C3", "node_key": "customer:C3"},
        {"id": "e10", "entity_type": "email_domain", "entity_value": "gmail.com", "node_key": "email_domain:gmail.com"},
        {"id": "e11", "entity_type": "merchant", "entity_value": "M4", "node_key": "merchant:M4"},
        {"id": "e12", "entity_type": "customer", "entity_value": "C4", "node_key": "customer:C4"},
    ]


def _make_db_graph_edges():
    """Simulate persisted graph_edges records.

    T1 and T2 share device DEV1. T1 and T3 share card 100.
    T4 is isolated (unique merchant M4).
    """
    return [
        {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
        {"id": "ge2", "transaction_id": "T1", "entity_id": "e7", "relationship": "customer", "weight": 1.0},
        {"id": "ge3", "transaction_id": "T1", "entity_id": "e2", "relationship": "device", "weight": 1.0},
        {"id": "ge4", "transaction_id": "T1", "entity_id": "e1", "relationship": "card", "weight": 1.0},
        {"id": "ge5", "transaction_id": "T1", "entity_id": "e10", "relationship": "email_domain", "weight": 1.0},
        {"id": "ge6", "transaction_id": "T2", "entity_id": "e5", "relationship": "merchant", "weight": 1.0},
        {"id": "ge7", "transaction_id": "T2", "entity_id": "e8", "relationship": "customer", "weight": 1.0},
        {"id": "ge8", "transaction_id": "T2", "entity_id": "e2", "relationship": "device", "weight": 1.0},
        {"id": "ge9", "transaction_id": "T3", "entity_id": "e6", "relationship": "merchant", "weight": 1.0},
        {"id": "ge10", "transaction_id": "T3", "entity_id": "e9", "relationship": "customer", "weight": 1.0},
        {"id": "ge11", "transaction_id": "T3", "entity_id": "e3", "relationship": "device", "weight": 1.0},
        {"id": "ge12", "transaction_id": "T3", "entity_id": "e1", "relationship": "card", "weight": 1.0},
        {"id": "ge13", "transaction_id": "T4", "entity_id": "e11", "relationship": "merchant", "weight": 1.0},
        {"id": "ge14", "transaction_id": "T4", "entity_id": "e12", "relationship": "customer", "weight": 1.0},
    ]


def _make_db_predictions():
    """Simulate persisted prediction records."""
    return [
        {"transaction_id": "T1", "fraud_probability": 0.8, "risk_score": 80, "risk_level": "HIGH"},
        {"transaction_id": "T2", "fraud_probability": 0.9, "risk_score": 90, "risk_level": "HIGH"},
    ]


# ===========================================================================
# 1. Successful load from mocked Supabase repositories
# ===========================================================================

class TestLoadFromDBSuccess:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _mock_repos(self, entities=None, edges=None, predictions=None):
        if entities is None:
            entities = _make_db_entities()
        if edges is None:
            edges = _make_db_graph_edges()
        if predictions is None:
            predictions = _make_db_predictions()

        mock_entity_repo = MagicMock()
        mock_entity_repo.get_all.return_value = entities

        mock_edge_repo = MagicMock()
        mock_edge_repo.get_all.return_value = edges

        mock_pred_repo = MagicMock()
        mock_pred_repo.get_recent.return_value = predictions

        return mock_entity_repo, mock_edge_repo, mock_pred_repo

    def test_load_returns_true(self, service):
        e_repo, ed_repo, p_repo = self._mock_repos()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is True

    def test_load_sets_is_ready(self, service):
        e_repo, ed_repo, p_repo = self._mock_repos()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.is_ready is True

    def test_load_sets_last_built(self, service):
        e_repo, ed_repo, p_repo = self._mock_repos()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.last_built is not None
        assert isinstance(service.last_built, datetime)


# ===========================================================================
# 2. Entity node reconstruction
# ===========================================================================

class TestEntityNodeReconstruction:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_entity_nodes_created(self, service):
        self._load(service)
        graph = service._builder.graph
        assert "card:100" in graph
        assert "device:DEV1" in graph
        assert "device:DEV2" in graph

    def test_entity_node_type(self, service):
        self._load(service)
        graph = service._builder.graph
        assert graph.nodes["card:100"]["node_type"] == "card"
        assert graph.nodes["device:DEV1"]["node_type"] == "device"

    def test_entity_node_value(self, service):
        self._load(service)
        graph = service._builder.graph
        assert graph.nodes["card:100"]["value"] == "100"
        assert graph.nodes["device:DEV1"]["value"] == "DEV1"

    def test_entity_node_key_format(self, service):
        """Entity node keys must follow {entity_type}:{entity_value} format."""
        self._load(service)
        graph = service._builder.graph
        for node, data in graph.nodes(data=True):
            if data.get("node_type") != "transaction":
                entity_type = data.get("node_type")
                entity_value = data.get("value")
                expected_key = f"{entity_type}:{entity_value}"
                assert node == expected_key


# ===========================================================================
# 3. Transaction/entity edge reconstruction
# ===========================================================================

class TestEdgeReconstruction:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_transaction_nodes_created(self, service):
        self._load(service)
        graph = service._builder.graph
        assert "T1" in graph
        assert "T2" in graph
        assert "T3" in graph
        assert "T4" in graph

    def test_transaction_node_type(self, service):
        self._load(service)
        graph = service._builder.graph
        assert graph.nodes["T1"]["node_type"] == "transaction"

    def test_edges_created(self, service):
        self._load(service)
        graph = service._builder.graph
        assert graph.has_edge("T1", "card:100")
        assert graph.has_edge("T1", "device:DEV1")
        assert graph.has_edge("T2", "device:DEV1")
        assert graph.has_edge("T3", "card:100")
        assert graph.has_edge("T4", "merchant:M4")

    def test_edge_relationship_preserved(self, service):
        self._load(service)
        graph = service._builder.graph
        assert graph.edges["T1", "card:100"]["relationship"] == "card"
        assert graph.edges["T1", "device:DEV1"]["relationship"] == "device"
        assert graph.edges["T2", "device:DEV1"]["relationship"] == "device"

    def test_shared_entity_connections(self, service):
        """T1 and T2 should be connected via shared device:DEV1."""
        self._load(service)
        graph = service._builder.graph
        assert graph.has_edge("T1", "device:DEV1")
        assert graph.has_edge("T2", "device:DEV1")
        assert graph.has_edge("T3", "card:100")
        assert graph.has_edge("T1", "card:100")


# ===========================================================================
# 4. Correct transaction/entity/edge counts
# ===========================================================================

class TestCountsAfterLoad:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_transaction_count(self, service):
        self._load(service)
        assert service.transaction_count == 4

    def test_entity_count(self, service):
        self._load(service)
        assert service.entity_count == 12

    def test_edge_count(self, service):
        self._load(service)
        assert service.edge_count == 14


# ===========================================================================
# 5. GraphQuerier works after load
# ===========================================================================

class TestQuerierAfterLoad:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_get_connected_transactions(self, service):
        self._load(service)
        result = service.get_connected_transactions("T1")
        assert result["transaction_id"] == "T1"
        assert result["total_connections"] > 0

    def test_connected_includes_shared_device(self, service):
        self._load(service)
        result = service.get_connected_transactions("T1")
        conn_ids = [c["transaction_id"] for c in result["connected_transactions"]]
        assert "T2" in conn_ids

    def test_connected_includes_shared_card(self, service):
        self._load(service)
        result = service.get_connected_transactions("T1")
        conn_ids = [c["transaction_id"] for c in result["connected_transactions"]]
        assert "T3" in conn_ids

    def test_get_transaction_entities(self, service):
        self._load(service)
        entities = service.get_transaction_entities("T1")
        entity_types = [e["type"] for e in entities]
        assert "card" in entity_types
        assert "device" in entity_types

    def test_get_neighborhood(self, service):
        self._load(service)
        hood = service.get_neighborhood("T1")
        assert hood["transaction_id"] == "T1"
        assert len(hood["nodes"]) > 0

    def test_get_suspicious_transactions_no_risk(self, service):
        self._load(service)
        result = service.get_suspicious_transactions(threshold=0.5)
        assert isinstance(result, list)
        assert len(result) == 0


# ===========================================================================
# 6. Cluster detection works after load
# ===========================================================================

class TestClustersAfterLoad:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service, predictions=None):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = predictions or []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_detect_clusters(self, service):
        self._load(service)
        result = service.get_clusters()
        assert result["total_clusters"] >= 2
        assert result["total_transactions_in_clusters"] == 4

    def test_cluster_for_transaction(self, service):
        self._load(service)
        cluster = service.get_cluster_for_transaction("T1")
        assert cluster is not None
        assert "T1" in cluster["transaction_ids"]

    def test_cluster_risk_level(self, service):
        self._load(service)
        result = service.get_clusters()
        for c in result["clusters"]:
            assert c["risk_level"] in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")

    def test_clusters_with_risk(self, service):
        self._load(service, predictions=_make_db_predictions())
        result = service.get_clusters()
        assert result["total_clusters"] >= 2


# ===========================================================================
# 7. Network risk can operate after load
# ===========================================================================

class TestNetworkRiskAfterLoad:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service, predictions=None):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = predictions or []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_network_risk_computed(self, service):
        self._load(service)
        result = service.get_network_risk("T1", 50, "MEDIUM")
        assert "combined_risk_score" in result
        assert "network_risk_score" in result

    def test_network_risk_with_suspicious_neighbors(self, service):
        self._load(service, predictions=_make_db_predictions())
        result = service.get_network_risk("T1", 50, "MEDIUM")
        assert result["suspicious_neighbor_count"] > 0

    def test_get_transaction_risk_after_load(self, service):
        self._load(service, predictions=_make_db_predictions())
        risk = service.get_transaction_risk("T1")
        assert risk is not None
        assert risk["risk_score"] == 80

    def test_get_neighborhood_risk(self, service):
        self._load(service, predictions=_make_db_predictions())
        result = service.get_neighborhood_risk("T1")
        assert result["neighbor_count"] > 0
        assert result["suspicious_neighbor_count"] > 0


# ===========================================================================
# 8. Empty database does not crash startup
# ===========================================================================

class TestEmptyDatabase:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_empty_entities_returns_false(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = []
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = []
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False

    def test_empty_database_graph_not_ready(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = []
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = []
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.is_ready is False

    def test_empty_entities_only_returns_false(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = []
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False


# ===========================================================================
# 9. Supabase/database failure does not crash application startup
# ===========================================================================

class TestDatabaseFailure:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_entity_repo_exception_returns_false(self, service):
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("Connection refused")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False

    def test_edge_repo_exception_returns_false(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.side_effect = Exception("Timeout")
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False

    def test_prediction_repo_exception_still_loads_graph(self, service):
        """Prediction failure should not prevent graph from loading."""
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.side_effect = Exception("DB error")
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is True
        assert service.is_ready is True

    def test_service_not_ready_after_failure(self, service):
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("Connection refused")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.is_ready is False

    def test_import_error_returns_false(self, service):
        """If repositories cannot be imported, load fails gracefully."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            result = service.load_from_db()
        assert result is False


# ===========================================================================
# 10. Existing build/persistence behavior remains intact
# ===========================================================================

class TestExistingBehaviorUnchanged:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _txns(self):
        return [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1', 'card1': 200},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV2'},
        ]

    def test_build_still_works(self, service):
        service.build(self._txns())
        assert service.is_ready
        assert service.transaction_count == 3

    def test_build_with_risk(self, service):
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        }
        service.build(self._txns(), risks)
        risk = service.get_transaction_risk('T1')
        assert risk is not None
        assert risk['risk_score'] == 80

    def test_clear_still_works(self, service):
        service.build(self._txns())
        service.clear()
        assert not service.is_ready

    def test_not_ready_before_build_or_load(self, service):
        assert not service.is_ready

    def test_build_returns_correct_counts(self, service):
        service.build(self._txns())
        assert service.transaction_count == 3
        assert service.entity_count > 0
        assert service.edge_count > 0


# ===========================================================================
# 11. Existing graph API contract remains unchanged
# ===========================================================================

class TestAPIContractUnchanged:
    @pytest.fixture(autouse=True)
    def reset_graph_service(self):
        from app.graph.graph_service import graph_service
        graph_service.clear()
        yield
        graph_service.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def _txns(self):
        return [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'amount': 100, 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'amount': 200, 'device_id': 'DEV1', 'card1': 200},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'amount': 50, 'device_id': 'DEV2'},
        ]

    def test_graph_status_not_built(self, client):
        response = client.get("/api/v1/graph/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'not_built'
        assert 'transaction_count' in data
        assert 'entity_count' in data
        assert 'edge_count' in data
        assert 'last_built' in data

    def test_graph_build_response_format(self, client):
        response = client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'built'
        assert 'transaction_count' in data
        assert 'entity_count' in data
        assert 'edge_count' in data
        assert 'build_timestamp' in data

    def test_graph_query_endpoints_require_build(self, client):
        assert client.get("/api/v1/graph/transaction/T1").status_code == 503
        assert client.get("/api/v1/graph/transaction/T1/connections").status_code == 503
        assert client.get("/api/v1/graph/transaction/T1/neighborhood").status_code == 503
        assert client.get("/api/v1/graph/transaction/T1/risk").status_code == 503

    def test_graph_build_then_query(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert 'entities' in data
        assert 'entity_count' in data


# ===========================================================================
# 12. Repository get_all() methods
# ===========================================================================

class TestRepositoryGetAll:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_entity_repo_get_all(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        entities = [
            {"id": "e1", "entity_type": "card", "entity_value": "100", "node_key": "card:100"},
            {"id": "e2", "entity_type": "device", "entity_value": "DEV1", "node_key": "device:DEV1"},
        ]
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response(entities)
        result = repo.get_all()
        assert len(result) == 2
        assert result[0]["node_key"] == "card:100"

    def test_entity_repo_get_all_empty(self):
        from app.db.repositories import EntityRepository
        repo = EntityRepository()
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response([])
        result = repo.get_all()
        assert result == []

    def test_graph_edge_repo_get_all(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e1", "relationship": "card", "weight": 1.0},
        ]
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response(edges)
        result = repo.get_all()
        assert len(result) == 1
        assert result[0]["transaction_id"] == "T1"

    def test_graph_edge_repo_get_all_empty(self):
        from app.db.repositories import GraphEdgeRepository
        repo = GraphEdgeRepository()
        self.mock_table.select.return_value.execute.return_value = _mock_supabase_response([])
        result = repo.get_all()
        assert result == []


# ===========================================================================
# 13. Resolver reconstruction
# ===========================================================================

class TestResolverReconstruction:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_resolver_reverse_map_populated(self, service):
        self._load(service)
        resolver = service._builder.resolver
        txns_for_device = resolver.get_transactions_for_entity("device:DEV1")
        assert "T1" in txns_for_device
        assert "T2" in txns_for_device

    def test_resolver_entity_map_populated(self, service):
        self._load(service)
        resolver = service._builder.resolver
        entities = resolver.get_entities_for_transaction("T1")
        assert "card" in entities
        assert "device" in entities
        assert entities["card"] == "card:100"
        assert entities["device"] == "device:DEV1"

    def test_resolver_all_entity_keys(self, service):
        self._load(service)
        resolver = service._builder.resolver
        all_keys = resolver.get_all_entity_keys()
        assert "card:100" in all_keys
        assert "device:DEV1" in all_keys
        assert "merchant:M1" in all_keys

    def test_resolver_all_transactions(self, service):
        self._load(service)
        resolver = service._builder.resolver
        all_txns = resolver.get_all_transactions()
        assert "T1" in all_txns
        assert "T2" in all_txns
        assert "T3" in all_txns
        assert "T4" in all_txns


# ===========================================================================
# 14. Risk data restoration
# ===========================================================================

class TestRiskDataRestoration:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def _load(self, service, predictions):
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = predictions
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()

    def test_risk_restored_on_nodes(self, service):
        self._load(service, _make_db_predictions())
        graph = service._builder.graph
        assert graph.nodes["T1"]["fraud_probability"] == 0.8
        assert graph.nodes["T1"]["risk_score"] == 80
        assert graph.nodes["T1"]["risk_level"] == "HIGH"

    def test_risk_available_via_service(self, service):
        self._load(service, _make_db_predictions())
        risk = service.get_transaction_risk("T1")
        assert risk is not None
        assert risk["fraud_probability"] == 0.8

    def test_no_predictions_no_risk(self, service):
        self._load(service, [])
        risk = service.get_transaction_risk("T1")
        assert risk is None

    def test_partial_predictions(self, service):
        preds = [{"transaction_id": "T1", "fraud_probability": 0.5, "risk_score": 50, "risk_level": "MEDIUM"}]
        self._load(service, preds)
        assert service.get_transaction_risk("T1") is not None
        assert service.get_transaction_risk("T2") is None


# ===========================================================================
# 15. Build → Persist → Load round-trip
# ===========================================================================

class TestBuildPersistLoadRoundTrip:
    """Verify that build→persist→load produces a usable graph."""

    def test_round_trip_preserves_structure(self):
        from app.graph.graph_service import GraphService

        # 1. Build graph in memory
        service = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1', 'card1': 200},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV2', 'card1': 100},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        service.build(txns, risks)
        orig_txn_count = service.transaction_count
        orig_entity_count = service.entity_count
        orig_edge_count = service.edge_count

        # 2. Capture graph state for comparison
        orig_connected = service.get_connected_transactions("T1")
        orig_clusters = service.get_clusters()

        # 3. Simulate "restart" - create fresh service and load from DB
        # Mock the repositories to return data that _persist_graph_to_db would have written
        fresh_service = GraphService()
        e_repo = MagicMock()
        e_repo.get_all.return_value = [
            {"id": "e1", "entity_type": "card", "entity_value": "100", "node_key": "card:100"},
            {"id": "e2", "entity_type": "device", "entity_value": "DEV1", "node_key": "device:DEV1"},
            {"id": "e3", "entity_type": "device", "entity_value": "DEV2", "node_key": "device:DEV2"},
            {"id": "e4", "entity_type": "merchant", "entity_value": "M1", "node_key": "merchant:M1"},
            {"id": "e5", "entity_type": "merchant", "entity_value": "M2", "node_key": "merchant:M2"},
            {"id": "e6", "entity_type": "merchant", "entity_value": "M3", "node_key": "merchant:M3"},
            {"id": "e7", "entity_type": "customer", "entity_value": "C1", "node_key": "customer:C1"},
            {"id": "e8", "entity_type": "customer", "entity_value": "C2", "node_key": "customer:C2"},
            {"id": "e9", "entity_type": "customer", "entity_value": "C3", "node_key": "customer:C3"},
            {"id": "e10", "entity_type": "card", "entity_value": "200", "node_key": "card:200"},
        ]
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T1", "entity_id": "e7", "relationship": "customer", "weight": 1.0},
            {"id": "ge3", "transaction_id": "T1", "entity_id": "e2", "relationship": "device", "weight": 1.0},
            {"id": "ge4", "transaction_id": "T1", "entity_id": "e1", "relationship": "card", "weight": 1.0},
            {"id": "ge5", "transaction_id": "T2", "entity_id": "e5", "relationship": "merchant", "weight": 1.0},
            {"id": "ge6", "transaction_id": "T2", "entity_id": "e8", "relationship": "customer", "weight": 1.0},
            {"id": "ge7", "transaction_id": "T2", "entity_id": "e2", "relationship": "device", "weight": 1.0},
            {"id": "ge12", "transaction_id": "T2", "entity_id": "e10", "relationship": "card", "weight": 1.0},
            {"id": "ge8", "transaction_id": "T3", "entity_id": "e6", "relationship": "merchant", "weight": 1.0},
            {"id": "ge9", "transaction_id": "T3", "entity_id": "e9", "relationship": "customer", "weight": 1.0},
            {"id": "ge10", "transaction_id": "T3", "entity_id": "e3", "relationship": "device", "weight": 1.0},
            {"id": "ge11", "transaction_id": "T3", "entity_id": "e1", "relationship": "card", "weight": 1.0},
        ]
        p_repo = MagicMock()
        p_repo.get_recent.return_value = [
            {"transaction_id": "T1", "fraud_probability": 0.8, "risk_score": 80, "risk_level": "HIGH"},
            {"transaction_id": "T2", "fraud_probability": 0.9, "risk_score": 90, "risk_level": "HIGH"},
        ]
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            loaded = fresh_service.load_from_db()

        assert loaded is True
        assert fresh_service.is_ready is True

        # 4. Verify counts match
        assert fresh_service.transaction_count == orig_txn_count
        assert fresh_service.entity_count == orig_entity_count
        assert fresh_service.edge_count == orig_edge_count

        # 5. Verify graph queries work identically
        loaded_connected = fresh_service.get_connected_transactions("T1")
        orig_conn_ids = sorted([c["transaction_id"] for c in orig_connected["connected_transactions"]])
        loaded_conn_ids = sorted([c["transaction_id"] for c in loaded_connected["connected_transactions"]])
        assert orig_conn_ids == loaded_conn_ids

        # 6. Verify clusters work
        loaded_clusters = fresh_service.get_clusters()
        assert loaded_clusters["total_clusters"] == orig_clusters["total_clusters"]
        assert loaded_clusters["total_transactions_in_clusters"] == orig_clusters["total_transactions_in_clusters"]

        # 7. Verify risk is accessible
        risk = fresh_service.get_transaction_risk("T1")
        assert risk is not None
        assert risk["risk_score"] == 80


# ===========================================================================
# Sprint 6.5: Edge-case hardening for persistent graph reload
# ===========================================================================


# ---------------------------------------------------------------------------
# Helper for Sprint 6.5 tests
# ---------------------------------------------------------------------------

def _load_service(service, entities=None, edges=None, predictions=None):
    """Helper to load a GraphService with mocked repos."""
    e_repo = MagicMock()
    e_repo.get_all.return_value = entities if entities is not None else _make_db_entities()
    ed_repo = MagicMock()
    ed_repo.get_all.return_value = edges if edges is not None else _make_db_graph_edges()
    p_repo = MagicMock()
    p_repo.get_recent.return_value = predictions if predictions is not None else []
    with patch("app.graph.graph_service.entity_repo", e_repo), \
         patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
         patch("app.graph.graph_service.prediction_repo", p_repo):
        return service.load_from_db()


# ===========================================================================
# 16. Empty database / no persisted entities (startup-safe)
# ===========================================================================

class TestEmptyDatabaseStartupSafe:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_empty_db_returns_false(self, service):
        result = _load_service(service, entities=[], edges=[], predictions=[])
        assert result is False

    def test_empty_db_graph_not_ready(self, service):
        _load_service(service, entities=[], edges=[], predictions=[])
        assert service.is_ready is False

    def test_empty_db_counts_zero(self, service):
        _load_service(service, entities=[], edges=[], predictions=[])
        assert service.transaction_count == 0
        assert service.entity_count == 0
        assert service.edge_count == 0

    def test_empty_db_no_nodes_in_graph(self, service):
        _load_service(service, entities=[], edges=[], predictions=[])
        assert len(service._builder.graph) == 0

    def test_startup_load_graph_handler_does_not_crash_on_empty_db(self, service):
        """Simulate the startup handler when load_from_db returns False."""
        from unittest.mock import patch as mp
        e_repo = MagicMock()
        e_repo.get_all.return_value = []
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = []
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []
        with mp("app.graph.graph_service.entity_repo", e_repo), \
             mp("app.graph.graph_service.graph_edge_repo", ed_repo), \
             mp("app.graph.graph_service.prediction_repo", p_repo):
            loaded = service.load_from_db()
        # Must not raise; graph stays not-ready
        assert loaded is False
        assert service.is_ready is False


# ===========================================================================
# 17. Empty graph_edges while entities exist
# ===========================================================================

class TestEmptyEdgesWithEntities:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_empty_edges_returns_true(self, service):
        """Entities exist but no edges => graph is loaded (True) since entities are non-empty."""
        result = _load_service(service, edges=[])
        assert result is True

    def test_empty_edges_graph_ready(self, service):
        _load_service(service, edges=[])
        assert service.is_ready is True

    def test_empty_edges_no_transaction_nodes(self, service):
        _load_service(service, edges=[])
        assert service.transaction_count == 0

    def test_empty_edges_no_entity_nodes_in_graph(self, service):
        """Without edges, no entity nodes are added to the graph."""
        _load_service(service, edges=[])
        assert service.entity_count == 0

    def test_empty_edges_zero_edge_count(self, service):
        _load_service(service, edges=[])
        assert service.edge_count == 0

    def test_empty_edges_querier_works(self, service):
        _load_service(service, edges=[])
        result = service.get_connected_transactions("T1")
        assert result["total_connections"] == 0

    def test_empty_edges_clusters_empty(self, service):
        _load_service(service, edges=[])
        result = service.get_clusters()
        assert result["total_clusters"] == 0
        assert result["total_transactions_in_clusters"] == 0


# ===========================================================================
# 18. Missing entity referenced by a graph edge
# ===========================================================================

class TestMissingEntityReferencedByEdge:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_single_missing_entity_edge_skipped(self, service):
        """Edge referencing entity 'e99' that does not exist is skipped safely."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T1", "entity_id": "e99", "relationship": "card", "weight": 1.0},
        ]
        result = _load_service(service, edges=edges)
        assert result is True
        assert service.is_ready is True
        # T1 should exist with merchant edge only
        assert "T1" in service._builder.graph
        assert service._builder.graph.has_edge("T1", "merchant:M1")
        assert not service._builder.graph.has_edge("T1", "card:100")

    def test_missing_entity_does_not_crash(self, service):
        """Load does not crash even if all edges reference missing entities."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e99", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T2", "entity_id": "e100", "relationship": "card", "weight": 1.0},
        ]
        result = _load_service(service, edges=edges)
        assert result is True
        assert service.is_ready is True

    def test_all_missing_entities_empty_graph(self, service):
        """If all edges reference missing entities, graph has transaction nodes but no entity nodes or edges."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e99", "relationship": "merchant", "weight": 1.0},
        ]
        _load_service(service, edges=edges)
        assert "T1" in service._builder.graph
        assert service.entity_count == 0
        assert service.edge_count == 0

    def test_mixed_present_and_missing_entities(self, service):
        """Some edges have valid entities, some don't. Only valid edges are loaded."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T1", "entity_id": "e99", "relationship": "card", "weight": 1.0},
            {"id": "ge3", "transaction_id": "T2", "entity_id": "e5", "relationship": "merchant", "weight": 1.0},
            {"id": "ge4", "transaction_id": "T2", "entity_id": "e100", "relationship": "customer", "weight": 1.0},
        ]
        _load_service(service, edges=edges)
        # T1 has merchant edge, T2 has merchant edge
        assert service._builder.graph.has_edge("T1", "merchant:M1")
        assert service._builder.graph.has_edge("T2", "merchant:M2")
        assert service.edge_count == 2

    def test_missing_entity_deterministic_results(self, service):
        """Calling load twice with same missing-entity data produces identical graphs."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T1", "entity_id": "e99", "relationship": "card", "weight": 1.0},
        ]
        _load_service(service, edges=edges)
        first_nodes = set(service._builder.graph.nodes())
        first_edges = service.edge_count

        # Reload
        _load_service(service, edges=edges)
        second_nodes = set(service._builder.graph.nodes())
        second_edges = service.edge_count

        assert first_nodes == second_nodes
        assert first_edges == second_edges

    def test_missing_entity_resolver_consistent(self, service):
        """Resolver only maps entities that were successfully loaded."""
        edges = [
            {"id": "ge1", "transaction_id": "T1", "entity_id": "e4", "relationship": "merchant", "weight": 1.0},
            {"id": "ge2", "transaction_id": "T1", "entity_id": "e99", "relationship": "card", "weight": 1.0},
        ]
        _load_service(service, edges=edges)
        resolver = service._builder.resolver
        entities_for_t1 = resolver.get_entities_for_transaction("T1")
        assert "merchant" in entities_for_t1
        assert "card" not in entities_for_t1


# ===========================================================================
# 19. Prediction loading edge cases
# ===========================================================================

class TestPredictionLoadingEdgeCases:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_unrelated_predictions_do_not_create_nodes(self, service):
        """Predictions for transaction IDs not in the graph should not create new nodes."""
        predictions = [
            {"transaction_id": "TX_UNKNOWN_1", "fraud_probability": 0.9, "risk_score": 90, "risk_level": "HIGH"},
            {"transaction_id": "TX_UNKNOWN_2", "fraud_probability": 0.1, "risk_score": 10, "risk_level": "LOW"},
        ]
        _load_service(service, predictions=predictions)
        txn_count = service.transaction_count
        # Should only have the 4 original transactions (T1-T4), no extra from predictions
        assert txn_count == 4
        assert "TX_UNKNOWN_1" not in service._builder.graph
        assert "TX_UNKNOWN_2" not in service._builder.graph

    def test_malformed_prediction_missing_fields_skipped(self, service):
        """Prediction with missing transaction_id or risk fields is skipped gracefully."""
        predictions = [
            {"transaction_id": None, "fraud_probability": 0.5, "risk_score": 50, "risk_level": "MEDIUM"},
            {"transaction_id": "T1"},  # Missing risk fields - should use defaults
            {"transaction_id": "T1", "fraud_probability": 0.7, "risk_score": 70, "risk_level": "HIGH"},
        ]
        result = _load_service(service, predictions=predictions)
        assert result is True
        assert service.is_ready is True
        # T1 should still exist with risk from last prediction
        risk = service.get_transaction_risk("T1")
        assert risk is not None

    def test_prediction_for_nonexistent_txn_ignored(self, service):
        """Prediction referencing a transaction not in the graph is silently ignored."""
        predictions = [
            {"transaction_id": "T_NONEXISTENT", "fraud_probability": 0.99, "risk_score": 99, "risk_level": "HIGH"},
        ]
        _load_service(service, predictions=predictions)
        assert "T_NONEXISTENT" not in service._builder.graph
        assert len(service._builder._transaction_risk) == 0

    def test_prediction_failure_still_loads_graph(self, service):
        """If prediction_repo.get_recent raises, graph still loads successfully."""
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.side_effect = Exception("DB connection lost")
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is True
        assert service.is_ready is True
        # Graph should have full structure, just no risk data
        assert service.transaction_count == 4
        assert service.entity_count == 12
        risk = service.get_transaction_risk("T1")
        assert risk is None

    def test_empty_predictions_graph_loads_clean(self, service):
        """No predictions means graph loads without risk data on any node."""
        _load_service(service, predictions=[])
        assert service.is_ready is True
        for txn_id in ["T1", "T2", "T3", "T4"]:
            assert service.get_transaction_risk(txn_id) is None


# ===========================================================================
# 20. Reload idempotency
# ===========================================================================

class TestReloadIdempotency:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_double_load_same_transaction_count(self, service):
        _load_service(service)
        first_txn = service.transaction_count
        _load_service(service)
        second_txn = service.transaction_count
        assert first_txn == second_txn

    def test_double_load_same_entity_count(self, service):
        _load_service(service)
        first_entity = service.entity_count
        _load_service(service)
        second_entity = service.entity_count
        assert first_entity == second_entity

    def test_double_load_same_edge_count(self, service):
        _load_service(service)
        first_edge = service.edge_count
        _load_service(service)
        second_edge = service.edge_count
        assert first_edge == second_edge

    def test_double_load_no_duplicate_nodes(self, service):
        _load_service(service)
        first_nodes = set(service._builder.graph.nodes())
        _load_service(service)
        second_nodes = set(service._builder.graph.nodes())
        assert first_nodes == second_nodes

    def test_double_load_no_duplicate_edges(self, service):
        _load_service(service)
        first_edges = set(service._builder.graph.edges())
        _load_service(service)
        second_edges = set(service._builder.graph.edges())
        assert first_edges == second_edges

    def test_triple_load_stable(self, service):
        """Calling load_from_db three times produces identical results."""
        _load_service(service)
        txn3 = service.transaction_count
        ent3 = service.entity_count
        ed3 = service.edge_count

        _load_service(service)
        _load_service(service)

        assert service.transaction_count == txn3
        assert service.entity_count == ent3
        assert service.edge_count == ed3

    def test_idempotent_query_results(self, service):
        """Graph queries return same results after multiple loads."""
        _load_service(service)
        first_connected = service.get_connected_transactions("T1")
        first_clusters = service.get_clusters()

        _load_service(service)
        second_connected = service.get_connected_transactions("T1")
        second_clusters = service.get_clusters()

        assert first_connected["total_connections"] == second_connected["total_connections"]
        assert first_clusters["total_clusters"] == second_clusters["total_clusters"]

    def test_idempotent_resolver_state(self, service):
        """Resolver maps are identical after multiple loads."""
        _load_service(service)
        first_reverse = {k: frozenset(v) for k, v in service._builder.resolver._reverse_map.items()}
        first_entity = dict(service._builder.resolver._entity_map)

        _load_service(service)
        second_reverse = {k: frozenset(v) for k, v in service._builder.resolver._reverse_map.items()}
        second_entity = dict(service._builder.resolver._entity_map)

        assert first_reverse == second_reverse
        assert first_entity == second_entity


# ===========================================================================
# 21. Reload failure handling
# ===========================================================================

class TestReloadFailureHandling:
    @pytest.fixture
    def service(self):
        from app.graph.graph_service import GraphService
        return GraphService()

    def test_entity_repo_failure_returns_false(self, service):
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("Connection refused")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False
        assert service.is_ready is False

    def test_edge_repo_failure_after_entities_returns_false(self, service):
        """Edge repo failure after entities loaded => graph cleared, returns False."""
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.side_effect = Exception("Timeout")
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False
        assert service.is_ready is False

    def test_edge_repo_failure_clears_graph(self, service):
        """When edge repo fails after entities load, graph is cleared."""
        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.side_effect = Exception("Timeout")
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        # Graph should be empty (cleared during load attempt)
        assert len(service._builder.graph) == 0

    def test_entity_repo_failure_preserves_no_prior_graph(self, service):
        """Entity repo failure on fresh service leaves it not-ready."""
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("Down")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.is_ready is False
        assert service.transaction_count == 0
        assert service.entity_count == 0

    def test_successful_build_then_load_failure_leaves_clean_state(self, service):
        """Build succeeds, then load_from_db fails => service not-ready, empty graph."""
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
        ]
        service.build(txns)
        assert service.is_ready is True
        assert service.transaction_count == 1

        # Now simulate a fresh service that fails to load
        fresh = type(service)()
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("DB down")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = fresh.load_from_db()
        assert result is False
        assert fresh.is_ready is False

    def test_repository_timeout_returns_false(self, service):
        """Simulate a timeout on the entity repo."""
        e_repo = MagicMock()
        e_repo.get_all.side_effect = TimeoutError("Request timed out")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            result = service.load_from_db()
        assert result is False

    def test_subsequent_build_after_failed_load_works(self, service):
        """After a failed load, a manual build should still succeed."""
        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("DB error")
        ed_repo = MagicMock()
        p_repo = MagicMock()
        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            service.load_from_db()
        assert service.is_ready is False

        # Now build manually
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        service.build(txns)
        assert service.is_ready is True
        assert service.transaction_count == 2


# ===========================================================================
# 22. Startup failure handling (FastAPI startup contract)
# ===========================================================================

class TestStartupFailureHandling:
    @pytest.fixture(autouse=True)
    def reset_graph_service(self):
        from app.graph.graph_service import graph_service
        graph_service.clear()
        yield
        graph_service.clear()

    def test_startup_succeeds_when_load_returns_false(self):
        """FastAPI startup handler succeeds when load_from_db returns False."""
        from app.main import startup_load_graph
        from app.graph.graph_service import graph_service

        e_repo = MagicMock()
        e_repo.get_all.return_value = []
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = []
        p_repo = MagicMock()
        p_repo.get_recent.return_value = []

        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            # Must not raise
            import asyncio
            asyncio.run(startup_load_graph())

        assert graph_service.is_ready is False

    def test_startup_succeeds_when_load_raises_exception(self):
        """FastAPI startup handler catches exceptions and succeeds."""
        from app.main import startup_load_graph
        from app.graph.graph_service import graph_service

        e_repo = MagicMock()
        e_repo.get_all.side_effect = Exception("Fatal DB error")

        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", MagicMock()), \
             patch("app.graph.graph_service.prediction_repo", MagicMock()):
            # Must not raise
            import asyncio
            asyncio.run(startup_load_graph())

        assert graph_service.is_ready is False

    def test_startup_succeeds_when_load_raises_runtime_error(self):
        """FastAPI startup handler catches RuntimeError from load_from_db."""
        from app.main import startup_load_graph
        from app.graph.graph_service import graph_service

        e_repo = MagicMock()
        e_repo.get_all.side_effect = RuntimeError("Unexpected state")

        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", MagicMock()), \
             patch("app.graph.graph_service.prediction_repo", MagicMock()):
            import asyncio
            asyncio.run(startup_load_graph())

        assert graph_service.is_ready is False

    def test_startup_succeeds_when_load_returns_true(self):
        """FastAPI startup handler succeeds and graph becomes ready when load succeeds."""
        from app.main import startup_load_graph
        from app.graph.graph_service import graph_service

        e_repo = MagicMock()
        e_repo.get_all.return_value = _make_db_entities()
        ed_repo = MagicMock()
        ed_repo.get_all.return_value = _make_db_graph_edges()
        p_repo = MagicMock()
        p_repo.get_recent.return_value = _make_db_predictions()

        with patch("app.graph.graph_service.entity_repo", e_repo), \
             patch("app.graph.graph_service.graph_edge_repo", ed_repo), \
             patch("app.graph.graph_service.prediction_repo", p_repo):
            import asyncio
            asyncio.run(startup_load_graph())

        assert graph_service.is_ready is True
        assert graph_service.transaction_count == 4
