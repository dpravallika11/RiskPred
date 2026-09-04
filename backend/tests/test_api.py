import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.main import app


class TestPredictionAPI:
    """Tests for the /predict endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _valid_payload(self, **overrides):
        base = {
            'transaction_id': 'API_TEST_001',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        }
        base.update(overrides)
        return base

    def test_predict_returns_200(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        assert response.status_code == 200

    def test_predict_returns_fraud_probability(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert 'fraud_probability' in data
        assert 0 <= data['fraud_probability'] <= 1

    def test_predict_returns_risk_score(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert 'risk_score' in data
        assert 0 <= data['risk_score'] <= 100

    def test_predict_returns_risk_level(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert data['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_predict_returns_recommended_action(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert data['recommended_action'] in ('ALLOW', 'VERIFY', 'MANUAL_REVIEW')

    def test_predict_returns_risk_factors(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert isinstance(data['top_risk_factors'], list)

    def test_predict_returns_risk_reducers(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert isinstance(data['top_risk_reducers'], list)

    def test_predict_returns_timestamp(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert 'prediction_timestamp' in data

    def test_predict_returns_transaction_id(self, client):
        response = client.post("/api/v1/predict", json=self._valid_payload())
        data = response.json()
        assert data['transaction_id'] == 'API_TEST_001'

    def test_predict_rejects_missing_amount(self, client):
        payload = {
            'transaction_id': 'TEST',
            'merchant_id': 'M1',
            'customer_id': 'C1',
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_zero_amount(self, client):
        payload = self._valid_payload(amount=0)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_negative_amount(self, client):
        payload = self._valid_payload(amount=-100)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_accepts_high_risk_transaction(self, client):
        payload = self._valid_payload(
            amount=50000,
            velocity_5m=10,
            failed_attempts_24h=5,
            is_new_device=True,
            is_new_location=True,
        )
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')


class TestGraphAPI:
    """Tests for graph API endpoints."""

    @pytest.fixture(autouse=True)
    def reset_graph_service(self):
        from app.graph.graph_service import graph_service
        graph_service.clear()
        yield
        graph_service.clear()

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _txns(self):
        return [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'amount': 100, 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'amount': 200, 'device_id': 'DEV1', 'card1': 200},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'amount': 50, 'device_id': 'DEV2'},
        ]

    def test_graph_status_before_build(self, client):
        response = client.get("/api/v1/graph/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'not_built'
        assert data['transaction_count'] == 0

    def test_graph_build(self, client):
        response = client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'built'
        assert data['transaction_count'] == 3
        assert data['entity_count'] > 0
        assert data['edge_count'] > 0
        assert 'build_timestamp' in data

    def test_graph_build_and_status_ready(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/status")
        data = response.json()
        assert data['status'] == 'ready'
        assert data['transaction_count'] == 3
        assert 'last_built' in data

    def test_graph_transaction_info(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert data['entity_count'] > 0
        assert isinstance(data['entities'], list)

    def test_graph_transaction_connections(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/connections")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert data['total_connections'] > 0
        assert isinstance(data['connected_transactions'], list)

    def test_graph_neighborhood(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert 'nodes' in data
        assert 'edges' in data

    def test_graph_risk(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/risk")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert 'ml_risk_score' in data
        assert 'network_risk_score' in data
        assert 'combined_risk_score' in data
        assert 'combined_risk_level' in data

    def test_graph_clusters(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters")
        assert response.status_code == 200
        data = response.json()
        assert data['total_clusters'] > 0
        assert isinstance(data['clusters'], list)

    def test_graph_cluster_for_transaction(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters/T1")
        assert response.status_code == 200
        data = response.json()
        assert 'T1' in data['transaction_ids']
        assert 'risk_level' in data
        assert 'suspicious_ratio' in data
        assert 'avg_risk_score' in data

    def test_graph_cluster_not_found(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters/T999")
        assert response.status_code == 404

    def test_graph_not_built_returns_503_transaction(self, client):
        response = client.get("/api/v1/graph/transaction/T1")
        assert response.status_code == 503

    def test_graph_not_built_returns_503_connections(self, client):
        response = client.get("/api/v1/graph/transaction/T1/connections")
        assert response.status_code == 503

    def test_graph_not_built_returns_503_risk(self, client):
        response = client.get("/api/v1/graph/transaction/T1/risk")
        assert response.status_code == 503

    def test_graph_not_built_returns_503_neighborhood(self, client):
        response = client.get("/api/v1/graph/transaction/T1/neighborhood")
        assert response.status_code == 503

    def test_graph_not_built_returns_503_clusters(self, client):
        response = client.get("/api/v1/graph/clusters")
        assert response.status_code == 503

    def test_graph_not_built_returns_503_cluster_lookup(self, client):
        response = client.get("/api/v1/graph/clusters/T1")
        assert response.status_code == 503

    def test_invalid_max_hops_negative(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=-1")
        assert response.status_code == 422

    def test_invalid_max_hops_too_large(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=10")
        assert response.status_code == 422

    def test_max_hops_valid_boundary_zero(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=0")
        assert response.status_code == 200

    def test_max_hops_valid_boundary_five(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=5")
        assert response.status_code == 200

    def test_graph_build_empty_transactions(self, client):
        response = client.post("/api/v1/graph/build", json={"transactions": []})
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_count'] == 0

    def test_graph_risk_response_fields(self, client):
        """Graph risk response contains ML, network and combined risk fields."""
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/risk")
        data = response.json()
        assert 'ml_risk_score' in data
        assert 'ml_risk_level' in data
        assert 'network_risk_score' in data
        assert 'network_risk_level' in data
        assert 'combined_risk_score' in data
        assert 'combined_risk_level' in data
        assert 'factors' in data
        assert 'neighbor_count' in data
        assert 'suspicious_neighbor_count' in data
        assert 'graph_available' in data

    def test_graph_risk_formula_via_api(self, client):
        """Verify the 70/30 formula via the API endpoint."""
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/risk")
        data = response.json()
        ml = data['ml_risk_score']
        net = data['network_risk_score']
        expected = round(0.70 * ml + 0.30 * net, 2)
        assert abs(data['combined_risk_score'] - expected) < 0.01

    def test_predict_still_works(self, client):
        payload = {
            'transaction_id': 'API_TEST_001',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200

    def test_health_still_works(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

    def test_graph_clusters_before_build(self, client):
        response = client.get("/api/v1/graph/clusters")
        assert response.status_code == 503

    def test_graph_cluster_lookup_before_build(self, client):
        response = client.get("/api/v1/graph/clusters/T1")
        assert response.status_code == 503
