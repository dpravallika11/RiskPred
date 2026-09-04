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
