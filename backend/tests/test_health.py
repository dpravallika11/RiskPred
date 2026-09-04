import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.main import app


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        response = self.client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_service_name(self):
        response = self.client.get("/api/v1/health")
        data = response.json()
        assert data["service"] == "RiskPred API"

    def test_health_returns_version(self):
        response = self.client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"

    def test_health_has_required_fields(self):
        response = self.client.get("/api/v1/health")
        data = response.json()
        required_fields = ["status", "service", "version"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RiskPred" in data["message"]
