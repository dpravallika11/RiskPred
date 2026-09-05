"""Tests for Supabase integration.

Uses mocks for the Supabase client so no live connection is required.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_txn(txn_id: str = "T1", **overrides) -> Dict[str, Any]:
    base = {
        "transaction_id": txn_id,
        "merchant_id": "M1",
        "customer_id": "C1",
        "amount": 100.0,
        "device_id": "DEV1",
        "is_new_device": False,
        "location": "US",
        "is_new_location": False,
        "payment_method": "credit_card",
        "velocity_5m": 1,
        "failed_attempts_24h": 0,
    }
    base.update(overrides)
    return base


def _mock_supabase_response(data=None, count=None):
    """Create a mock Supabase response object."""
    resp = MagicMock()
    resp.data = data if data is not None else []
    resp.count = count
    return resp


def _mock_sb_table():
    """Create a mock Supabase table with chainable query builder."""
    table = MagicMock()
    # Chain: table.select().eq().execute() -> response
    chain = MagicMock()
    table.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.range.return_value = chain
    chain.execute.return_value = _mock_supabase_response([])
    # Chain: table.insert().execute() -> response
    table.insert.return_value.execute.return_value = _mock_supabase_response([{"id": "fake-uuid"}])
    # Chain: table.update().eq().execute() -> response
    table.update.return_value = chain
    # Chain: table.delete().eq().execute() -> response
    table.delete.return_value = chain
    return table


# ===========================================================================
# 1. TransactionRepository round-trip
# ===========================================================================

class TestTransactionRepository:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_create_returnsInsertedData(self):
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        txn = _make_txn("T1")
        inserted = {**txn, "id": "uuid-1"}
        self.mock_table.insert.return_value.execute.return_value = _mock_supabase_response([inserted])

        result = repo.create(txn)

        assert result is not None
        assert result["transaction_id"] == "T1"
        self.mock_table.insert.assert_called_once_with(txn)

    def test_get_by_transaction_id_found(self):
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        stored = {**_make_txn("T1"), "id": "uuid-1"}
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([stored])

        result = repo.get_by_transaction_id("T1")

        assert result is not None
        assert result["transaction_id"] == "T1"

    def test_get_by_transaction_id_notFound(self):
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        self.mock_table.select.return_value.eq.return_value.execute.return_value = _mock_supabase_response([])

        result = repo.get_by_transaction_id("NONEXISTENT")

        assert result is None

    def test_get_all_returnsList(self):
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        txns = [_make_txn("T1"), _make_txn("T2")]
        self.mock_table.select.return_value.range.return_value.order.return_value.execute.return_value = _mock_supabase_response(txns)

        result = repo.get_all()

        assert len(result) == 2
        assert result[0]["transaction_id"] == "T1"


# ===========================================================================
# 2. PredictionRepository create mapping
# ===========================================================================

class TestPredictionRepository:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_create_prediction(self):
        from app.db.repositories import PredictionRepository
        repo = PredictionRepository()
        pred = {
            "transaction_id": "T1",
            "fraud_probability": 0.85,
            "risk_score": 85,
            "risk_level": "HIGH",
            "recommended_action": "MANUAL_REVIEW",
        }
        inserted = {**pred, "id": "pred-uuid-1"}
        self.mock_table.insert.return_value.execute.return_value = _mock_supabase_response([inserted])

        result = repo.create(pred)

        assert result is not None
        assert result["transaction_id"] == "T1"
        assert result["risk_level"] == "HIGH"

    def test_get_by_transaction_id_returnsLatest(self):
        from app.db.repositories import PredictionRepository
        repo = PredictionRepository()
        pred = {"transaction_id": "T1", "risk_score": 85}
        self.mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = _mock_supabase_response([pred])

        result = repo.get_by_transaction_id("T1")

        assert result is not None
        assert result["risk_score"] == 85


# ===========================================================================
# 3. InvestigationRepository create/get round-trip
# ===========================================================================

class TestInvestigationRepository:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_create_investigation(self):
        from app.db.repositories import InvestigationRepository
        repo = InvestigationRepository()
        inv = {"transaction_id": "T1", "status": "completed", "conclusion": "Low risk."}
        inserted = {**inv, "id": "inv-uuid-1"}
        self.mock_table.insert.return_value.execute.return_value = _mock_supabase_response([inserted])

        result = repo.create(inv)

        assert result is not None
        assert result["transaction_id"] == "T1"

    def test_get_by_transaction_id(self):
        from app.db.repositories import InvestigationRepository
        repo = InvestigationRepository()
        inv = {"transaction_id": "T1", "status": "completed"}
        self.mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = _mock_supabase_response([inv])

        result = repo.get_by_transaction_id("T1")

        assert result is not None
        assert result["status"] == "completed"


# ===========================================================================
# 4. Investigation API fallback: store empty, Supabase has transaction
# ===========================================================================

class TestInvestigationFallback:
    def setup_method(self):
        self.patcher_store = patch("app.api.routes.investigation.transaction_store")
        self.patcher_repo = patch("app.api.routes.investigation.transaction_repo")
        self.mock_store = self.patcher_store.start()
        self.mock_repo = self.patcher_repo.start()

    def teardown_method(self):
        self.patcher_store.stop()
        self.patcher_repo.stop()

    def test_fallback_toSupabase_whenStoreEmpty(self):
        from app.api.routes.investigation import _lookup_transaction
        self.mock_store.get.return_value = None
        stored_txn = _make_txn("T1")
        self.mock_repo.get_by_transaction_id.return_value = stored_txn

        result = _lookup_transaction("T1")

        assert result is not None
        assert result["transaction_id"] == "T1"
        self.mock_store.put.assert_called_once_with(stored_txn)

    def test_usesStore_whenAvailable(self):
        from app.api.routes.investigation import _lookup_transaction
        stored_txn = _make_txn("T1")
        self.mock_store.get.return_value = stored_txn

        result = _lookup_transaction("T1")

        assert result is not None
        assert result["transaction_id"] == "T1"
        self.mock_repo.get_by_transaction_id.assert_not_called()

    def test_returnsNone_whenNotFoundAnywhere(self):
        from app.api.routes.investigation import _lookup_transaction
        self.mock_store.get.return_value = None
        self.mock_repo.get_by_transaction_id.return_value = None

        result = _lookup_transaction("NONEXISTENT")

        assert result is None


# ===========================================================================
# 5. Investigation API works from store without DB lookup
# ===========================================================================

class TestInvestigationFromStore:
    def setup_method(self):
        self.patcher_store = patch("app.api.routes.investigation.transaction_store")
        self.patcher_repo = patch("app.api.routes.investigation.transaction_repo")
        self.mock_store = self.patcher_store.start()
        self.mock_repo = self.patcher_repo.start()

    def teardown_method(self):
        self.patcher_store.stop()
        self.patcher_repo.stop()

    def test_storeHit_skipsRepo(self):
        from app.api.routes.investigation import _lookup_transaction
        txn = _make_txn("T1")
        self.mock_store.get.return_value = txn

        result = _lookup_transaction("T1")

        assert result["transaction_id"] == "T1"
        self.mock_repo.get_by_transaction_id.assert_not_called()


# ===========================================================================
# 6. Duplicate transaction handling
# ===========================================================================

class TestDuplicateTransactionHandling:
    def setup_method(self):
        self.patcher = patch("app.db.repositories.get_supabase")
        self.mock_get_sb = self.patcher.start()
        self.mock_table = _mock_sb_table()
        self.mock_get_sb.return_value.table.return_value = self.mock_table

    def teardown_method(self):
        self.patcher.stop()

    def test_create_duplicate_raisesFromSupabase(self):
        """Duplicate key violation should propagate as an exception from Supabase."""
        from app.db.repositories import TransactionRepository
        repo = TransactionRepository()
        self.mock_table.insert.return_value.execute.side_effect = Exception(
            "duplicate key value violates unique constraint"
        )

        with pytest.raises(Exception, match="duplicate key"):
            repo.create(_make_txn("T1"))


# ===========================================================================
# 7. Supabase persistence failure handling
# ===========================================================================

class TestSupabaseFailureHandling:
    def setup_method(self):
        self.patcher_store = patch("app.api.routes.investigation.transaction_store")
        self.patcher_repo = patch("app.api.routes.investigation.transaction_repo")
        self.mock_store = self.patcher_store.start()
        self.mock_repo = self.patcher_repo.start()

    def teardown_method(self):
        self.patcher_store.stop()
        self.patcher_repo.stop()

    def test_fallback_handlesSupabaseError(self):
        """If Supabase is down, fallback should return None gracefully."""
        from app.api.routes.investigation import _lookup_transaction
        self.mock_store.get.return_value = None
        self.mock_repo.get_by_transaction_id.side_effect = Exception("Connection refused")

        result = _lookup_transaction("T1")

        assert result is None

    def test_predict_persistenceFailureDoesNotAffectResponse(self):
        """Prediction endpoint should succeed even if Supabase write fails."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)

        with patch("app.api.routes.predictions.transaction_repo") as mock_repo:
            mock_repo.create.side_effect = Exception("Supabase down")
            response = client.post("/api/v1/predict", json={
                "transaction_id": "TEST_DUP",
                "merchant_id": "M1",
                "customer_id": "C1",
                "amount": 50.0,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["transaction_id"] == "TEST_DUP"
            assert "risk_score" in data


# ===========================================================================
# 8. Prediction persistence integration tests
# ===========================================================================

class TestPredictionPersistence:
    """Tests for prediction + risk factor persistence in /predict endpoint."""

    def setup_method(self):
        self.patcher_transaction_repo = patch("app.api.routes.predictions.transaction_repo")
        self.patcher_prediction_repo = patch("app.api.routes.predictions.prediction_repo")
        self.patcher_risk_factor_repo = patch("app.api.routes.predictions.risk_factor_repo")

        self.mock_transaction_repo = self.patcher_transaction_repo.start()
        self.mock_prediction_repo = self.patcher_prediction_repo.start()
        self.mock_risk_factor_repo = self.patcher_risk_factor_repo.start()

    def teardown_method(self):
        self.patcher_transaction_repo.stop()
        self.patcher_prediction_repo.stop()
        self.patcher_risk_factor_repo.stop()

    def test_prediction_persistence_successful(self):
        """Prediction row should be persisted when prediction succeeds."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-1"}

        response = client.post("/api/v1/predict", json={
            "transaction_id": "PERSIST_TEST_001",
            "merchant_id": "M1",
            "customer_id": "C1",
            "amount": 100.0,
        })

        assert response.status_code == 200
        self.mock_prediction_repo.create.assert_called_once()
        call_args = self.mock_prediction_repo.create.call_args[0][0]
        assert call_args["transaction_id"] == "PERSIST_TEST_001"
        assert "fraud_probability" in call_args
        assert "risk_score" in call_args
        assert "risk_level" in call_args
        assert "recommended_action" in call_args
        assert "prediction_timestamp" in call_args

    def test_prediction_row_mapping(self):
        """Prediction record should map all required fields correctly."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-2"}

        response = client.post("/api/v1/predict", json={
            "transaction_id": "MAPPING_TEST_001",
            "merchant_id": "M2",
            "customer_id": "C2",
            "amount": 250.0,
        })

        assert response.status_code == 200
        call_args = self.mock_prediction_repo.create.call_args[0][0]
        assert call_args["transaction_id"] == "MAPPING_TEST_001"
        assert isinstance(call_args["fraud_probability"], float)
        assert isinstance(call_args["risk_score"], int)
        assert call_args["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert call_args["recommended_action"] in ("ALLOW", "VERIFY", "MANUAL_REVIEW")
        assert isinstance(call_args["prediction_timestamp"], str)

    def test_risk_factors_persisted(self):
        """Risk factors should be persisted with correct prediction_id."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-3"}

        response = client.post("/api/v1/predict", json={
            "transaction_id": "FACTORS_TEST_001",
            "merchant_id": "M3",
            "customer_id": "C3",
            "amount": 500.0,
        })

        assert response.status_code == 200
        self.mock_risk_factor_repo.create_many.assert_called_once()
        call_args = self.mock_risk_factor_repo.create_many.call_args[0]
        prediction_id = call_args[0]
        factors = call_args[1]
        assert prediction_id == "pred-uuid-3"
        assert isinstance(factors, list)
        assert len(factors) > 0
        for factor in factors:
            assert "feature" in factor
            assert "impact" in factor
            assert "direction" in factor
            assert "description" in factor

    def test_risk_reducers_persisted(self):
        """Risk reducers should be included in risk factor persistence."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-4"}

        response = client.post("/api/v1/predict", json={
            "transaction_id": "REDUCERS_TEST_001",
            "merchant_id": "M4",
            "customer_id": "C4",
            "amount": 150.0,
        })

        assert response.status_code == 200
        call_args = self.mock_risk_factor_repo.create_many.call_args[0]
        factors = call_args[1]
        directions = [f["direction"] for f in factors]
        assert "increases_risk" in directions
        assert "decreases_risk" in directions

    def test_persistence_failure_does_not_break_response(self):
        """Prediction response should succeed even if persistence fails."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.side_effect = Exception("DB connection lost")

        response = client.post("/api/v1/predict", json={
            "transaction_id": "FAIL_TEST_001",
            "merchant_id": "M5",
            "customer_id": "C5",
            "amount": 75.0,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "FAIL_TEST_001"
        assert "risk_score" in data

    def test_prediction_service_called_exactly_once(self):
        """prediction_service.predict() should be called only once per request."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.services import prediction_service as ps_module

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-5"}

        original_predict = ps_module.prediction_service.predict
        call_count = [0]

        def counting_predict(txn_dict):
            call_count[0] += 1
            return original_predict(txn_dict)

        with patch.object(ps_module.prediction_service, 'predict', counting_predict):
            response = client.post("/api/v1/predict", json={
                "transaction_id": "CALLCOUNT_TEST_001",
                "merchant_id": "M6",
                "customer_id": "C6",
                "amount": 200.0,
            })

        assert response.status_code == 200
        assert call_count[0] == 1

    def test_api_response_unchanged(self):
        """API response structure should remain exactly the same."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-6"}

        response = client.post("/api/v1/predict", json={
            "transaction_id": "RESPONSE_TEST_001",
            "merchant_id": "M7",
            "customer_id": "C7",
            "amount": 300.0,
        })

        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "recommended_action" in data
        assert "top_risk_factors" in data
        assert "top_risk_reducers" in data
        assert "prediction_timestamp" in data
        assert data["transaction_id"] == "RESPONSE_TEST_001"
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["risk_score"] <= 100
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_prediction_persistence_failure_logged(self):
        """Prediction persistence failure should be logged, not silenced."""
        from fastapi.testclient import TestClient
        from app.main import app
        import io
        import sys

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.side_effect = Exception("Storage full")

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            response = client.post("/api/v1/predict", json={
                "transaction_id": "LOG_TEST_001",
                "merchant_id": "M8",
                "customer_id": "C8",
                "amount": 400.0,
            })
        finally:
            sys.stdout = sys.__stdout__

        assert response.status_code == 200
        assert "[PERSISTENCE ERROR]" in captured_output.getvalue()
        assert "prediction persistence failed" in captured_output.getvalue()

    def test_risk_factor_persistence_failure_logged(self):
        """Risk factor persistence failure should be logged, not silenced."""
        from fastapi.testclient import TestClient
        from app.main import app
        import io
        import sys

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.return_value = {"id": "pred-uuid-7"}
        self.mock_risk_factor_repo.create_many.side_effect = Exception("Table locked")

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            response = client.post("/api/v1/predict", json={
                "transaction_id": "LOG_TEST_002",
                "merchant_id": "M9",
                "customer_id": "C9",
                "amount": 350.0,
            })
        finally:
            sys.stdout = sys.__stdout__

        assert response.status_code == 200
        assert "[PERSISTENCE ERROR]" in captured_output.getvalue()
        assert "risk factors persistence failed" in captured_output.getvalue()

    def test_no_secrets_in_error_logs(self):
        """Error logs should not contain sensitive information."""
        from fastapi.testclient import TestClient
        from app.main import app
        import io
        import sys

        client = TestClient(app, raise_server_exceptions=False)
        self.mock_prediction_repo.create.side_effect = Exception("Supabase key invalid")

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            response = client.post("/api/v1/predict", json={
                "transaction_id": "SECRET_TEST_001",
                "merchant_id": "M10",
                "customer_id": "C10",
                "amount": 500.0,
            })
        finally:
            sys.stdout = sys.__stdout__

        assert response.status_code == 200
        log_output = captured_output.getvalue()
        # Should not contain actual secret values like API keys, passwords, tokens
        assert "eyJ" not in log_output  # JWT token prefix
        assert "password" not in log_output.lower()
        assert "token" not in log_output.lower()
        assert "supabase_" not in log_output.lower()  # Key name prefixes
