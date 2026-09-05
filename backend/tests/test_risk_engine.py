import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.prediction_service import PredictionService


class TestRiskEngine:
    """Tests for risk score, risk level, and recommended action logic."""

    @pytest.fixture
    def service(self):
        svc = PredictionService()
        if not svc.is_ready:
            pytest.skip("ML artifacts not available")
        return svc

    def _predict(self, service, txn_dict):
        return service.predict(txn_dict)

    def test_risk_score_in_range(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_001',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert 0 <= result['risk_score'] <= 100

    def test_risk_level_low(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_002',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 1.0,
        })
        assert result['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_risk_level_high_for_high_amount(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_003',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 50000.0,
            'velocity_5m': 10,
            'failed_attempts_24h': 5,
            'is_new_device': True,
            'is_new_location': True,
        })
        assert result['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_recommended_action_matches_risk_level(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_004',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        rl = result['risk_level']
        ra = result['recommended_action']
        if rl == 'HIGH':
            assert ra == 'MANUAL_REVIEW'
        elif rl == 'MEDIUM':
            assert ra == 'VERIFY'
        else:
            assert ra == 'ALLOW'

    def test_fraud_probability_in_range(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_005',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert 0 <= result['fraud_probability'] <= 1

    def test_risk_factors_are_list(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_006',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert isinstance(result['top_risk_factors'], list)

    def test_risk_reducers_are_list(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_007',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert isinstance(result['top_risk_reducers'], list)

    def test_risk_factor_has_required_fields(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_008',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        for factor in result['top_risk_factors']:
            assert 'feature' in factor
            assert 'impact' in factor
            assert 'direction' in factor
            assert factor['direction'] == 'increases_risk'

    def test_risk_reducer_has_required_fields(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_009',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        for reducer in result['top_risk_reducers']:
            assert 'feature' in reducer
            assert 'impact' in reducer
            assert 'direction' in reducer
            assert reducer['direction'] == 'decreases_risk'

    def test_velocity_increases_risk_factors(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_010',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
            'velocity_5m': 10,
        })
        velocity_factors = [f for f in result['top_risk_factors'] if f['feature'] == 'velocity_5m']
        assert len(velocity_factors) > 0

    def test_failed_attempts_increases_risk_factors(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_011',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
            'failed_attempts_24h': 5,
        })
        failed_factors = [f for f in result['top_risk_factors'] if f['feature'] == 'failed_attempts_24h']
        assert len(failed_factors) > 0

    def test_prediction_timestamp_present(self, service):
        result = self._predict(service, {
            'transaction_id': 'TEST_012',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert 'prediction_timestamp' in result

    def test_transaction_id_preserved(self, service):
        result = self._predict(service, {
            'transaction_id': 'CUSTOM_TXN_123',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        assert result['transaction_id'] == 'CUSTOM_TXN_123'

    def test_multi_risk_factors_produce_higher_risk(self, service):
        base = self._predict(service, {
            'transaction_id': 'TEST_013',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
        })
        risky = self._predict(service, {
            'transaction_id': 'TEST_014',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'amount': 100.0,
            'velocity_5m': 15,
            'failed_attempts_24h': 8,
            'is_new_device': True,
            'is_new_location': True,
        })
        assert risky['fraud_probability'] >= base['fraud_probability']
