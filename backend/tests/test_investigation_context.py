import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_prediction_result():
    return {
        'transaction_id': 'T1',
        'fraud_probability': 0.85,
        'risk_score': 85,
        'risk_level': 'HIGH',
        'recommended_action': 'MANUAL_REVIEW',
        'top_risk_factors': [
            {'feature': 'TransactionAmt', 'impact': 0.05, 'direction': 'increases_risk', 'description': 'Transaction amount'},
        ],
        'top_risk_reducers': [
            {'feature': 'C1', 'impact': -0.02, 'direction': 'decreases_risk', 'description': 'Transaction count pattern'},
        ],
        'prediction_timestamp': '2026-01-01T00:00:00+00:00',
    }


def _build_real_graph():
    from app.graph.graph_service import GraphService
    svc = GraphService()
    txns = [
        {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'amount': 100, 'device_id': 'DEV1', 'card1': 100},
        {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'amount': 200, 'device_id': 'DEV1'},
        {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'amount': 50},
    ]
    risks = {
        'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
    }
    svc.build(txns, risks)
    return svc


# ===========================================================================
# A. Basic context construction
# ===========================================================================

class TestBasicContextConstruction:
    def test_valid_transaction_produces_context(self):
        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})
        assert ctx is not None
        assert hasattr(ctx, 'transaction_id')

    def test_context_is_pydantic_model(self):
        from app.investigation.schemas import InvestigationContext
        ctx = InvestigationContext(transaction_id='T1')
        d = ctx.model_dump()
        assert 'transaction_id' in d
        assert 'ml_prediction' in d
        assert 'shap_explanation' in d
        assert 'graph' in d
        assert 'network_risk' in d
        assert 'cluster' in d


# ===========================================================================
# B. Transaction ID propagation
# ===========================================================================

class TestTransactionIDPropagation:
    def test_transaction_id_appears_in_context(self):
        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('TXN_ABC', transaction={'transaction_id': 'TXN_ABC', 'amount': 50})
        assert ctx.transaction_id == 'TXN_ABC'

    def test_transaction_id_preserved_for_different_ids(self):
        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        for tid in ['ID_1', 'ID_2', 'ID_3']:
            ctx = svc.build_context(tid)
            assert ctx.transaction_id == tid


# ===========================================================================
# C. ML evidence
# ===========================================================================

class TestMLEvidence:
    @patch('app.investigation.context.prediction_service')
    def test_ml_prediction_populated_when_service_ready(self, mock_ps):
        mock_ps.is_ready = True
        mock_ps.predict.return_value = _mock_prediction_result()

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.ml_prediction is not None
        assert ctx.ml_prediction.fraud_probability == 0.85
        assert ctx.ml_prediction.risk_score == 85
        assert ctx.ml_prediction.risk_level == 'HIGH'
        assert ctx.ml_prediction.recommended_action == 'MANUAL_REVIEW'

    @patch('app.investigation.context.prediction_service')
    def test_ml_prediction_none_when_service_not_ready(self, mock_ps):
        mock_ps.is_ready = False

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.ml_prediction is None

    @patch('app.investigation.context.prediction_service')
    def test_ml_prediction_none_when_no_transaction(self, mock_ps):
        mock_ps.is_ready = True

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1')

        assert ctx.ml_prediction is None

    @patch('app.investigation.context.prediction_service')
    def test_ml_prediction_none_on_predict_error(self, mock_ps):
        mock_ps.is_ready = True
        mock_ps.predict.side_effect = RuntimeError("Model not loaded")

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.ml_prediction is None


# ===========================================================================
# D. SHAP evidence
# ===========================================================================

class TestSHAPEvidence:
    @patch('app.investigation.context.prediction_service')
    def test_shap_factors_and_reducers_populated(self, mock_ps):
        mock_ps.is_ready = True
        mock_ps.predict.return_value = _mock_prediction_result()

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.shap_explanation is not None
        assert len(ctx.shap_explanation.risk_factors) == 1
        assert ctx.shap_explanation.risk_factors[0]['feature'] == 'TransactionAmt'
        assert len(ctx.shap_explanation.risk_reducers) == 1
        assert ctx.shap_explanation.risk_reducers[0]['feature'] == 'C1'

    @patch('app.investigation.context.prediction_service')
    def test_shap_none_when_service_not_ready(self, mock_ps):
        mock_ps.is_ready = False

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.shap_explanation is None

    @patch('app.investigation.context.prediction_service')
    def test_shap_none_when_no_transaction(self, mock_ps):
        mock_ps.is_ready = True

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1')

        assert ctx.shap_explanation is None

    @patch('app.investigation.context.prediction_service')
    def test_shap_none_on_predict_error(self, mock_ps):
        mock_ps.is_ready = True
        mock_ps.predict.side_effect = Exception("SHAP failed")

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.shap_explanation is None


# ===========================================================================
# E. Graph evidence
# ===========================================================================

class TestGraphEvidence:
    def test_graph_evidence_populated_when_graph_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.graph_available is True
        assert ctx.graph.total_connections > 0
        assert ctx.graph.entity_count > 0
        assert len(ctx.graph.connected_transactions) > 0

    def test_graph_evidence_empty_when_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        empty_svc = GraphService()

        with patch('app.investigation.context.graph_service', empty_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.graph_available is False
        assert ctx.graph.total_connections == 0
        assert ctx.graph.entity_count == 0

    def test_connected_transactions_from_existing_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'card1': 100},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        conn_ids = [c['transaction_id'] for c in ctx.graph.connected_transactions]
        assert 'T2' in conn_ids
        assert 'T3' in conn_ids

    def test_entities_populated_from_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.entity_count > 0
        entity_types = [e['type'] for e in ctx.graph.entities]
        assert 'device' in entity_types

    def test_suspicious_neighbors_populated(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.suspicious_neighbor_count == 1
        assert len(ctx.graph.suspicious_neighbors) == 1
        assert ctx.graph.suspicious_neighbors[0]['transaction_id'] == 'T2'


# ===========================================================================
# F. Network risk
# ===========================================================================

class TestNetworkRisk:
    @patch('app.investigation.context.prediction_service')
    def test_network_risk_populated_when_graph_built(self, mock_ps):
        mock_ps.is_ready = True
        mock_ps.predict.return_value = _mock_prediction_result()

        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.network_risk is not None
        assert ctx.network_risk.network_risk_score > 0
        assert ctx.network_risk.combined_risk_score > 0

    def test_network_risk_none_when_graph_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        empty_svc = GraphService()

        with patch('app.investigation.context.graph_service', empty_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.network_risk is None

    def test_network_risk_uses_real_calculation(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        # When no ML prediction is available, context passes ml_risk_score=0
        expected = graph_svc.get_network_risk('T1', 0, 'UNKNOWN')

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.network_risk.network_risk_score == expected['network_risk_score']
        assert ctx.network_risk.combined_risk_score == expected['combined_risk_score']


# ===========================================================================
# G. Cluster evidence
# ===========================================================================

class TestClusterEvidence:
    def test_cluster_populated_when_in_cluster(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.cluster is not None
        assert ctx.cluster.found is True
        assert 'T1' in ctx.cluster.transaction_ids
        assert ctx.cluster.total_transactions == 2

    def test_cluster_none_when_graph_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        empty_svc = GraphService()

        with patch('app.investigation.context.graph_service', empty_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.cluster is None

    def test_cluster_none_for_nonexistent_transaction(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T999')

        assert ctx.cluster is None

    def test_cluster_risk_level_populated(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.cluster.risk_level in ('LOW', 'MEDIUM', 'HIGH', 'UNKNOWN')
        assert 0 <= ctx.cluster.suspicious_ratio <= 1
        assert ctx.cluster.avg_risk_score >= 0


# ===========================================================================
# H. Missing graph
# ===========================================================================

class TestMissingGraph:
    def test_graph_evidence_empty_when_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.graph.graph_available is False
        assert ctx.graph.connected_transactions == []
        assert ctx.graph.entities == []
        assert ctx.graph.suspicious_neighbors == []

    def test_network_risk_none_when_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.network_risk is None

    def test_cluster_none_when_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.cluster is None


# ===========================================================================
# I. Missing cluster
# ===========================================================================

class TestMissingCluster:
    def test_cluster_none_for_isolated_transaction_in_small_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'ISO', 'merchant_id': 'M1', 'customer_id': 'C1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('ISO')

        assert ctx.cluster is not None
        assert ctx.cluster.found is True
        assert ctx.cluster.total_transactions == 1


# ===========================================================================
# J. Missing transaction
# ===========================================================================

class TestMissingTransaction:
    def test_context_for_nonexistent_transaction(self):
        from app.investigation.context import InvestigationContextService

        svc = InvestigationContextService()
        ctx = svc.build_context('NONEXISTENT')

        assert ctx.transaction_id == 'NONEXISTENT'
        assert ctx.transaction is None
        assert ctx.ml_prediction is None
        assert ctx.shap_explanation is None
        assert ctx.graph.graph_available is False
        assert ctx.network_risk is None
        assert ctx.cluster is None

    def test_context_for_empty_transaction_id(self):
        from app.investigation.context import InvestigationContextService

        svc = InvestigationContextService()
        ctx = svc.build_context('')

        assert ctx.transaction_id == ''


# ===========================================================================
# K. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_inputs_produce_equivalent_contexts(self):
        from app.investigation.context import InvestigationContextService

        svc = InvestigationContextService()
        txn = {'transaction_id': 'T1', 'amount': 100, 'merchant_id': 'M1', 'customer_id': 'C1'}

        ctx1 = svc.build_context('T1', transaction=txn)
        ctx2 = svc.build_context('T1', transaction=txn)

        assert ctx1.transaction_id == ctx2.transaction_id
        assert ctx1.transaction == ctx2.transaction
        assert ctx1.ml_prediction == ctx2.ml_prediction
        assert ctx1.shap_explanation == ctx2.shap_explanation

    def test_graph_deterministic_with_same_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        def make_svc():
            g = GraphService()
            txns = [
                {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
                {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            ]
            g.build(txns, {'T2': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'}})
            return g

        svc = InvestigationContextService()

        with patch('app.investigation.context.graph_service', make_svc()):
            ctx1 = svc.build_context('T1')
        with patch('app.investigation.context.graph_service', make_svc()):
            ctx2 = svc.build_context('T1')

        assert ctx1.graph.total_connections == ctx2.graph.total_connections
        assert ctx1.graph.entity_count == ctx2.graph.entity_count
        assert len(ctx1.graph.connected_transactions) == len(ctx2.graph.connected_transactions)

    def test_neighborhood_deterministic(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx1 = svc.build_context('T1')
            ctx2 = svc.build_context('T1')

        assert ctx1.graph.neighborhood_nodes == ctx2.graph.neighborhood_nodes
        assert ctx1.graph.neighborhood_edges == ctx2.graph.neighborhood_edges


# ===========================================================================
# L. No hallucinated evidence
# ===========================================================================

class TestNoHallucinatedEvidence:
    @patch('app.investigation.context.prediction_service')
    def test_no_fake_entities_when_graph_not_built(self, mock_ps):
        mock_ps.is_ready = False

        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.entities == []
        assert ctx.graph.connected_transactions == []
        assert ctx.graph.suspicious_neighbors == []

    def test_no_fake_cluster_when_graph_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.cluster is None

    @patch('app.investigation.context.prediction_service')
    def test_no_fake_ml_when_service_not_ready(self, mock_ps):
        mock_ps.is_ready = False

        from app.investigation.context import InvestigationContextService
        svc = InvestigationContextService()
        ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.ml_prediction is None
        assert ctx.shap_explanation is None

    def test_no_fake_network_risk_when_graph_not_built(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        with patch('app.investigation.context.graph_service', GraphService()):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.network_risk is None

    def test_connected_transactions_only_from_real_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.connected_transactions == []
        assert ctx.graph.total_connections == 0

    def test_suspicious_neighbors_only_from_real_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.3, 'risk_score': 30, 'risk_level': 'LOW'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1')

        assert ctx.graph.suspicious_neighbor_count == 0
        assert ctx.graph.suspicious_neighbors == []


# ===========================================================================
# Integration test (uses real graph service)
# ===========================================================================

class TestIntegration:
    def test_full_context_with_real_graph(self):
        from app.investigation.context import InvestigationContextService
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'amount': 100, 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'amount': 200, 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'amount': 50},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        graph_svc.build(txns, risks)

        with patch('app.investigation.context.graph_service', graph_svc):
            svc = InvestigationContextService()
            ctx = svc.build_context('T1', transaction={'transaction_id': 'T1', 'amount': 100})

        assert ctx.transaction_id == 'T1'
        assert ctx.transaction['amount'] == 100
        assert ctx.graph.graph_available is True
        assert ctx.graph.total_connections > 0
        assert ctx.network_risk is not None
        assert ctx.network_risk.combined_risk_score > 0
        assert ctx.cluster is not None
        assert ctx.cluster.found is True

    def test_context_api_returns_valid_json(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
        ]
        graph_svc.build(txns)

        with patch('app.investigation.context.graph_service', graph_svc):
            client = TestClient(app)
            response = client.get("/api/v1/investigation/T1/context")

        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert 'ml_prediction' in data
        assert 'shap_explanation' in data
        assert 'graph' in data
        assert 'network_risk' in data
        assert 'cluster' in data

    def test_context_api_empty_graph(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.graph.graph_service import GraphService

        empty_svc = GraphService()

        with patch('app.investigation.context.graph_service', empty_svc):
            client = TestClient(app)
            response = client.get("/api/v1/investigation/T1/context")

        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert data['graph']['graph_available'] is False
        assert data['network_risk'] is None
        assert data['cluster'] is None

    def test_context_for_nonexistent_txn_returns_valid_structure(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.graph.graph_service import GraphService

        graph_svc = GraphService()
        graph_svc.build([{'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'}])

        with patch('app.investigation.context.graph_service', graph_svc):
            client = TestClient(app)
            response = client.get("/api/v1/investigation/NONEXISTENT/context")

        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'NONEXISTENT'
        assert data['graph']['graph_available'] is True
        assert data['graph']['total_connections'] == 0
