import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEntityExtractor:
    """Tests for entity extraction from transactions."""

    @pytest.fixture
    def extractor(self):
        from app.graph.entity_extractor import EntityExtractor
        return EntityExtractor()

    def test_extract_valid_entities(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': 'DEV1',
            'card1': 100,
            'card4': 'visa',
            'P_emaildomain': 'gmail.com',
            'addr1': 315,
        }
        entities = extractor.extract(txn)
        assert 'card' in entities
        assert 'device' in entities
        assert 'email' in entities
        assert 'address' in entities
        assert 'merchant' in entities
        assert 'customer' in entities
        assert '100' in entities['card']
        assert 'visa' in entities['card']
        assert 'DEV1' in entities['device']
        assert 'gmail.com' in entities['email']
        assert '315' in entities['address']

    def test_extract_missing_identifier(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
        }
        entities = extractor.extract(txn)
        assert 'card' not in entities
        assert 'email' not in entities
        assert 'address' not in entities
        assert 'device' not in entities
        assert 'merchant' in entities
        assert 'customer' in entities

    def test_extract_nan_values(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'card1': float('nan'),
            'addr1': float('nan'),
        }
        entities = extractor.extract(txn)
        assert 'card' not in entities
        assert 'address' not in entities

    def test_extract_empty_strings(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': '',
            'card1': '',
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities
        assert 'card' not in entities

    def test_extract_UNKNOWN_values(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': 'UNKNOWN',
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities

    def test_extract_none_values(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': None,
            'card1': None,
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities
        assert 'card' not in entities

    def test_extract_batch(self, extractor):
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV2'},
        ]
        results = extractor.extract_batch(txns)
        assert len(results) == 2
        assert 'device' in results[0]
        assert 'device' in results[1]

    def test_extract_float_card_normalized(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'card1': 13926.0,
        }
        entities = extractor.extract(txn)
        assert 'card' in entities
        assert '13926' in entities['card']

    def test_extract_multiple_email_domains(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'P_emaildomain': 'gmail.com',
            'R_emaildomain': 'yahoo.com',
        }
        entities = extractor.extract(txn)
        assert 'email' in entities
        assert 'gmail.com' in entities['email']
        assert 'yahoo.com' in entities['email']

    def test_extract_r_emaildomain_only(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'R_emaildomain': 'outlook.com',
        }
        entities = extractor.extract(txn)
        assert 'email' in entities
        assert 'outlook.com' in entities['email']

    def test_extract_partial_identity_data(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'card1': 100,
            'addr1': 315,
        }
        entities = extractor.extract(txn)
        assert 'card' in entities
        assert 'address' in entities
        assert 'email' not in entities
        assert 'device' not in entities


class TestEntityResolver:
    """Tests for deterministic entity resolution."""

    @pytest.fixture
    def resolver(self):
        from app.graph.entity_resolver import EntityResolver
        return EntityResolver()

    def test_same_identifier_same_entity(self, resolver):
        key1 = resolver.link('card', '100', 'T1')
        key2 = resolver.link('card', '100', 'T2')
        assert key1 == key2

    def test_different_identifier_different_entities(self, resolver):
        key1 = resolver.link('card', '100', 'T1')
        key2 = resolver.link('card', '200', 'T2')
        assert key1 != key2

    def test_normalization(self, resolver):
        key1 = resolver.link('card', '100', 'T1')
        key2 = resolver.link('card', 100, 'T2')
        assert key1 == key2

    def test_invalid_identifier_raises(self, resolver):
        from app.graph.entity_resolver import EntityResolver
        with pytest.raises(ValueError):
            resolver.resolve('card', None)

    def test_invalid_identifier_empty_raises(self, resolver):
        from app.graph.entity_resolver import EntityResolver
        with pytest.raises(ValueError):
            resolver.resolve('card', '')

    def test_transactions_for_entity(self, resolver):
        resolver.link('card', '100', 'T1')
        resolver.link('card', '100', 'T2')
        txns = resolver.get_transactions_for_entity('card:100')
        assert 'T1' in txns
        assert 'T2' in txns

    def test_entities_for_transaction(self, resolver):
        resolver.link('card', '100', 'T1')
        resolver.link('device', 'DEV1', 'T1')
        entities = resolver.get_entities_for_transaction('T1')
        assert entities['card'] == 'card:100'
        assert entities['device'] == 'device:DEV1'

    def test_clear(self, resolver):
        resolver.link('card', '100', 'T1')
        resolver.clear()
        assert resolver.entity_count == 0
        assert resolver.transaction_count == 0

    def test_entity_count(self, resolver):
        resolver.link('card', '100', 'T1')
        resolver.link('card', '200', 'T2')
        resolver.link('device', 'DEV1', 'T1')
        assert resolver.entity_count == 3

    def test_transaction_count(self, resolver):
        resolver.link('card', '100', 'T1')
        resolver.link('card', '100', 'T2')
        resolver.link('card', '200', 'T3')
        assert resolver.transaction_count == 3

    def test_different_entity_types_same_value(self, resolver):
        key1 = resolver.link('card', '100', 'T1')
        key2 = resolver.link('device', '100', 'T1')
        assert key1 != key2


class TestGraphBuilder:
    """Tests for graph construction."""

    @pytest.fixture
    def builder(self):
        from app.graph.graph_builder import GraphBuilder
        return GraphBuilder()

    def _make_txn(self, txn_id='T1', **kwargs):
        base = {
            'transaction_id': txn_id,
            'merchant_id': 'M1',
            'customer_id': 'C1',
        }
        base.update(kwargs)
        return base

    def test_build_creates_transaction_node(self, builder):
        builder.build([self._make_txn('T1')])
        assert 'T1' in builder.graph
        assert builder.graph.nodes['T1']['node_type'] == 'transaction'

    def test_build_creates_entity_nodes(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1', card1=100)])
        assert 'device:DEV1' in builder.graph
        assert 'card:100' in builder.graph

    def test_build_creates_edges(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1')])
        assert builder.graph.has_edge('T1', 'device:DEV1')

    def test_build_shared_entity(self, builder):
        txns = [
            self._make_txn('T1', device_id='DEV1'),
            self._make_txn('T2', device_id='DEV1'),
        ]
        builder.build(txns)
        assert builder.graph.has_edge('T1', 'device:DEV1')
        assert builder.graph.has_edge('T2', 'device:DEV1')

    def test_transaction_count(self, builder):
        txns = [self._make_txn(f'T{i}') for i in range(5)]
        builder.build(txns)
        assert builder.transaction_count == 5

    def test_entity_count(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1', card1=100)])
        assert builder.entity_count == 4

    def test_edge_count(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1', card1=100)])
        assert builder.edge_count == 4

    def test_add_risk_to_transaction(self, builder):
        builder.build([self._make_txn('T1')])
        risk = {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'}
        builder.add_risk_to_transaction('T1', risk)
        assert builder.get_transaction_risk('T1') == risk
        assert builder.graph.nodes['T1']['fraud_probability'] == 0.8

    def test_isolated_transactions(self, builder):
        txns = [
            self._make_txn('T1', device_id='DEV1'),
            self._make_txn('T2'),
        ]
        builder.build(txns)
        assert 'T1' in builder.graph
        assert 'T2' in builder.graph
        assert not builder.graph.has_edge('T1', 'T2')

    def test_build_with_no_identity_data(self, builder):
        builder.build([self._make_txn('T1')])
        assert builder.transaction_count == 1
        assert builder.entity_count == 2

    def test_clear(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1')])
        builder.clear()
        assert builder.transaction_count == 0
        assert builder.entity_count == 0

    def test_build_with_risk_results(self, builder):
        risk_results = {
            'T1': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        builder.build([self._make_txn('T1', device_id='DEV1')], risk_results)
        assert builder.get_transaction_risk('T1')['risk_score'] == 90
        assert builder.graph.nodes['T1']['fraud_probability'] == 0.9

    def test_no_transaction_id_skipped(self, builder):
        builder.build([{'merchant_id': 'M1', 'customer_id': 'C1'}])
        assert builder.transaction_count == 0

    def test_edge_relationship_metadata(self, builder):
        builder.build([self._make_txn('T1', device_id='DEV1')])
        edge_data = builder.graph.edges['T1', 'device:DEV1']
        assert edge_data['relationship'] == 'device'


class TestGraphQueries:
    """Tests for graph queries and connection discovery."""

    @pytest.fixture
    def built_graph(self):
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1', 'card1': 200},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV2', 'card1': 100},
            {'transaction_id': 'T4', 'merchant_id': 'M4', 'customer_id': 'C4'},
        ]
        builder.build(txns)
        return builder

    @pytest.fixture
    def querier(self, built_graph):
        from app.graph.graph_queries import GraphQuerier
        return GraphQuerier(built_graph.graph, built_graph.resolver)

    def test_connected_transactions_via_entity(self, querier):
        connected = querier.get_connected_transactions('T1')
        conn_ids = [c['transaction_id'] for c in connected]
        assert 'T2' in conn_ids
        assert 'T3' in conn_ids

    def test_shared_entities(self, querier):
        connected = querier.get_connected_transactions('T1')
        for c in connected:
            if c['transaction_id'] == 'T2':
                entity_types = [e['type'] for e in c['shared_entities']]
                assert 'device' in entity_types
            if c['transaction_id'] == 'T3':
                entity_types = [e['type'] for e in c['shared_entities']]
                assert 'card' in entity_types

    def test_isolated_transaction(self, querier):
        connected = querier.get_connected_transactions('T4')
        assert len(connected) == 0

    def test_nonexistent_transaction(self, querier):
        connected = querier.get_connected_transactions('T999')
        assert len(connected) == 0

    def test_entity_usage_count(self, querier):
        count = querier.get_entity_usage_count('device:DEV1')
        assert count == 2

    def test_entity_usage_count_single(self, querier):
        count = querier.get_entity_usage_count('device:DEV2')
        assert count == 1

    def test_entity_usage_count_nonexistent(self, querier):
        count = querier.get_entity_usage_count('device:DEV999')
        assert count == 0

    def test_get_transaction_entities(self, querier):
        entities = querier.get_transaction_entities('T1')
        entity_types = [e['type'] for e in entities]
        assert 'device' in entity_types
        assert 'card' in entity_types
        assert 'merchant' in entity_types

    def test_get_neighborhood(self, querier):
        hood = querier.get_neighborhood('T1', max_hops=1)
        assert hood['transaction_id'] == 'T1'
        assert len(hood['nodes']) > 0
        assert len(hood['edges']) > 0

    def test_get_neighborhood_max_hops(self, querier):
        hood_1 = querier.get_neighborhood('T1', max_hops=1)
        hood_2 = querier.get_neighborhood('T1', max_hops=2)
        assert len(hood_2['nodes']) >= len(hood_1['nodes'])

    def test_get_suspicious_transactions(self, querier):
        suspicious = querier.get_suspicious_transactions(threshold=0.5)
        assert isinstance(suspicious, list)


class TestClusterDetector:
    """Tests for fraud cluster detection."""

    @pytest.fixture
    def detector(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3'},
            {'transaction_id': 'T4', 'merchant_id': 'M4', 'customer_id': 'C4', 'card1': 100},
            {'transaction_id': 'T5', 'merchant_id': 'M5', 'customer_id': 'C5', 'card1': 100},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
            'T4': {'fraud_probability': 0.6, 'risk_score': 60, 'risk_level': 'MEDIUM'},
        }
        builder.build(txns, risks)
        return ClusterDetector(builder.graph)

    def test_detect_clusters(self, detector):
        clusters = detector.detect_clusters()
        assert len(clusters) >= 2

    def test_cluster_has_transactions(self, detector):
        clusters = detector.detect_clusters()
        for c in clusters:
            assert c['total_transactions'] > 0

    def test_cluster_suspicious_ratio(self, detector):
        clusters = detector.detect_clusters()
        for c in clusters:
            assert 0 <= c['suspicious_ratio'] <= 1

    def test_cluster_risk_level(self, detector):
        clusters = detector.detect_clusters()
        for c in clusters:
            assert c['risk_level'] in ('LOW', 'MEDIUM', 'HIGH', 'UNKNOWN')

    def test_cluster_for_transaction(self, detector):
        cluster = detector.get_cluster_for_transaction('T1')
        assert cluster is not None
        assert 'T1' in cluster['transaction_ids']

    def test_cluster_for_nonexistent(self, detector):
        cluster = detector.get_cluster_for_transaction('T999')
        assert cluster is None

    def test_isolated_transaction_cluster(self, detector):
        cluster = detector.get_cluster_for_transaction('T3')
        assert cluster is not None
        assert cluster['total_transactions'] == 1

    def test_cluster_entity_types(self, detector):
        clusters = detector.detect_clusters()
        for c in clusters:
            assert isinstance(c['entity_types'], list)

    def test_cluster_shared_identifiers(self, detector):
        clusters = detector.detect_clusters()
        for c in clusters:
            assert isinstance(c['shared_identifiers'], dict)

    def test_suspicious_cluster_detection(self, detector):
        clusters = detector.detect_clusters()
        high_risk_clusters = [c for c in clusters if c['risk_level'] == 'HIGH']
        assert len(high_risk_clusters) > 0


class TestNetworkRisk:
    """Tests for network-level risk calculation."""

    @pytest.fixture
    def calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV1'},
            {'transaction_id': 'T4', 'merchant_id': 'M4', 'customer_id': 'C4'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        return NetworkRiskCalculator(builder.graph)

    def test_network_risk_no_connections(self, calc):
        result = calc.compute_network_risk('T4', 10, 'LOW')
        assert result['neighbor_count'] == 0
        assert result['suspicious_neighbor_count'] == 0
        assert result['graph_available'] is True

    def test_network_risk_suspicious_neighbors(self, calc):
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] > 0

    def test_network_risk_boost(self, calc):
        no_conn = calc.compute_network_risk('T4', 50, 'MEDIUM')
        with_conn = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert with_conn['network_risk_score'] >= no_conn['network_risk_score']

    def test_network_risk_factors(self, calc):
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert isinstance(result['factors'], list)
        assert len(result['factors']) > 0

    def test_combined_risk(self, calc):
        result = calc.compute_combined_risk('T1', 50, 'MEDIUM')
        assert 'ml_risk_score' in result
        assert 'network_risk_score' in result
        assert 'combined_risk_score' in result
        assert 'combined_risk_level' in result

    def test_combined_risk_level(self, calc):
        result = calc.compute_combined_risk('T1', 50, 'MEDIUM')
        assert result['combined_risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_combined_risk_deterministic(self, calc):
        r1 = calc.compute_combined_risk('T1', 50, 'MEDIUM')
        r2 = calc.compute_combined_risk('T1', 50, 'MEDIUM')
        assert r1['combined_risk_score'] == r2['combined_risk_score']

    def test_network_risk_level_range(self, calc):
        for score in [0, 25, 50, 75, 100]:
            result = calc.compute_network_risk('T1', score, 'LOW')
            assert result['network_risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_nonexistent_transaction(self, calc):
        result = calc.compute_network_risk('T999', 50, 'MEDIUM')
        assert result['graph_available'] is False

    def test_combined_weight(self, calc):
        result = calc.compute_combined_risk('T1', 100, 'HIGH')
        assert result['combined_risk_score'] <= 100


class TestGraphService:
    """Tests for the graph service orchestrator."""

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

    def test_build(self, service):
        service.build(self._txns())
        assert service.is_ready
        assert service.transaction_count == 3

    def test_not_ready_initially(self, service):
        assert not service.is_ready

    def test_get_connected_transactions(self, service):
        service.build(self._txns())
        result = service.get_connected_transactions('T1')
        assert result['transaction_id'] == 'T1'
        assert result['total_connections'] > 0

    def test_get_transaction_entities(self, service):
        service.build(self._txns())
        entities = service.get_transaction_entities('T1')
        assert len(entities) > 0

    def test_get_clusters(self, service):
        service.build(self._txns())
        result = service.get_clusters()
        assert result['total_clusters'] > 0
        assert result['total_transactions_in_clusters'] == 3

    def test_get_cluster_for_transaction(self, service):
        service.build(self._txns())
        cluster = service.get_cluster_for_transaction('T1')
        assert cluster is not None

    def test_get_network_risk(self, service):
        service.build(self._txns())
        result = service.get_network_risk('T1', 50, 'MEDIUM')
        assert 'combined_risk_score' in result

    def test_add_transaction_risk(self, service):
        service.build(self._txns())
        service.add_transaction_risk('T1', {
            'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'
        })
        result = service.get_network_risk('T1', 90, 'HIGH')
        assert result['suspicious_neighbor_count'] >= 0

    def test_clear(self, service):
        service.build(self._txns())
        service.clear()
        assert not service.is_ready

    def test_not_ready_raises(self, service):
        with pytest.raises(RuntimeError):
            service.get_connected_transactions('T1')

    def test_get_suspicious_transactions(self, service):
        service.build(self._txns(), {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        })
        result = service.get_suspicious_transactions(threshold=0.5)
        assert len(result) >= 1

    def test_get_neighborhood(self, service):
        service.build(self._txns())
        hood = service.get_neighborhood('T1')
        assert hood['transaction_id'] == 'T1'

    def test_last_built(self, service):
        assert service.last_built is None
        service.build(self._txns())
        assert service.last_built is not None


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

    def test_graph_build(self, client):
        response = client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'built'
        assert data['transaction_count'] == 3

    def test_graph_build_and_status(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/status")
        data = response.json()
        assert data['status'] == 'ready'
        assert data['transaction_count'] == 3

    def test_graph_transaction_info(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert data['entity_count'] > 0

    def test_graph_transaction_connections(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/connections")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'
        assert data['total_connections'] > 0

    def test_graph_clusters(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters")
        assert response.status_code == 200
        data = response.json()
        assert data['total_clusters'] > 0

    def test_graph_cluster_for_transaction(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters/T1")
        assert response.status_code == 200

    def test_graph_cluster_not_found(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/clusters/T999")
        assert response.status_code == 404

    def test_graph_transaction_not_built(self, client):
        response = client.get("/api/v1/graph/transaction/T1")
        assert response.status_code == 503

    def test_graph_connections_not_built(self, client):
        response = client.get("/api/v1/graph/transaction/T1/connections")
        assert response.status_code == 503

    def test_graph_neighborhood(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'

    def test_graph_risk(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/risk")
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_id'] == 'T1'

    def test_graph_risk_not_built(self, client):
        response = client.get("/api/v1/graph/transaction/T1/risk")
        assert response.status_code == 503

    def test_graph_neighborhood_not_built(self, client):
        response = client.get("/api/v1/graph/transaction/T1/neighborhood")
        assert response.status_code == 503

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

    def test_graph_build_empty_transactions(self, client):
        response = client.post("/api/v1/graph/build", json={"transactions": []})
        assert response.status_code == 200
        data = response.json()
        assert data['transaction_count'] == 0


class TestIntegration:
    """Integration test: full pipeline from transactions to network risk."""

    def test_full_pipeline(self):
        from app.graph.graph_service import GraphService
        service = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100, 'P_emaildomain': 'gmail.com', 'addr1': 315},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1', 'card1': 200, 'P_emaildomain': 'gmail.com'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV2', 'card1': 100},
            {'transaction_id': 'T4', 'merchant_id': 'M4', 'customer_id': 'C4'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.85, 'risk_score': 85, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.75, 'risk_score': 75, 'risk_level': 'HIGH'},
            'T3': {'fraud_probability': 0.3, 'risk_score': 30, 'risk_level': 'LOW'},
        }
        service.build(txns, risks)
        assert service.is_ready
        assert service.transaction_count == 4
        assert service.entity_count > 0

        connected = service.get_connected_transactions('T1')
        assert connected['total_connections'] > 0

        clusters = service.get_clusters()
        assert clusters['total_clusters'] >= 2

        risk = service.get_network_risk('T1', 85, 'HIGH')
        assert risk['combined_risk_score'] > 0
        assert risk['neighbor_count'] > 0
        assert risk['suspicious_neighbor_count'] > 0

        hood = service.get_neighborhood('T1')
        assert len(hood['nodes']) > 0

        suspicious = service.get_suspicious_transactions(threshold=0.5)
        assert len(suspicious) >= 2
