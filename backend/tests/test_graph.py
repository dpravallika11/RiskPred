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
        assert 'email_domain' in entities
        assert 'address' in entities
        assert 'merchant' in entities
        assert 'customer' in entities
        assert '100' in entities['card']
        assert 'visa' in entities['card']
        assert 'DEV1' in entities['device']
        assert 'gmail.com' in entities['email_domain']
        assert '315' in entities['address']

    def test_extract_missing_identifier(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
        }
        entities = extractor.extract(txn)
        assert 'card' not in entities
        assert 'email_domain' not in entities
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
        assert 'email_domain' in entities
        assert 'gmail.com' in entities['email_domain']
        assert 'yahoo.com' in entities['email_domain']

    def test_extract_r_emaildomain_only(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'R_emaildomain': 'outlook.com',
        }
        entities = extractor.extract(txn)
        assert 'email_domain' in entities
        assert 'outlook.com' in entities['email_domain']

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
        assert 'email_domain' not in entities
        assert 'device' not in entities

    def test_invalid_identifier_nan_string(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': 'nan',
            'card1': 'NaN',
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities
        assert 'card' not in entities

    def test_invalid_identifier_none_string(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': 'None',
            'card1': 'null',
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities
        assert 'card' not in entities

    def test_invalid_identifier_sentinel_values(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'device_id': '-1',
            'card1': '-1.0',
        }
        entities = extractor.extract(txn)
        assert 'device' not in entities
        assert 'card' not in entities

    def test_email_domain_not_email(self, extractor):
        """Email fields represent domains, not individual addresses."""
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
            'P_emaildomain': 'gmail.com',
        }
        entities = extractor.extract(txn)
        assert 'email' not in entities
        assert 'email_domain' in entities

    def test_missing_email_domain_no_node(self, extractor):
        txn = {
            'transaction_id': 'T1',
            'merchant_id': 'M1',
            'customer_id': 'C1',
        }
        entities = extractor.extract(txn)
        assert 'email_domain' not in entities
        assert 'email' not in entities


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

    def test_email_domain_node_key(self, builder):
        builder.build([self._make_txn('T1', P_emaildomain='gmail.com')])
        assert 'email_domain:gmail.com' in builder.graph


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

    def test_network_score_independent_of_ml(self, calc):
        """Network score must NOT use ML risk as its base."""
        r1 = calc.compute_network_risk('T1', 10, 'LOW')
        r2 = calc.compute_network_risk('T1', 90, 'HIGH')
        # Network score is purely graph-derived, ML score should not affect it
        assert r1['network_risk_score'] == r2['network_risk_score']

    def test_combined_uses_correct_weights(self, calc):
        """Combined = 0.70 * ML + 0.30 * network."""
        ml_score = 80.0
        result = calc.compute_combined_risk('T1', ml_score, 'HIGH')
        network_score = result['network_risk_score']
        expected = 0.70 * ml_score + 0.30 * network_score
        assert abs(result['combined_risk_score'] - round(expected, 2)) < 0.01

    def test_ml_change_does_not_affect_graph_score(self, calc):
        """Changing ML score should not change graph score."""
        r1 = calc.compute_combined_risk('T1', 50, 'MEDIUM')
        r2 = calc.compute_combined_risk('T1', 90, 'HIGH')
        assert r1['network_risk_score'] == r2['network_risk_score']
        assert r1['combined_risk_score'] != r2['combined_risk_score']

    def test_graph_score_bounded_0_100(self, calc):
        for ml in [0, 50, 100]:
            result = calc.compute_combined_risk('T1', ml, 'LOW')
            assert 0 <= result['network_risk_score'] <= 100

    def test_combined_score_bounded_0_100(self, calc):
        for ml in [0, 50, 100]:
            result = calc.compute_combined_risk('T1', ml, 'LOW')
            assert 0 <= result['combined_risk_score'] <= 100

    def test_no_connections_produces_baseline(self, calc):
        result = calc.compute_network_risk('T4', 0, 'LOW')
        assert result['network_risk_score'] == 0
        assert result['neighbor_count'] == 0

    def test_suspicious_neighbors_increase_graph_risk(self, calc):
        """T1 has 1 suspicious neighbor (T2), T4 has 0 neighbors."""
        r1 = calc.compute_network_risk('T1', 50, 'MEDIUM')
        r4 = calc.compute_network_risk('T4', 50, 'MEDIUM')
        assert r1['network_risk_score'] > r4['network_risk_score']
        assert r1['suspicious_neighbor_count'] > r4['suspicious_neighbor_count']

    def test_weak_entities_do_not_create_high_risk(self, calc):
        """A transaction connected only via weak entities (merchant, customer)
        should not automatically get HIGH risk from graph alone."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'A1', 'merchant_id': 'COMMON_MERCHANT', 'customer_id': 'C1'},
            {'transaction_id': 'A2', 'merchant_id': 'COMMON_MERCHANT', 'customer_id': 'C2'},
            {'transaction_id': 'A3', 'merchant_id': 'COMMON_MERCHANT', 'customer_id': 'C3'},
        ]
        builder.build(txns)
        calc_weak = NetworkRiskCalculator(builder.graph)
        result = calc_weak.compute_network_risk('A1', 0, 'LOW')
        # Connected only via merchant (weight=1.0) and customer (weight=0.5)
        # Should NOT be HIGH risk
        assert result['network_risk_level'] != 'HIGH'


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

    def test_get_transaction_risk_public(self, service):
        """get_transaction_risk should work via public method."""
        service.build(self._txns(), {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        })
        risk = service.get_transaction_risk('T1')
        assert risk is not None
        assert risk['risk_score'] == 80

    def test_neighborhood_risk_safe_neighbor(self):
        """A safe neighbor (low fraud probability) should not be marked suspicious."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.2, 'risk_score': 20, 'risk_level': 'LOW'},
        }
        svc.build(txns, risks)
        result = svc.get_neighborhood_risk('T1')
        assert result['neighbor_count'] == 1
        assert result['suspicious_neighbor_count'] == 0
        assert len(result['suspicious_neighbors']) == 0

    def test_neighborhood_risk_suspicious_neighbor(self):
        """A suspicious neighbor should be correctly identified."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        svc.build(txns, risks)
        result = svc.get_neighborhood_risk('T1')
        assert result['neighbor_count'] == 1
        assert result['suspicious_neighbor_count'] == 1
        assert len(result['suspicious_neighbors']) == 1
        assert result['suspicious_neighbors'][0]['transaction_id'] == 'T2'

    def test_neighborhood_risk_multiple_neighbors(self):
        """Multiple neighbors counted correctly."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'card1': 100},
        ]
        risks = {
            'T1': {'fraud_probability': 0.1, 'risk_score': 10, 'risk_level': 'LOW'},
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
            'T3': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
        }
        svc.build(txns, risks)
        result = svc.get_neighborhood_risk('T1')
        assert result['neighbor_count'] == 2
        assert result['suspicious_neighbor_count'] == 2

    def test_neighborhood_risk_shared_entity_types(self):
        """shared_entity_types correctly populated."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'card1': 100},
        ]
        svc.build(txns)
        result = svc.get_neighborhood_risk('T1')
        assert 'device' in result['shared_entity_types']
        assert 'card' in result['shared_entity_types']

    def test_neighborhood_risk_empty(self):
        """Isolated transaction returns correct empty values."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'},
        ]
        svc.build(txns)
        result = svc.get_neighborhood_risk('T1')
        assert result['neighbor_count'] == 0
        assert result['suspicious_neighbor_count'] == 0
        assert result['suspicious_neighbors'] == []
        assert result['network_context_added'] is False


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

    def test_graph_invalid_max_hops(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=-1")
        assert response.status_code == 422

    def test_graph_max_hops_too_large(self, client):
        client.post("/api/v1/graph/build", json={"transactions": self._txns()})
        response = client.get("/api/v1/graph/transaction/T1/neighborhood?max_hops=10")
        assert response.status_code == 422


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

    def test_data_contract_fixture(self):
        """Test the specific fixture from the Sprint 3.1 spec."""
        from app.graph.graph_service import GraphService
        svc = GraphService()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1', 'card1': 'C1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1', 'card1': 'C2'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'D2', 'card1': 'C1'},
            {'transaction_id': 'T4', 'merchant_id': 'M1', 'customer_id': 'C4', 'P_emaildomain': 'gmail.com'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
            'T3': {'fraud_probability': 0.2, 'risk_score': 20, 'risk_level': 'LOW'},
            'T4': {'fraud_probability': 0.1, 'risk_score': 10, 'risk_level': 'LOW'},
        }
        svc.build(txns, risks)

        # T1 and T2 share device D1
        conn_t1 = svc.get_connected_transactions('T1')
        conn_ids = [c['transaction_id'] for c in conn_t1['connected_transactions']]
        assert 'T2' in conn_ids

        # T1 and T3 share card C1
        assert 'T3' in conn_ids

        # T4 only shares merchant M1 with T1
        conn_t4 = svc.get_connected_transactions('T4')
        conn_ids_t4 = [c['transaction_id'] for c in conn_t4['connected_transactions']]
        assert 'T1' in conn_ids_t4

        # Neighborhood risk: T1 has suspicious neighbor T2
        risk_t1 = svc.get_neighborhood_risk('T1')
        assert risk_t1['neighbor_count'] >= 1
        suspicious_ids = [n['transaction_id'] for n in risk_t1['suspicious_neighbors']]
        assert 'T2' in suspicious_ids
        assert 'T3' not in suspicious_ids


class TestNetworkRiskIndependence:
    """A. Network-risk score must be identical regardless of ML risk input."""

    @pytest.fixture
    def calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        return NetworkRiskCalculator(builder.graph)

    def test_network_risk_identical_across_ml_scores(self, calc):
        """Build the same graph; call compute_network_risk with ML=10,50,90.
        Assert network_risk_score is identical for all three."""
        r_low = calc.compute_network_risk('T1', 10, 'LOW')
        r_med = calc.compute_network_risk('T1', 50, 'MEDIUM')
        r_high = calc.compute_network_risk('T1', 90, 'HIGH')
        assert r_low['network_risk_score'] == r_med['network_risk_score']
        assert r_med['network_risk_score'] == r_high['network_risk_score']

    def test_network_risk_level_identical_across_ml(self, calc):
        r_low = calc.compute_network_risk('T1', 10, 'LOW')
        r_high = calc.compute_network_risk('T1', 90, 'HIGH')
        assert r_low['network_risk_level'] == r_high['network_risk_level']

    def test_network_risk_factors_identical(self, calc):
        r_low = calc.compute_network_risk('T1', 10, 'LOW')
        r_high = calc.compute_network_risk('T1', 90, 'HIGH')
        assert r_low['factors'] == r_high['factors']

    def test_neighbor_counts_identical(self, calc):
        r_low = calc.compute_network_risk('T1', 10, 'LOW')
        r_high = calc.compute_network_risk('T1', 90, 'HIGH')
        assert r_low['neighbor_count'] == r_high['neighbor_count']
        assert r_low['suspicious_neighbor_count'] == r_high['suspicious_neighbor_count']

    def test_isolated_transaction_independent_of_ml(self, calc):
        """Isolated transaction (no connections) should also be independent."""
        r_low = calc.compute_network_risk('T3', 0, 'LOW')
        r_high = calc.compute_network_risk('T3', 100, 'HIGH')
        assert r_low['network_risk_score'] == r_high['network_risk_score']
        assert r_low['network_risk_score'] == 0


class TestExactCombination:
    """B. Verify exact 70/30 combination formula."""

    @pytest.fixture
    def calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T2': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        return NetworkRiskCalculator(builder.graph)

    def test_combined_formula_ml_50(self, calc):
        """combined = 0.70 * ml + 0.30 * network, tested with ML=50."""
        ml_score = 50.0
        result = calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        network_score = result['network_risk_score']
        expected = round(0.70 * ml_score + 0.30 * network_score, 2)
        assert result['combined_risk_score'] == pytest.approx(expected, abs=0.01)

    def test_combined_formula_ml_80(self, calc):
        """Test with ML=80."""
        ml_score = 80.0
        result = calc.compute_combined_risk('T1', ml_score, 'HIGH')
        network_score = result['network_risk_score']
        expected = round(0.70 * ml_score + 0.30 * network_score, 2)
        assert result['combined_risk_score'] == pytest.approx(expected, abs=0.01)

    def test_combined_formula_ml_0(self, calc):
        """Test with ML=0."""
        ml_score = 0.0
        result = calc.compute_combined_risk('T1', ml_score, 'LOW')
        network_score = result['network_risk_score']
        expected = round(0.70 * ml_score + 0.30 * network_score, 2)
        assert result['combined_risk_score'] == pytest.approx(expected, abs=0.01)

    def test_combined_formula_ml_100(self, calc):
        """Test with ML=100."""
        ml_score = 100.0
        result = calc.compute_combined_risk('T1', ml_score, 'HIGH')
        network_score = result['network_risk_score']
        expected = round(0.70 * ml_score + 0.30 * network_score, 2)
        assert result['combined_risk_score'] == pytest.approx(expected, abs=0.01)

    def test_combined_formula_ml_37(self, calc):
        """Test with odd ML=37."""
        ml_score = 37.0
        result = calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        network_score = result['network_risk_score']
        expected = round(0.70 * ml_score + 0.30 * network_score, 2)
        assert result['combined_risk_score'] == pytest.approx(expected, abs=0.01)

    def test_isolated_uses_ml_only(self, calc):
        """For a nonexistent/absent transaction, combined == ML score."""
        ml_score = 60.0
        result = calc.compute_combined_risk('T999', ml_score, 'MEDIUM')
        assert result['combined_risk_score'] == pytest.approx(ml_score, abs=0.01)


class TestGraphOnlyChange:
    """C. Change graph topology with fixed ML score; assert network changes, ML unchanged."""

    @pytest.fixture
    def base_calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
        ]
        builder.build(txns)
        return NetworkRiskCalculator(builder.graph)

    @pytest.fixture
    def connected_calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        return NetworkRiskCalculator(builder.graph)

    def test_network_score_changes_with_topology(self, base_calc, connected_calc):
        """T1 with no connections vs T1 with a suspicious connected transaction."""
        ml_score = 50.0
        r_alone = base_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        r_connected = connected_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        assert r_connected['network_risk_score'] > r_alone['network_risk_score']

    def test_ml_score_unchanged_by_topology(self, base_calc, connected_calc):
        """ML score must be the same regardless of graph topology."""
        ml_score = 50.0
        r_alone = base_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        r_connected = connected_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        assert r_alone['ml_risk_score'] == r_connected['ml_risk_score']
        assert r_alone['ml_risk_score'] == ml_score

    def test_combined_changes_when_graph_changes(self, base_calc, connected_calc):
        """Combined score should differ when graph topology changes."""
        ml_score = 50.0
        r_alone = base_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        r_connected = connected_calc.compute_combined_risk('T1', ml_score, 'MEDIUM')
        assert r_alone['combined_risk_score'] != r_connected['combined_risk_score']

    def test_adding_strong_entity_increases_risk(self):
        """Adding a shared card (strong entity) should increase network risk."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator

        builder_alone = GraphBuilder()
        builder_alone.build([
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'},
        ])
        calc_alone = NetworkRiskCalculator(builder_alone.graph)

        builder_shared = GraphBuilder()
        builder_shared.build([
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'card1': 100},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'card1': 100},
        ])
        calc_shared = NetworkRiskCalculator(builder_shared.graph)

        r_alone = calc_alone.compute_combined_risk('T1', 50, 'MEDIUM')
        r_shared = calc_shared.compute_combined_risk('T1', 50, 'MEDIUM')
        assert r_shared['network_risk_score'] > r_alone['network_risk_score']


class TestSuspiciousNeighborDetection:
    """D. Only transactions with fraud_probability >= 0.5 are counted as suspicious."""

    @pytest.fixture
    def calc(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV1'},
        ]
        risks = {
            'T1': {'fraud_probability': 0.1, 'risk_score': 10, 'risk_level': 'LOW'},
            'T2': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'T3': {'fraud_probability': 0.3, 'risk_score': 30, 'risk_level': 'LOW'},
        }
        builder.build(txns, risks)
        return NetworkRiskCalculator(builder.graph)

    def test_only_suspicious_neighbor_counted(self, calc):
        """T1 has 2 neighbors: T2 (fp=0.8 suspicious) and T3 (fp=0.3 safe)."""
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] == 1

    def test_total_neighbor_count_includes_all(self, calc):
        """Both neighbors are counted in total neighbor count."""
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['neighbor_count'] == 2

    def test_boundary_threshold_05_is_suspicious(self):
        """Exactly 0.5 IS suspicious (>= threshold)."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator, SUSPICIOUS_THRESHOLD
        assert SUSPICIOUS_THRESHOLD == 0.5

        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.5, 'risk_score': 50, 'risk_level': 'MEDIUM'},
        }
        builder.build(txns, risks)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] == 1

    def test_boundary_below_threshold_not_suspicious(self):
        """fraud_probability = 0.49 is NOT suspicious (< threshold)."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator

        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.49, 'risk_score': 49, 'risk_level': 'MEDIUM'},
        }
        builder.build(txns, risks)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] == 0

    def test_boundary_just_above_threshold(self):
        """fraud_probability = 0.501 is suspicious."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator

        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.501, 'risk_score': 50, 'risk_level': 'MEDIUM'},
        }
        builder.build(txns, risks)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] == 1

    def test_all_suspicious_neighbors(self):
        """All neighbors suspicious."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator

        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'T2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
            {'transaction_id': 'T3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'DEV1'},
        ]
        risks = {
            'T2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
            'T3': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('T1', 50, 'MEDIUM')
        assert result['suspicious_neighbor_count'] == 2
        assert result['neighbor_count'] == 2


class TestWeakEntityBehavior:
    """E. Common weak entities do not automatically produce HIGH network risk."""

    def test_merchant_only_no_high(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'A1', 'merchant_id': 'COMMON_M', 'customer_id': 'C1'},
            {'transaction_id': 'A2', 'merchant_id': 'COMMON_M', 'customer_id': 'C2'},
            {'transaction_id': 'A3', 'merchant_id': 'COMMON_M', 'customer_id': 'C3'},
            {'transaction_id': 'A4', 'merchant_id': 'COMMON_M', 'customer_id': 'C4'},
            {'transaction_id': 'A5', 'merchant_id': 'COMMON_M', 'customer_id': 'C5'},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('A1', 0, 'LOW')
        assert result['network_risk_level'] != 'HIGH'

    def test_email_domain_only_no_high(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'E1', 'merchant_id': 'M1', 'customer_id': 'C1', 'P_emaildomain': 'gmail.com'},
            {'transaction_id': 'E2', 'merchant_id': 'M2', 'customer_id': 'C2', 'P_emaildomain': 'gmail.com'},
            {'transaction_id': 'E3', 'merchant_id': 'M3', 'customer_id': 'C3', 'P_emaildomain': 'gmail.com'},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('E1', 0, 'LOW')
        assert result['network_risk_level'] != 'HIGH'

    def test_customer_only_no_high(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'CU1', 'merchant_id': 'M1', 'customer_id': 'C_SHARED'},
            {'transaction_id': 'CU2', 'merchant_id': 'M2', 'customer_id': 'C_SHARED'},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('CU1', 0, 'LOW')
        assert result['network_risk_level'] != 'HIGH'

    def test_weak_entities_still_contribute_to_score(self):
        """Weak entities do contribute some score, just not HIGH."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'W1', 'merchant_id': 'M1', 'customer_id': 'C1'},
            {'transaction_id': 'W2', 'merchant_id': 'M1', 'customer_id': 'C2'},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('W1', 0, 'LOW')
        assert result['network_risk_score'] > 0


class TestStrongEntityBehavior:
    """F. Shared device/card/address contribute to graph-derived risk."""

    def test_shared_device_increases_risk(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'S1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'S2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result_shared = calc.compute_network_risk('S1', 0, 'LOW')

        builder2 = GraphBuilder()
        txns2 = [
            {'transaction_id': 'S1', 'merchant_id': 'M1', 'customer_id': 'C1'},
            {'transaction_id': 'S2', 'merchant_id': 'M2', 'customer_id': 'C2'},
        ]
        builder2.build(txns2)
        calc2 = NetworkRiskCalculator(builder2.graph)
        result_alone = calc2.compute_network_risk('S1', 0, 'LOW')

        assert result_shared['network_risk_score'] > result_alone['network_risk_score']

    def test_shared_card_increases_risk(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'SC1', 'merchant_id': 'M1', 'customer_id': 'C1', 'card1': 999},
            {'transaction_id': 'SC2', 'merchant_id': 'M2', 'customer_id': 'C2', 'card1': 999},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('SC1', 0, 'LOW')
        assert result['network_risk_score'] > 0

    def test_shared_address_increases_risk(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'SA1', 'merchant_id': 'M1', 'customer_id': 'C1', 'addr1': 315},
            {'transaction_id': 'SA2', 'merchant_id': 'M2', 'customer_id': 'C2', 'addr1': 315},
        ]
        builder.build(txns)
        calc = NetworkRiskCalculator(builder.graph)
        result = calc.compute_network_risk('SA1', 0, 'LOW')
        assert result['network_risk_score'] > 0

    def test_multiple_strong_entities_higher_than_single(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.network_risk import NetworkRiskCalculator

        builder_single = GraphBuilder()
        builder_single.build([
            {'transaction_id': 'MS1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1'},
            {'transaction_id': 'MS2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1'},
        ])
        calc_single = NetworkRiskCalculator(builder_single.graph)

        builder_multi = GraphBuilder()
        builder_multi.build([
            {'transaction_id': 'MS1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'DEV1', 'card1': 500, 'addr1': 100},
            {'transaction_id': 'MS2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'DEV1', 'card1': 500, 'addr1': 100},
        ])
        calc_multi = NetworkRiskCalculator(builder_multi.graph)

        r_single = calc_single.compute_network_risk('MS1', 0, 'LOW')
        r_multi = calc_multi.compute_network_risk('MS1', 0, 'LOW')
        assert r_multi['network_risk_score'] >= r_single['network_risk_score']


class TestInvalidIdentifiersGraphLevel:
    """G. Invalid identifiers never create graph entity nodes."""

    @pytest.mark.parametrize("device_val", [
        None, float('nan'), '', 'UNKNOWN', 'null', 'None', 'nan', '-1', '-1.0',
    ])
    def test_invalid_device_no_entity_node(self, device_val):
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        txn = {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'}
        if device_val is not None and isinstance(device_val, float) and math.isnan(device_val):
            txn['device_id'] = device_val
        else:
            txn['device_id'] = device_val
        builder.build([txn])
        entity_nodes = [n for n, d in builder.graph.nodes(data=True) if d.get('node_type') == 'device']
        assert len(entity_nodes) == 0

    @pytest.mark.parametrize("card_val", [
        None, float('nan'), '', 'UNKNOWN', 'null', 'None', 'nan', '-1', '-1.0',
    ])
    def test_invalid_card_no_entity_node(self, card_val):
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        txn = {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'}
        if card_val is not None and isinstance(card_val, float) and math.isnan(card_val):
            txn['card1'] = card_val
        else:
            txn['card1'] = card_val
        builder.build([txn])
        entity_nodes = [n for n, d in builder.graph.nodes(data=True) if d.get('node_type') == 'card']
        assert len(entity_nodes) == 0

    @pytest.mark.parametrize("addr_val", [
        None, float('nan'), '', 'UNKNOWN', 'null', 'None', 'nan', '-1', '-1.0',
    ])
    def test_invalid_address_no_entity_node(self, addr_val):
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        txn = {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'}
        if addr_val is not None and isinstance(addr_val, float) and math.isnan(addr_val):
            txn['addr1'] = addr_val
        else:
            txn['addr1'] = addr_val
        builder.build([txn])
        entity_nodes = [n for n, d in builder.graph.nodes(data=True) if d.get('node_type') == 'address']
        assert len(entity_nodes) == 0

    @pytest.mark.parametrize("email_val", [
        None, float('nan'), '', 'UNKNOWN', 'null', 'None', 'nan', '-1', '-1.0',
    ])
    def test_invalid_email_domain_no_entity_node(self, email_val):
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        txn = {'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1'}
        if email_val is not None and isinstance(email_val, float) and math.isnan(email_val):
            txn['P_emaildomain'] = email_val
        else:
            txn['P_emaildomain'] = email_val
        builder.build([txn])
        entity_nodes = [n for n, d in builder.graph.nodes(data=True) if d.get('node_type') == 'email_domain']
        assert len(entity_nodes) == 0

    def test_valid_identifiers_do_create_nodes(self):
        """Sanity check: valid identifiers DO create entity nodes."""
        from app.graph.graph_builder import GraphBuilder
        builder = GraphBuilder()
        builder.build([{
            'transaction_id': 'T1', 'merchant_id': 'M1', 'customer_id': 'C1',
            'device_id': 'DEV1', 'card1': 100, 'addr1': 315, 'P_emaildomain': 'gmail.com',
        }])
        assert any(d.get('node_type') == 'device' for _, d in builder.graph.nodes(data=True))
        assert any(d.get('node_type') == 'card' for _, d in builder.graph.nodes(data=True))
        assert any(d.get('node_type') == 'address' for _, d in builder.graph.nodes(data=True))
        assert any(d.get('node_type') == 'email_domain' for _, d in builder.graph.nodes(data=True))


class TestClusterGrouping:
    """Verify cluster detection groups connected transactions correctly."""

    def test_shared_device_same_cluster(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'A1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1'},
            {'transaction_id': 'A2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
        ]
        builder.build(txns)
        detector = ClusterDetector(builder.graph)
        cluster_a1 = detector.get_cluster_for_transaction('A1')
        cluster_a2 = detector.get_cluster_for_transaction('A2')
        assert cluster_a1['transaction_ids'] == cluster_a2['transaction_ids']

    def test_isolated_transaction_own_cluster(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'ISO1', 'merchant_id': 'M1', 'customer_id': 'C1'},
            {'transaction_id': 'ISO2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
            {'transaction_id': 'ISO3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'D1'},
        ]
        builder.build(txns)
        detector = ClusterDetector(builder.graph)
        cluster_iso = detector.get_cluster_for_transaction('ISO1')
        assert cluster_iso is not None
        assert cluster_iso['total_transactions'] == 1
        assert 'ISO1' in cluster_iso['transaction_ids']

    def test_disconnected_groups_separate_clusters(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'G1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1'},
            {'transaction_id': 'G2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
            {'transaction_id': 'G3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'D2'},
            {'transaction_id': 'G4', 'merchant_id': 'M4', 'customer_id': 'C4', 'device_id': 'D2'},
        ]
        builder.build(txns)
        detector = ClusterDetector(builder.graph)
        cluster_g1 = detector.get_cluster_for_transaction('G1')
        cluster_g3 = detector.get_cluster_for_transaction('G3')
        assert set(cluster_g1['transaction_ids']) == {'G1', 'G2'}
        assert set(cluster_g3['transaction_ids']) == {'G3', 'G4'}
        assert cluster_g1['transaction_ids'] != cluster_g3['transaction_ids']

    def test_weak_only_cluster_not_high(self):
        """Cluster connected only via weak entities should not be HIGH risk."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'W1', 'merchant_id': 'SHARED_M', 'customer_id': 'C1'},
            {'transaction_id': 'W2', 'merchant_id': 'SHARED_M', 'customer_id': 'C2'},
            {'transaction_id': 'W3', 'merchant_id': 'SHARED_M', 'customer_id': 'C3'},
        ]
        builder.build(txns)
        detector = ClusterDetector(builder.graph)
        cluster = detector.get_cluster_for_transaction('W1')
        assert cluster['risk_level'] != 'HIGH'

    def test_strong_entity_high_suspicious_cluster(self):
        """Cluster with strong entities + high suspicious ratio can be HIGH."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'ST1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1'},
            {'transaction_id': 'ST2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
            {'transaction_id': 'ST3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'D1'},
        ]
        risks = {
            'ST1': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
            'ST2': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'ST3': {'fraud_probability': 0.7, 'risk_score': 70, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        detector = ClusterDetector(builder.graph)
        cluster = detector.get_cluster_for_transaction('ST1')
        assert cluster['risk_level'] == 'HIGH'

    def test_suspicious_ratio_calculation(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'SR1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1'},
            {'transaction_id': 'SR2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
            {'transaction_id': 'SR3', 'merchant_id': 'M3', 'customer_id': 'C3', 'device_id': 'D1'},
            {'transaction_id': 'SR4', 'merchant_id': 'M4', 'customer_id': 'C4', 'device_id': 'D1'},
        ]
        risks = {
            'SR1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'SR2': {'fraud_probability': 0.9, 'risk_score': 90, 'risk_level': 'HIGH'},
        }
        builder.build(txns, risks)
        detector = ClusterDetector(builder.graph)
        cluster = detector.get_cluster_for_transaction('SR1')
        expected_ratio = 2 / 4
        assert cluster['suspicious_ratio'] == pytest.approx(expected_ratio, abs=0.01)

    def test_average_risk_calculation(self):
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'AR1', 'merchant_id': 'M1', 'customer_id': 'C1', 'device_id': 'D1'},
            {'transaction_id': 'AR2', 'merchant_id': 'M2', 'customer_id': 'C2', 'device_id': 'D1'},
        ]
        risks = {
            'AR1': {'fraud_probability': 0.8, 'risk_score': 80, 'risk_level': 'HIGH'},
            'AR2': {'fraud_probability': 0.2, 'risk_score': 20, 'risk_level': 'LOW'},
        }
        builder.build(txns, risks)
        detector = ClusterDetector(builder.graph)
        cluster = detector.get_cluster_for_transaction('AR1')
        expected_avg = (80 + 20) / 2
        assert cluster['avg_risk_score'] == pytest.approx(expected_avg, abs=0.01)

    def test_weak_entities_no_strong_types_recorded(self):
        """Clusters with only merchant/email_domain should list no strong entity types."""
        from app.graph.graph_builder import GraphBuilder
        from app.graph.cluster_detector import ClusterDetector
        builder = GraphBuilder()
        txns = [
            {'transaction_id': 'WT1', 'merchant_id': 'M1', 'customer_id': 'C1'},
            {'transaction_id': 'WT2', 'merchant_id': 'M1', 'customer_id': 'C2'},
        ]
        builder.build(txns)
        detector = ClusterDetector(builder.graph)
        cluster = detector.get_cluster_for_transaction('WT1')
        assert len(cluster['strong_entity_types']) == 0
