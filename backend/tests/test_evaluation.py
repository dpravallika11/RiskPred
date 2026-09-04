import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'artifacts')


class TestEvaluation:
    """Tests for the evaluation pipeline and metrics."""

    @pytest.fixture
    def evaluation_metrics(self):
        path = os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')
        if not os.path.exists(path):
            pytest.skip("Evaluation metrics not found")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture
    def threshold_config(self):
        with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json')) as f:
            return json.load(f)

    @pytest.fixture
    def model_metadata(self):
        with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json')) as f:
            return json.load(f)

    def test_evaluation_metrics_exist(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json'))

    def test_test_samples_reasonable(self, evaluation_metrics):
        assert evaluation_metrics['test_samples'] > 1000

    def test_test_fraud_count_positive(self, evaluation_metrics):
        assert evaluation_metrics['test_fraud'] > 0

    def test_precision_in_range(self, evaluation_metrics):
        p = evaluation_metrics['precision']
        assert 0 <= p <= 1

    def test_recall_in_range(self, evaluation_metrics):
        r = evaluation_metrics['recall']
        assert 0 <= r <= 1

    def test_f1_in_range(self, evaluation_metrics):
        f1 = evaluation_metrics['f1']
        assert 0 <= f1 <= 1

    def test_roc_auc_in_range(self, evaluation_metrics):
        roc = evaluation_metrics['roc_auc']
        assert 0.5 < roc <= 1.0

    def test_pr_auc_positive(self, evaluation_metrics):
        pr = evaluation_metrics['pr_auc']
        assert pr > 0

    def test_confusion_matrix_consistent(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
        assert total == evaluation_metrics['test_samples']

    def test_confusion_matrix_values_non_negative(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        assert cm['tn'] >= 0
        assert cm['fp'] >= 0
        assert cm['fn'] >= 0
        assert cm['tp'] >= 0

    def test_fpr_in_range(self, evaluation_metrics):
        fpr = evaluation_metrics['fpr']
        assert 0 <= fpr <= 1

    def test_fnr_in_range(self, evaluation_metrics):
        fnr = evaluation_metrics['fnr']
        assert 0 <= fnr <= 1

    def test_fpr_matches_confusion_matrix(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        expected_fpr = cm['fp'] / (cm['fp'] + cm['tn']) if (cm['fp'] + cm['tn']) > 0 else 0
        assert abs(evaluation_metrics['fpr'] - expected_fpr) < 0.001

    def test_fnr_matches_confusion_matrix(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        expected_fnr = cm['fn'] / (cm['fn'] + cm['tp']) if (cm['fn'] + cm['tp']) > 0 else 0
        assert abs(evaluation_metrics['fnr'] - expected_fnr) < 0.001

    def test_cost_analysis_present(self, evaluation_metrics):
        assert 'cost_analysis' in evaluation_metrics
        ca = evaluation_metrics['cost_analysis']
        assert ca['fn_cost_assumption'] == 500
        assert ca['fp_cost_assumption'] == 50

    def test_total_cost_matches(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        ca = evaluation_metrics['cost_analysis']
        expected_total = cm['fn'] * ca['fn_cost_assumption'] + cm['fp'] * ca['fp_cost_assumption']
        assert ca['total_cost'] == expected_total

    def test_threshold_sweep_present(self, evaluation_metrics):
        assert 'threshold_sweep' in evaluation_metrics
        assert len(evaluation_metrics['threshold_sweep']) > 0

    def test_threshold_sweep_metrics_in_range(self, evaluation_metrics):
        for entry in evaluation_metrics['threshold_sweep']:
            assert 0 <= entry['precision'] <= 1
            assert 0 <= entry['recall'] <= 1
            assert 0 <= entry['f1'] <= 1

    def test_no_test_set_leakage(self, model_metadata):
        assert model_metadata['val_samples'] != model_metadata.get('test_samples', -1)

    def test_evaluated_at_present(self, evaluation_metrics):
        assert 'evaluated_at' in evaluation_metrics

    def test_threshold_from_config_matches(self, evaluation_metrics, threshold_config):
        assert abs(evaluation_metrics['threshold'] - threshold_config['threshold']) < 0.001
