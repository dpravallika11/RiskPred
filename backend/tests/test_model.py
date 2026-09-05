import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'artifacts')


class TestModelArtifacts:
    """Tests for trained model artifacts."""

    @pytest.fixture
    def model(self):
        return joblib.load(os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))

    @pytest.fixture
    def preprocessor(self):
        return joblib.load(os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    @pytest.fixture
    def threshold_config(self):
        with open(os.path.join(ARTIFACTS_DIR, 'threshold_config.json')) as f:
            return json.load(f)

    @pytest.fixture
    def feature_columns(self):
        with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json')) as f:
            return json.load(f)

    @pytest.fixture
    def model_metadata(self):
        with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json')) as f:
            return json.load(f)

    @pytest.fixture
    def evaluation_metrics(self):
        with open(os.path.join(ARTIFACTS_DIR, 'evaluation_metrics.json')) as f:
            return json.load(f)

    def test_model_file_exists(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))

    def test_preprocessor_file_exists(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

    def test_threshold_config_exists(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'threshold_config.json'))

    def test_feature_columns_file_exists(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'))

    def test_model_metadata_exists(self):
        assert os.path.exists(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'))

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, 'predict_proba')

    def test_model_has_predict(self, model):
        assert hasattr(model, 'predict')

    def test_model_is_xgboost(self, model, model_metadata):
        assert model_metadata['model'] == 'XGBClassifier'

    def test_feature_columns_count_matches_metadata(self, feature_columns, model_metadata):
        assert len(feature_columns) == model_metadata['feature_count']

    def test_feature_columns_count_matches_preprocessor(self, feature_columns, preprocessor):
        assert len(feature_columns) == len(preprocessor.get_feature_names())

    def test_threshold_is_reasonable(self, threshold_config):
        threshold = threshold_config['threshold']
        assert 0.0 <= threshold <= 1.0

    def test_threshold_method_is_cost_sensitive(self, threshold_config):
        assert threshold_config['method'] == 'cost_sensitive'

    def test_validation_metrics_present(self, model_metadata):
        vm = model_metadata['validation_metrics']
        assert 'roc_auc' in vm
        assert 'pr_auc' in vm
        assert 'f1' in vm
        assert 'precision' in vm
        assert 'recall' in vm

    def test_validation_roc_auc_reasonable(self, model_metadata):
        roc = model_metadata['validation_metrics']['roc_auc']
        assert 0.5 < roc <= 1.0, f"ROC-AUC {roc} is outside reasonable range"

    def test_validation_pr_auc_positive(self, model_metadata):
        pr = model_metadata['validation_metrics']['pr_auc']
        assert pr > 0, f"PR-AUC {pr} should be positive"

    def test_class_distribution_present(self, model_metadata):
        cd = model_metadata['class_distribution']
        assert cd['train_fraud'] > 0
        assert cd['val_fraud'] > 0
        assert cd['train_legit'] > cd['train_fraud']

    def test_model_comparison_has_entries(self, model_metadata):
        assert len(model_metadata['model_comparison']) >= 3

    def test_xgboost_best_in_comparison(self, model_metadata):
        comparison = model_metadata['model_comparison']
        xgb_entries = [m for m in comparison if 'XGBoost' in m['name']]
        assert len(xgb_entries) > 0
        best_xgb = max(xgb_entries, key=lambda m: m['pr_auc'])
        for m in comparison:
            if 'XGBoost' not in m['name']:
                assert best_xgb['pr_auc'] >= m['pr_auc'], "XGBoost should have best PR-AUC"

    def test_evaluation_metrics_present(self, evaluation_metrics):
        assert 'precision' in evaluation_metrics
        assert 'recall' in evaluation_metrics
        assert 'f1' in evaluation_metrics
        assert 'roc_auc' in evaluation_metrics
        assert 'pr_auc' in evaluation_metrics

    def test_evaluation_confusion_matrix(self, evaluation_metrics):
        cm = evaluation_metrics['confusion_matrix']
        assert cm['tp'] + cm['fp'] + cm['fn'] + cm['tn'] == evaluation_metrics['test_samples']

    def test_model_predicts_without_error(self, model, preprocessor, feature_columns):
        X_dummy = pd.DataFrame(
            np.random.rand(5, len(feature_columns)),
            columns=feature_columns
        )
        proba = model.predict_proba(X_dummy)[:, 1]
        assert len(proba) == 5
        assert all(0 <= p <= 1 for p in proba)

    def test_cost_assumptions_documented(self, threshold_config):
        assert 'fn_cost_assumption' in threshold_config
        assert 'fp_cost_assumption' in threshold_config
        assert threshold_config['fn_cost_assumption'] > threshold_config['fp_cost_assumption']
