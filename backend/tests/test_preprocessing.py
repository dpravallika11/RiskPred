import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.preprocessing import FraudPreprocessor


class TestFraudPreprocessor:
    """Tests for FraudPreprocessor pipeline."""

    @pytest.fixture
    def sample_transaction_df(self):
        return pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'TransactionDT': [0, 3600, 7200, 10800, 14400],
            'isFraud': [0, 1, 0, 0, 1],
            'TransactionAmt': [100.0, 500.0, 75.0, 2000.0, 50.0],
            'ProductCD': ['W', 'C', 'W', 'R', 'C'],
            'card1': [1000, 2000, 1000, 3000, 2000],
            'card2': [100, 200, 100, 300, 200],
            'card4': ['visa', 'mastercard', 'visa', 'amex', 'mastercard'],
            'card6': ['credit', 'debit', 'credit', 'credit', 'debit'],
            'addr1': [100, 200, 100, 300, 200],
            'C1': [1, 5, 1, 10, 5],
            'C2': [1, 3, 1, 8, 3],
            'D1': [10.0, 20.0, 10.0, 30.0, 20.0],
            'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', None, 'yahoo.com'],
            'R_emaildomain': ['gmail.com', 'yahoo.com', None, None, 'yahoo.com'],
            'dist1': [10.0, None, 5.0, 15.0, None],
            'dist2': [5.0, 2.0, None, 7.0, 3.0],
        })

    @pytest.fixture
    def fitted_preprocessor(self, sample_transaction_df):
        pp = FraudPreprocessor()
        pp.fit(sample_transaction_df)
        return pp

    def test_fit_sets_is_fitted(self, fitted_preprocessor):
        assert fitted_preprocessor._is_fitted is True

    def test_fit_populates_feature_columns(self, fitted_preprocessor):
        assert len(fitted_preprocessor.feature_columns) > 0

    def test_fit_populates_numerical_columns(self, fitted_preprocessor):
        assert len(fitted_preprocessor.numerical_columns) > 0

    def test_transform_produces_output(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_transaction_df)

    def test_transform_output_columns_match_features(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert list(result.columns) == fitted_preprocessor.feature_columns

    def test_transform_no_nan_values(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert not result.isnull().any().any()

    def test_transform_raises_if_not_fitted(self, sample_transaction_df):
        pp = FraudPreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(sample_transaction_df)

    def test_get_feature_names(self, fitted_preprocessor):
        names = fitted_preprocessor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == len(fitted_preprocessor.feature_columns)

    def test_log_amount_engineered(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert 'log_TransactionAmt' in result.columns

    def test_distance_ratio_engineered(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert 'dist_ratio' in result.columns

    def test_missingness_features(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert 'dist1_missing' in result.columns
        assert 'dist2_missing' in result.columns
        assert 'd_missing_count' in result.columns
        assert 'email_known' in result.columns

    def test_same_email_domain(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        assert 'same_email_domain' in result.columns

    def test_categorical_encoding_produces_integers(self, fitted_preprocessor, sample_transaction_df):
        result = fitted_preprocessor.transform(sample_transaction_df)
        for col in fitted_preprocessor.categorical_columns_used:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should be numeric"

    def test_save_and_load(self, fitted_preprocessor, tmp_path):
        path = tmp_path / "preprocessor.joblib"
        fitted_preprocessor.save(str(path))
        loaded = FraudPreprocessor.load(str(path))
        assert loaded._is_fitted is True
        assert loaded.feature_columns == fitted_preprocessor.feature_columns

    def test_multiple_transforms_consistent(self, fitted_preprocessor, sample_transaction_df):
        r1 = fitted_preprocessor.transform(sample_transaction_df)
        r2 = fitted_preprocessor.transform(sample_transaction_df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_transform_with_identity(self, fitted_preprocessor, sample_transaction_df):
        identity_df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'DeviceType': ['desktop', 'mobile', 'desktop', 'mobile', 'desktop'],
        })
        result = fitted_preprocessor.transform(sample_transaction_df, identity_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_transaction_df)
