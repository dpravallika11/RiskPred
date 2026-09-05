import pandas as pd
import numpy as np
import json
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """
    IEEE-CIS Fraud Detection Preprocessing Pipeline.

    Handles:
    - Data merging (transaction + identity)
    - Feature selection (drops V columns, high-missingness columns, identifiers)
    - Feature engineering (log amounts, distance ratios, missingness signals, email features)
    - Missing value imputation (median for numerical, mode for categorical)
    - Categorical encoding (label encoding)
    """

    HIGH_MISSING_THRESHOLD = 0.50

    CATEGORICAL_COLUMNS = [
        'ProductCD', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType',
        'id_12', 'id_15', 'id_16', 'id_30', 'id_31',
        'id_33', 'id_34', 'id_36', 'id_37', 'id_38',
    ]

    def __init__(self):
        self.feature_columns = []
        self.numerical_columns = []
        self.categorical_columns_used = []
        self.fill_values = {}
        self.label_encoders = {}
        self.drop_columns = []
        self._is_fitted = False

    def _merge_data(self, transaction_df, identity_df=None):
        df = transaction_df.copy()
        if identity_df is not None:
            df = df.merge(identity_df, on='TransactionID', how='left')
        return df

    def _select_columns(self, df):
        cols_to_drop = ['TransactionID', 'TransactionDT']

        v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
        cols_to_drop.extend(v_cols)

        if 'isFraud' in df.columns:
            cols_to_drop.append('isFraud')

        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        remaining = [c for c in df.columns if c not in cols_to_drop]

        missing_pct = df[remaining].isnull().mean()
        high_missing = missing_pct[missing_pct > self.HIGH_MISSING_THRESHOLD].index.tolist()
        cols_to_drop.extend(high_missing)

        return [c for c in df.columns if c not in cols_to_drop]

    def _engineer_features(self, df):
        if 'TransactionAmt' in df.columns:
            df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])

        if 'dist1' in df.columns and 'dist2' in df.columns:
            df['dist_ratio'] = df['dist1'] / df['dist2'].replace(0, np.nan)
            df['dist_ratio'] = df['dist_ratio'].fillna(0)
            df['dist1_missing'] = df['dist1'].isnull().astype(int)
            df['dist2_missing'] = df['dist2'].isnull().astype(int)
        elif 'dist1' in df.columns:
            df['dist_ratio'] = 0
            df['dist1_missing'] = df['dist1'].isnull().astype(int)
            df['dist2_missing'] = 1
        else:
            df['dist_ratio'] = 0
            df['dist1_missing'] = 1
            df['dist2_missing'] = 1

        d_cols = [c for c in df.columns if c.startswith('D') and c[1:].isdigit() and c != 'DeviceType']
        if d_cols:
            df['d_missing_count'] = df[d_cols].isnull().sum(axis=1)
        else:
            df['d_missing_count'] = 0

        if 'id_01' in df.columns:
            df['id_01_missing'] = df['id_01'].isnull().astype(int)
        else:
            df['id_01_missing'] = 1

        if 'P_emaildomain' in df.columns:
            df['email_known'] = df['P_emaildomain'].notnull().astype(int)
        else:
            df['email_known'] = 0

        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['same_email_domain'] = (
                (df['P_emaildomain'] == df['R_emaildomain']) &
                df['P_emaildomain'].notnull()
            ).astype(int)
        else:
            df['same_email_domain'] = 0

        c_cols = [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
        if c_cols:
            c_data = df[c_cols].fillna(0)
            df['risk_signal_count'] = ((c_data > c_data.quantile(0.95)).sum(axis=1))
        else:
            df['risk_signal_count'] = 0

        if 'addr1' in df.columns and 'card1' in df.columns:
            df['card_addr_ratio'] = df.groupby('card1')['addr1'].transform('nunique')
        else:
            df['card_addr_ratio'] = 1

        return df

    def _encode_categoricals(self, df, fit=False):
        for col in self.categorical_columns_used:
            if col not in df.columns:
                df[col] = '__MISSING__'

            if fit:
                le = LabelEncoder()
                df[col] = df[col].fillna('__MISSING__').astype(str)
                combined = np.append(df[col].values, '__MISSING__')
                le.fit(combined)
                self.label_encoders[col] = le

            le = self.label_encoders.get(col)
            if le is not None:
                df[col] = df[col].fillna('__MISSING__').astype(str)
                known_classes = set(le.classes_)
                if '__MISSING__' not in known_classes:
                    known_classes.add('__MISSING__')
                    le.classes_ = np.append(le.classes_, '__MISSING__')
                df[col] = df[col].apply(lambda x: x if x in known_classes else '__MISSING__')
                df[col] = le.transform(df[col])

        return df

    def fit(self, transaction_df, identity_df=None, y=None):
        df = self._merge_data(transaction_df, identity_df)

        self.selected_columns = self._select_columns(df)
        df = df[self.selected_columns].copy()

        df = self._engineer_features(df)

        self.categorical_columns_used = [
            c for c in self.CATEGORICAL_COLUMNS if c in df.columns
        ]

        df = self._encode_categoricals(df, fit=True)

        self.numerical_columns = [
            c for c in df.columns if c not in self.categorical_columns_used
        ]

        self.fill_values = {}
        for col in self.numerical_columns:
            if col in df.columns:
                self.fill_values[col] = float(df[col].median()) if df[col].notnull().any() else 0.0

        for col in self.categorical_columns_used:
            if col in df.columns:
                mode_vals = df[col].mode()
                self.fill_values[col] = int(mode_vals.iloc[0]) if len(mode_vals) > 0 else 0

        self.feature_columns = self.numerical_columns + self.categorical_columns_used
        self._is_fitted = True

        return self

    def transform(self, transaction_df, identity_df=None, y=None):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df = self._merge_data(transaction_df, identity_df)

        available_cols = [c for c in self.selected_columns if c in df.columns]
        missing_cols = [c for c in self.selected_columns if c not in df.columns]

        for col in missing_cols:
            df[col] = np.nan

        df = df[self.selected_columns].copy()

        df = self._engineer_features(df)

        df = self._encode_categoricals(df, fit=False)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        for col, fill in self.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill)

        df = df[self.feature_columns].copy()

        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def get_feature_names(self):
        return list(self.feature_columns)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
