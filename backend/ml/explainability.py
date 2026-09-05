"""
IEEE-CIS Fraud Detection — SHAP Explainability

Provides SHAP-based explanations for individual predictions
and global feature importance analysis.

Usage:
    cd backend
    python -m ml.explainability
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings('ignore')

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class FraudExplainer:
    """
    SHAP-based explainer for the fraud detection model.

    Provides:
    - Individual transaction explanations (top risk factors, top risk reducers)
    - Global feature importance
    - Reason generation in human-readable format
    """

    def __init__(self, model=None, preprocessor=None):
        if model is None:
            model = joblib.load(os.path.join(ARTIFACTS_DIR, 'fraud_model.joblib'))
        if preprocessor is None:
            preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))

        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.get_feature_names()

        self._explainer = shap.TreeExplainer(model)

    def explain_transaction(self, X_row, top_k=5):
        """
        Explain a single transaction prediction.

        Args:
            X_row: pandas DataFrame with one row (preprocessed features)
            top_k: number of top features to return

        Returns:
            dict with fraud_probability, risk_factors, risk_reducers, feature_impacts
        """
        if isinstance(X_row, pd.Series):
            X_row = X_row.to_frame().T

        shap_values = self._explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        impacts = list(zip(self.feature_names, sv))
        impacts.sort(key=lambda x: x[1], reverse=True)

        risk_factors = []
        for feat, val in impacts[:top_k]:
            if val > 0:
                risk_factors.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'increases_risk',
                    'value': round(float(X_row[feat].iloc[0]), 4) if feat in X_row.columns else None
                })

        risk_reducers = []
        for feat, val in reversed(impacts[-top_k:]):
            if val < 0:
                risk_reducers.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'decreases_risk',
                    'value': round(float(X_row[feat].iloc[0]), 4) if feat in X_row.columns else None
                })

        proba = float(self.model.predict_proba(X_row)[:, 1][0])

        return {
            'fraud_probability': round(proba, 4),
            'risk_factors': risk_factors,
            'risk_reducers': risk_reducers,
            'all_impacts': [{'feature': f, 'impact': round(float(v), 6)} for f, v in impacts],
        }

    def get_risk_reasons(self, X_row, top_k=3):
        """
        Generate human-readable risk reasons for a transaction.

        Returns:
            list of strings describing the top risk factors
        """
        explanation = self.explain_transaction(X_row, top_k=top_k)

        FEATURE_DESCRIPTIONS = {
            'TransactionAmt': 'Transaction amount',
            'log_TransactionAmt': 'Transaction amount (log scale)',
            'dist1': 'Distance from billing address',
            'dist2': 'Distance from shipping address',
            'dist_ratio': 'Billing/shipping distance ratio',
            'dist1_missing': 'Missing billing distance signal',
            'dist2_missing': 'Missing shipping distance signal',
            'C1': 'Transaction count (card, address)',
            'C2': 'Transaction count (card)',
            'C3': 'Address match pattern',
            'C4': 'Card-addr interaction',
            'C5': 'Transaction frequency',
            'C6': 'Card usage pattern',
            'C7': 'Addr usage pattern',
            'C8': 'Card-addr frequency',
            'C9': 'Card recency',
            'C10': 'Addr recency',
            'C11': 'Card-addr recency',
            'C12': 'Transaction recency',
            'C13': 'Card velocity',
            'C14': 'Address velocity',
            'D1': 'Days since card first transaction',
            'D2': 'Days since address first transaction',
            'D3': 'Card transaction interval',
            'D4': 'Address transaction interval',
            'D5': 'Card inactive period',
            'D6': 'Address inactive period',
            'D7': 'Card dormancy',
            'D8': 'Address dormancy',
            'D9': 'Card last seen',
            'D10': 'Address last seen',
            'D11': 'Card address co-occurrence',
            'D12': 'Transaction timing',
            'D13': 'Card pattern',
            'D14': 'Address pattern',
            'D15': 'Card-addr timing',
            'd_missing_count': 'Missing timedelta signals',
            'card1': 'Card identifier',
            'card2': 'Card brand/type',
            'card3': 'Card country',
            'card4': 'Card issuer',
            'card5': 'Card currency',
            'card6': 'Card type',
            'addr1': 'Billing region',
            'addr2': 'Billing country',
            'id_01': 'Identity verification score',
            'id_01_missing': 'Missing identity score',
            'id_02': 'Identity risk score',
            'id_12': 'Device registration status',
            'id_13': 'Device connection type',
            'id_14': 'Device count',
            'id_15': 'Device geolocation match',
            'id_16': 'Device session status',
            'id_17': 'Device fingerprint',
            'id_19': 'Device geo accuracy',
            'id_20': 'Device IP match',
            'id_30': 'Device OS',
            'id_31': 'Device browser',
            'id_33': 'Device screen resolution',
            'id_34': 'Device accuracy',
            'id_36': 'Device confidence',
            'id_37': 'Device timezone',
            'id_38': 'Device language',
            'ProductCD': 'Product category',
            'P_emaildomain': 'Purchaser email domain',
            'R_emaildomain': 'Recipient email domain',
            'M1': 'Card-addr name match',
            'M2': 'Card-addr email match',
            'M3': 'Card-addr phone match',
            'M4': 'Card-addr address match',
            'M5': 'Card-addr birth match',
            'M6': 'Card-addr SSN match',
            'M7': 'Card-addr device match',
            'M8': 'Card-addr IP match',
            'M9': 'Card-addr identity match',
            'DeviceType': 'Device type',
            'DeviceInfo': 'Device information',
            'email_known': 'Known email domain',
            'same_email_domain': 'Email domain consistency',
            'risk_signal_count': 'Elevated risk signal count',
            'card_addr_ratio': 'Card-address uniqueness',
        }

        reasons = []
        for factor in explanation['risk_factors']:
            feat = factor['feature']
            desc = FEATURE_DESCRIPTIONS.get(feat, feat)
            reasons.append(f"{desc} (impact: +{factor['impact']:.4f})")

        return reasons

    def global_feature_importance(self, X_sample, top_k=20):
        """
        Compute global feature importance using SHAP values.

        Args:
            X_sample: sample of preprocessed data (DataFrame)
            top_k: number of top features to return

        Returns:
            list of dicts with feature name and mean absolute SHAP value
        """
        shap_values = self._explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        mean_abs = np.mean(np.abs(sv), axis=0)

        importance = list(zip(self.feature_names, mean_abs))
        importance.sort(key=lambda x: x[1], reverse=True)

        return [
            {'feature': feat, 'importance': round(float(imp), 6)}
            for feat, imp in importance[:top_k]
        ]


def main():
    print("=" * 70)
    print("  IEEE-CIS FRAUD DETECTION — SHAP EXPLAINABILITY")
    print("=" * 70)

    print("\nLoading model and preprocessor...")
    explainer = FraudExplainer()

    print("Loading test data for sample explanations...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'held_out_test_set.csv'))
    test_for_preprocess = test_df.drop(columns=['isFraud'], errors='ignore')
    X_test = explainer.preprocessor.transform(test_for_preprocess)

    fraud_mask = test_df['isFraud'] == 1
    X_fraud = X_test[fraud_mask].head(5)
    X_legit = X_test[~fraud_mask].head(5)

    print("\n--- FRAUD CASES (Sample Explanations) ---")
    for i, (_, row) in enumerate(X_fraud.iterrows()):
        X_row = row.to_frame().T
        proba = float(explainer.model.predict_proba(X_row)[:, 1][0])
        print(f"\n  Transaction {i+1}: Fraud Probability = {proba:.4f}")
        reasons = explainer.get_risk_reasons(X_row, top_k=3)
        for j, reason in enumerate(reasons, 1):
            print(f"    {j}. {reason}")

    print("\n--- LEGITIMATE CASES (Sample Explanations) ---")
    for i, (_, row) in enumerate(X_legit.iterrows()):
        X_row = row.to_frame().T
        proba = float(explainer.model.predict_proba(X_row)[:, 1][0])
        print(f"\n  Transaction {i+1}: Fraud Probability = {proba:.4f}")
        reasons = explainer.get_risk_reasons(X_row, top_k=3)
        for j, reason in enumerate(reasons, 1):
            print(f"    {j}. {reason}")

    print("\n--- GLOBAL FEATURE IMPORTANCE ---")
    importance = explainer.global_feature_importance(X_test.sample(min(5000, len(X_test)), random_state=42))
    print(f"\n  {'Rank':>4} {'Feature':<25} {'Importance':>12}")
    print("  " + "-" * 42)
    for rank, item in enumerate(importance, 1):
        print(f"  {rank:>4} {item['feature']:<25} {item['importance']:>12.6f}")

    report_path = os.path.join(ARTIFACTS_DIR, 'feature_importance.json')
    with open(report_path, 'w') as f:
        json.dump(importance, f, indent=2)
    print(f"\n  Saved feature importance: {report_path}")

    print("\n" + "=" * 70)
    print("  EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
