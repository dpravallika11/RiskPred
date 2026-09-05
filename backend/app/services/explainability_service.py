import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ml_dir = os.path.join(_backend_dir, 'ml')
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)


class ExplainabilityService:
    """
    SHAP-based explainability service for fraud detection.

    Provides human-readable explanations for individual predictions,
    including both risk-increasing and risk-decreasing factors.
    """

    FEATURE_DESCRIPTIONS = {
        'TransactionAmt': 'Transaction amount',
        'log_TransactionAmt': 'Transaction amount (log scale)',
        'dist1': 'Distance from billing address',
        'dist2': 'Distance from shipping address',
        'dist_ratio': 'Billing/shipping distance ratio',
        'C1': 'Transaction count pattern (card-addr)',
        'C2': 'Transaction count pattern (card)',
        'C3': 'Address match pattern',
        'C5': 'Transaction frequency signal',
        'C6': 'Card usage pattern',
        'C7': 'Address usage pattern',
        'C13': 'Card velocity signal',
        'C14': 'Address velocity signal',
        'D1': 'Days since card first seen',
        'D2': 'Days since address first seen',
        'D3': 'Card transaction interval',
        'D4': 'Address transaction interval',
        'D15': 'Card-address timing',
        'card1': 'Card identifier',
        'card2': 'Card brand/type',
        'card4': 'Card issuer',
        'card6': 'Card type',
        'addr1': 'Billing region',
        'addr2': 'Billing country',
        'id_01': 'Identity verification score',
        'id_02': 'Identity risk score',
        'id_15': 'Device geolocation match',
        'id_17': 'Device fingerprint',
        'ProductCD': 'Product category',
        'P_emaildomain': 'Purchaser email domain',
        'R_emaildomain': 'Recipient email domain',
        'M1': 'Card-addr name match',
        'M2': 'Card-addr email match',
        'M3': 'Card-addr phone match',
        'M4': 'Card-addr address match',
        'M5': 'Card-addr birth date match',
        'M6': 'Card-addr SSN match',
        'DeviceType': 'Device type',
        'email_known': 'Known email domain',
        'same_email_domain': 'Email domain consistency',
        'risk_signal_count': 'Elevated risk signal count',
        'd_missing_count': 'Missing timedelta signals',
    }

    def __init__(self, model=None, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self._explainer = None

        if model is not None:
            import shap
            self._explainer = shap.TreeExplainer(model)

    def explain(self, X_row: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        if self._explainer is None:
            return {
                'risk_factors': [],
                'risk_reducers': [],
                'feature_impacts': [],
            }

        shap_values = self._explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        feature_names = self.preprocessor.get_feature_names() if self.preprocessor else []
        impacts = list(zip(feature_names, sv))
        impacts.sort(key=lambda x: x[1], reverse=True)

        risk_factors = []
        for feat, val in impacts[:top_k * 2]:
            if val > 0 and len(risk_factors) < top_k:
                risk_factors.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'increases_risk',
                    'description': self.FEATURE_DESCRIPTIONS.get(feat, feat),
                })

        risk_reducers = []
        for feat, val in reversed(impacts):
            if val < 0 and len(risk_reducers) < top_k:
                risk_reducers.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'decreases_risk',
                    'description': self.FEATURE_DESCRIPTIONS.get(feat, feat),
                })

        return {
            'risk_factors': risk_factors,
            'risk_reducers': risk_reducers,
            'feature_impacts': [
                {'feature': f, 'impact': round(float(v), 6)}
                for f, v in impacts[:20]
            ],
        }
