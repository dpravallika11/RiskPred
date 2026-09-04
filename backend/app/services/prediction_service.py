import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from typing import List, Dict, Any

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ml_dir = os.path.join(_backend_dir, 'ml')
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)

from ml.preprocessing import FraudPreprocessor


class PredictionService:
    """
    Production prediction service for fraud detection.

    Loads the trained model and preprocessor, accepts transaction data,
    and returns fraud probability, risk score, risk level, and SHAP-based
    explanations.
    """

    FEATURE_DESCRIPTIONS = {
        'TransactionAmt': 'Transaction amount',
        'log_TransactionAmt': 'Transaction amount (log scale)',
        'dist1': 'Distance from billing address',
        'dist2': 'Distance from shipping address',
        'dist_ratio': 'Billing/shipping distance ratio',
        'dist1_missing': 'Missing billing distance signal',
        'dist2_missing': 'Missing shipping distance signal',
        'C1': 'Transaction count pattern (card-addr)',
        'C2': 'Transaction count pattern (card)',
        'C3': 'Address match pattern',
        'C4': 'Card-addr interaction pattern',
        'C5': 'Transaction frequency signal',
        'C6': 'Card usage pattern',
        'C7': 'Address usage pattern',
        'C8': 'Card-addr frequency',
        'C9': 'Card recency signal',
        'C10': 'Address recency signal',
        'C11': 'Card-addr recency',
        'C12': 'Transaction recency',
        'C13': 'Card velocity signal',
        'C14': 'Address velocity signal',
        'D1': 'Days since card first seen',
        'D2': 'Days since address first seen',
        'D3': 'Card transaction interval',
        'D4': 'Address transaction interval',
        'D5': 'Card inactive period',
        'D6': 'Address inactive period',
        'D7': 'Card dormancy',
        'D8': 'Address dormancy',
        'D9': 'Card last seen interval',
        'D10': 'Address last seen interval',
        'D11': 'Card-address co-occurrence',
        'D12': 'Transaction timing',
        'D13': 'Card behavior pattern',
        'D14': 'Address behavior pattern',
        'D15': 'Card-address timing',
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
        'M5': 'Card-addr birth date match',
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

    def __init__(self):
        self.is_ready = False
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.threshold = 0.5
        self.threshold_config = None
        self.feature_columns = []

        self._load_artifacts()

    def _load_artifacts(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifacts_dir = os.path.join(backend_dir, 'ml', 'artifacts')

        model_path = os.path.join(artifacts_dir, 'fraud_model.joblib')
        preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.joblib')
        threshold_path = os.path.join(artifacts_dir, 'threshold_config.json')
        features_path = os.path.join(artifacts_dir, 'feature_columns.json')

        if not all(os.path.exists(p) for p in [model_path, preprocessor_path]):
            print("Warning: ML artifacts not found. Prediction service unavailable.")
            return

        try:
            self.model = joblib.load(model_path)
            self.preprocessor = FraudPreprocessor.load(preprocessor_path)
            self.feature_columns = self.preprocessor.get_feature_names()

            if os.path.exists(threshold_path):
                with open(threshold_path) as f:
                    self.threshold_config = json.load(f)
                self.threshold = self.threshold_config.get('threshold', 0.5)

            import shap
            self.explainer = shap.TreeExplainer(self.model)

            self.is_ready = True
            print(f"Prediction service loaded: {type(self.model).__name__}, "
                  f"{len(self.feature_columns)} features, threshold={self.threshold:.3f}")
        except Exception as e:
            print(f"Warning: Failed to load ML artifacts: {e}")
            self.is_ready = False

    def _build_feature_vector(self, txn_dict: Dict[str, Any]) -> pd.DataFrame:
        raw_data = {
            'TransactionAmt': txn_dict.get('amount', 0),
            'TransactionDT': 86400,
            'isFraud': 0,
        }

        optional_raw = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
            'C11', 'C12', 'C13', 'C14',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
            'D11', 'D12', 'D13', 'D14', 'D15',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain', 'R_emaildomain',
            'DeviceType', 'DeviceInfo',
            'id_01', 'id_02', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
            'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_33', 'id_34',
            'id_36', 'id_37', 'id_38',
        ]
        for col in optional_raw:
            if col in txn_dict and txn_dict[col] is not None:
                raw_data[col] = txn_dict[col]

        return pd.DataFrame([raw_data])

    def predict(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Model artifacts not found. Run training pipeline first.")

        raw_df = self._build_feature_vector(txn_dict)
        X = self.preprocessor.transform(raw_df)

        proba = float(self.model.predict_proba(X)[:, 1][0])

        risk_score = int(round(proba * 100))

        if risk_score >= 71:
            risk_level = "HIGH"
            recommended_action = "MANUAL_REVIEW"
        elif risk_score >= 31:
            risk_level = "MEDIUM"
            recommended_action = "VERIFY"
        else:
            risk_level = "LOW"
            recommended_action = "ALLOW"

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        impacts = list(zip(self.feature_columns, sv))
        impacts.sort(key=lambda x: x[1], reverse=True)

        risk_factors = []
        for feat, val in impacts[:5]:
            if val > 0:
                risk_factors.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'increases_risk',
                    'description': self.FEATURE_DESCRIPTIONS.get(feat, feat),
                })

        risk_reducers = []
        for feat, val in reversed(impacts[-5:]):
            if val < 0:
                risk_reducers.append({
                    'feature': feat,
                    'impact': round(float(val), 6),
                    'direction': 'decreases_risk',
                    'description': self.FEATURE_DESCRIPTIONS.get(feat, feat),
                })

        if txn_dict.get('velocity_5m', 1) > 3:
            risk_factors.append({
                'feature': 'velocity_5m',
                'impact': 0.01,
                'direction': 'increases_risk',
                'description': f"High transaction velocity ({txn_dict['velocity_5m']} txns in 5 min)",
            })

        if txn_dict.get('failed_attempts_24h', 0) >= 2:
            risk_factors.append({
                'feature': 'failed_attempts_24h',
                'impact': 0.01,
                'direction': 'increases_risk',
                'description': f"Multiple failed attempts ({txn_dict['failed_attempts_24h']} in 24h)",
            })

        if txn_dict.get('is_new_device', False):
            risk_factors.append({
                'feature': 'is_new_device',
                'impact': 0.005,
                'direction': 'increases_risk',
                'description': "Unrecognized device fingerprint",
            })

        if txn_dict.get('is_new_location', False):
            risk_factors.append({
                'feature': 'is_new_location',
                'impact': 0.005,
                'direction': 'increases_risk',
                'description': "New geographical location",
            })

        return {
            'transaction_id': txn_dict.get('transaction_id', 'UNKNOWN'),
            'fraud_probability': round(proba, 4),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommended_action': recommended_action,
            'top_risk_factors': risk_factors,
            'top_risk_reducers': risk_reducers,
            'prediction_timestamp': datetime.now(timezone.utc).isoformat(),
        }


prediction_service = PredictionService()
