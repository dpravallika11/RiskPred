import os
import joblib
import pandas as pd
from datetime import datetime
from app.core.config import settings
from ml.preprocessing import FraudDataPreprocessor
from app.services.explainability_service import ExplainabilityService
from app.schemas.transaction import TransactionCreate, PredictionResponse

class PredictionService:
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.scaler_path = settings.SCALER_PATH
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.preprocessor = FraudDataPreprocessor()
            self.preprocessor.load_scaler(self.scaler_path)
            self.explainability_service = ExplainabilityService(self.preprocessor.feature_columns)
            self.is_ready = True
        else:
            self.is_ready = False

    def predict(self, txn: TransactionCreate) -> PredictionResponse:
        if not self.is_ready:
            raise RuntimeError("Model artifacts not found. Run training pipeline first.")

        # 1. Convert input schema to DataFrame
        raw_df = pd.DataFrame([{
            'amount': txn.amount,
            'is_new_device': int(txn.is_new_device),
            'is_new_location': int(txn.is_new_location),
            'velocity_5m': txn.velocity_5m,
            'failed_attempts_24h': txn.failed_attempts_24h
        }])

        # 2. Scale features
        scaled_features = self.preprocessor.transform(raw_df)

        # 3. Model Inference (Fraud Probability)
        proba = float(self.model.predict_proba(scaled_features)[0][1])
        
        # 4. Map probability to Risk Score (0 - 100)
        risk_score = int(round(proba * 100))

        # 5. Risk Classification & Action Recommendation
        if risk_score >= settings.CRITICAL_RISK_THRESHOLD:
            risk_level = "CRITICAL"
            recommended_action = "BLOCK"
        elif risk_score >= settings.HIGH_RISK_THRESHOLD:
            risk_level = "HIGH"
            recommended_action = "REVIEW"
        elif risk_score >= settings.MEDIUM_RISK_THRESHOLD:
            risk_level = "MEDIUM"
            recommended_action = "ALLOW + MONITOR"
        else:
            risk_level = "LOW"
            recommended_action = "ALLOW"

        # 6. Generate Risk Explanations
        reasons = self.explainability_service.explain_transaction(txn.dict(), proba)

        return PredictionResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=round(proba, 4),
            risk_score=risk_score,
            risk_level=risk_level,
            recommended_action=recommended_action,
            top_reasons=reasons,
            prediction_timestamp=datetime.utcnow()
        )

prediction_service = PredictionService()