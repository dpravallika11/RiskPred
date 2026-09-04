from fastapi import APIRouter, HTTPException, status
from app.schemas.transaction import TransactionCreate, PredictionResponse, RiskFactor
from app.services.prediction_service import prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_fraud_risk(transaction: TransactionCreate):
    """
    Predict fraud risk for a transaction.

    Returns fraud probability, calibrated risk score (0-100),
    risk level (LOW/MEDIUM/HIGH), recommended action, and
    SHAP-based risk factor explanations.
    """
    try:
        txn_dict = transaction.model_dump()
        result = prediction_service.predict(txn_dict)

        response = PredictionResponse(
            transaction_id=result['transaction_id'],
            fraud_probability=result['fraud_probability'],
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            recommended_action=result['recommended_action'],
            top_risk_factors=[
                RiskFactor(**f) for f in result['top_risk_factors']
            ],
            top_risk_reducers=[
                RiskFactor(**f) for f in result['top_risk_reducers']
            ],
            prediction_timestamp=result['prediction_timestamp'],
        )
        return response
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
