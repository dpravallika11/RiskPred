from fastapi import APIRouter, HTTPException, status
from app.schemas.transaction import TransactionCreate, PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_fraud_risk(transaction: TransactionCreate):
    try:
        response = prediction_service.predict(transaction)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )