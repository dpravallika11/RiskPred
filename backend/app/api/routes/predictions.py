from fastapi import APIRouter, HTTPException, status
from app.schemas.transaction import TransactionCreate, PredictionResponse, RiskFactor
from app.services.prediction_service import prediction_service
from app.services.transaction_store import transaction_store
from app.db.repositories import transaction_repo, prediction_repo, risk_factor_repo

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
        transaction_store.put(txn_dict)
        try:
            transaction_repo.create(txn_dict)
        except Exception as exc:
            print(f"[PERSISTENCE ERROR] transaction_id={txn_dict.get('transaction_id')}: {exc}")
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

        # Persist prediction result
        prediction_id = None
        try:
            prediction_data = {
                "transaction_id": result['transaction_id'],
                "fraud_probability": result['fraud_probability'],
                "risk_score": result['risk_score'],
                "risk_level": result['risk_level'],
                "recommended_action": result['recommended_action'],
                "prediction_timestamp": result['prediction_timestamp'],
            }
            prediction_record = prediction_repo.create(prediction_data)
            if prediction_record:
                prediction_id = prediction_record.get("id")
        except Exception as exc:
            print(f"[PERSISTENCE ERROR] prediction persistence failed for transaction_id={result.get('transaction_id')}: {exc}")

        # Persist risk factors
        if prediction_id:
            all_risk_factors = result['top_risk_factors'] + result['top_risk_reducers']
            if all_risk_factors:
                try:
                    risk_factor_repo.create_many(prediction_id, all_risk_factors)
                except Exception as exc:
                    print(f"[PERSISTENCE ERROR] risk factors persistence failed for prediction_id={prediction_id}: {exc}")

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
