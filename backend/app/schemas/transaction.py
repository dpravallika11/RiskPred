from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TransactionCreate(BaseModel):
    transaction_id: str = Field(..., example="TXN_987654321")
    merchant_id: str = Field(..., example="MERCHANT_001")
    customer_id: str = Field(..., example="CUST_10022")
    amount: float = Field(..., gt=0, example=18500.00)
    device_id: str = Field(..., example="DEV_MOBILE_88")
    is_new_device: bool = Field(default=False, example=True)
    location: str = Field(..., example="Mumbai, IN")
    is_new_location: bool = Field(default=False, example=True)
    payment_method: str = Field(..., example="credit_card")
    velocity_5m: int = Field(default=1, example=5)
    failed_attempts_24h: int = Field(default=0, example=4)

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int
    risk_level: str
    recommended_action: str
    top_reasons: List[str]
    prediction_timestamp: datetime