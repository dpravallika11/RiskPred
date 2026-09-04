from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TransactionCreate(BaseModel):
    transaction_id: str = Field(..., example="TXN_987654321")
    merchant_id: str = Field(..., example="MERCHANT_001")
    customer_id: str = Field(..., example="CUST_10022")
    amount: float = Field(..., gt=0, example=18500.00)
    device_id: str = Field(default="UNKNOWN", example="DEV_MOBILE_88")
    is_new_device: bool = Field(default=False, example=True)
    location: str = Field(default="UNKNOWN", example="Mumbai, IN")
    is_new_location: bool = Field(default=False, example=True)
    payment_method: str = Field(default="credit_card", example="credit_card")
    velocity_5m: int = Field(default=1, ge=0, example=5)
    failed_attempts_24h: int = Field(default=0, ge=0, example=4)

    ProductCD: Optional[str] = Field(default=None, example="W")
    card1: Optional[float] = Field(default=None, example=13926)
    card2: Optional[float] = Field(default=None, example=404)
    card3: Optional[float] = Field(default=None, example=150)
    card4: Optional[str] = Field(default=None, example="visa")
    card5: Optional[float] = Field(default=None, example=226)
    card6: Optional[str] = Field(default=None, example="credit")
    addr1: Optional[float] = Field(default=None, example=315)
    addr2: Optional[float] = Field(default=None, example=87)
    dist1: Optional[float] = Field(default=None, example=24.0)
    dist2: Optional[float] = Field(default=None, example=6.0)
    P_emaildomain: Optional[str] = Field(default=None, example="gmail.com")
    R_emaildomain: Optional[str] = Field(default=None, example="gmail.com")
    DeviceType: Optional[str] = Field(default=None, example="desktop")


class RiskFactor(BaseModel):
    feature: str
    impact: float
    direction: str
    description: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int
    risk_level: str
    recommended_action: str
    top_risk_factors: List[RiskFactor]
    top_risk_reducers: List[RiskFactor]
    prediction_timestamp: datetime
