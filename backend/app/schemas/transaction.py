from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from datetime import datetime


class TransactionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str = Field(..., json_schema_extra={"example": "TXN_987654321"})
    merchant_id: str = Field(..., json_schema_extra={"example": "MERCHANT_001"})
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_10022"})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 18500.00})
    device_id: str = Field(default="UNKNOWN", json_schema_extra={"example": "DEV_MOBILE_88"})
    is_new_device: bool = Field(default=False, json_schema_extra={"example": True})
    location: str = Field(default="UNKNOWN", json_schema_extra={"example": "Mumbai, IN"})
    is_new_location: bool = Field(default=False, json_schema_extra={"example": True})
    payment_method: str = Field(default="credit_card", json_schema_extra={"example": "credit_card"})
    velocity_5m: int = Field(default=1, ge=0, json_schema_extra={"example": 5})
    failed_attempts_24h: int = Field(default=0, ge=0, json_schema_extra={"example": 4})

    productcd: Optional[str] = Field(default=None, alias="ProductCD", json_schema_extra={"example": "W"})
    card1: Optional[float] = Field(default=None, json_schema_extra={"example": 13926})
    card2: Optional[float] = Field(default=None, json_schema_extra={"example": 404})
    card3: Optional[float] = Field(default=None, json_schema_extra={"example": 150})
    card4: Optional[str] = Field(default=None, json_schema_extra={"example": "visa"})
    card5: Optional[float] = Field(default=None, json_schema_extra={"example": 226})
    card6: Optional[str] = Field(default=None, json_schema_extra={"example": "credit"})
    addr1: Optional[float] = Field(default=None, json_schema_extra={"example": 315})
    addr2: Optional[float] = Field(default=None, json_schema_extra={"example": 87})
    dist1: Optional[float] = Field(default=None, json_schema_extra={"example": 24.0})
    dist2: Optional[float] = Field(default=None, json_schema_extra={"example": 6.0})
    p_emaildomain: Optional[str] = Field(default=None, alias="P_emaildomain", json_schema_extra={"example": "gmail.com"})
    r_emaildomain: Optional[str] = Field(default=None, alias="R_emaildomain", json_schema_extra={"example": "gmail.com"})
    devicetype: Optional[str] = Field(default=None, alias="DeviceType", json_schema_extra={"example": "desktop"})


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
