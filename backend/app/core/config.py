import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "RiskPred AI Risk Manager"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://postgres:postgres@localhost:5432/riskpred"
    )
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ml/artifacts/fraud_model.joblib")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "ml/artifacts/scaler.joblib")
    
    MEDIUM_RISK_THRESHOLD: float = 30.0
    HIGH_RISK_THRESHOLD: float = 60.0
    CRITICAL_RISK_THRESHOLD: float = 80.0

    class Config:
        case_sensitive = True

settings = Settings()