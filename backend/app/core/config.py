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

    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "ml", "artifacts", "fraud_model.joblib")
    )
    PREPROCESSOR_PATH: str = os.getenv(
        "PREPROCESSOR_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "ml", "artifacts", "preprocessor.joblib")
    )

    # Risk thresholds — cost-optimized on validation set
    # LOW: 0-30, MEDIUM: 31-70, HIGH: 71-100
    # Determined by minimizing total expected cost (FN=$500, FP=$50)
    LOW_RISK_MAX: int = 30
    MEDIUM_RISK_MAX: int = 70

    class Config:
        case_sensitive = True


settings = Settings()
