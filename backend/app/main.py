from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, predictions, graph, investigation

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(predictions.router, prefix=settings.API_V1_STR, tags=["Predictions"])
app.include_router(graph.router, prefix=settings.API_V1_STR, tags=["Graph Intelligence"])
app.include_router(investigation.router, prefix=settings.API_V1_STR, tags=["Investigation"])


@app.get("/")
async def root():
    return {"message": "Welcome to RiskPred API - AI Fraud Risk Manager"}
