import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.api.routes import health, predictions, graph, investigation

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(predictions.router, prefix=settings.API_V1_STR, tags=["Predictions"])
app.include_router(graph.router, prefix=settings.API_V1_STR, tags=["Graph Intelligence"])
app.include_router(investigation.router, prefix=settings.API_V1_STR, tags=["Investigation"])

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    app.mount("/dashboard/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="dashboard-css")
    app.mount("/dashboard/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="dashboard-js")


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "Frontend not found"}


@app.get("/dashboard")
async def serve_dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "Frontend not found"}


@app.on_event("startup")
async def startup_load_graph():
    """Attempt to load the persisted graph from Supabase on startup.

    If the database is empty or unavailable, startup succeeds normally
    and the graph remains not-ready (requires POST /api/v1/graph/build).
    """
    try:
        from app.graph.graph_service import graph_service
        loaded = graph_service.load_from_db()
        if loaded:
            logger.info("Graph successfully restored from Supabase on startup.")
        else:
            logger.info("No persisted graph found on startup. Graph will remain not-built.")
    except Exception as exc:
        logger.warning("Startup graph load skipped: %s", exc)
