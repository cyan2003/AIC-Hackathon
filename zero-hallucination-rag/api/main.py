"""
FastAPI application entry point for the Zero-Hallucination RAG Pipeline.

Provides resume → job description matching via:
  POST /ingest/resume  — Ingest a resume (PDF/DOCX)
  POST /ingest/jd      — Ingest a job description (PDF/DOCX)
  POST /match          — Match a resume to indexed JDs

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
      - Configure logging
      - Initialize vector store collections

    On shutdown:
      - Close connections
    """
    settings = get_settings()
    setup_logging(level=settings.log_level)
    logger = get_logger(__name__)
    logger.info(
        "Starting Zero-Hallucination RAG Pipeline",
        version=settings.app_version,
    )

    # Initialize Qdrant collections (best-effort; won't crash if Qdrant is down)
    try:
        from api.dependencies.database import get_vector_store

        vector_store = get_vector_store()
        await vector_store.initialize()
        logger.info("Vector store collections initialized")
    except Exception as exc:
        logger.warning(
            "Could not initialize vector store (continuing without it)",
            error=str(exc),
        )

    yield

    # Clean up connections on shutdown
    try:
        from api.dependencies.database import get_vector_store
        vector_store = get_vector_store()
        await vector_store.close()
        logger.info("Vector store connection closed")
    except Exception as exc:
        logger.warning(
            "Error closing vector store connection on shutdown",
            error=str(exc),
        )

    logger.info("Shutting down Zero-Hallucination RAG Pipeline")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="Zero-Hallucination RAG — Resume↔JD Matcher",
    description=(
        "A zero-hallucination retrieval pipeline that matches resumes "
        "to job descriptions using hybrid vector + lexical search with "
        "cross-encoder re-ranking."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS — allow all origins in dev; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from api.routers.ingestion import router as ingestion_router
from api.routers.retrieval import router as retrieval_router
from api.routers.health import router as health_router

app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(health_router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


@app.get("/", tags=["root"])
async def root():
    """Root endpoint — returns API info."""
    return {
        "service": "zero-hallucination-rag",
        "version": settings.app_version,
        "endpoints": {
            "ingest_resume": "POST /ingest/resume",
            "ingest_jd": "POST /ingest/jd",
            "match": "POST /match",
            "health": "GET /health",
            "readiness": "GET /health/ready",
            "metrics": "GET /metrics",
        },
    }
