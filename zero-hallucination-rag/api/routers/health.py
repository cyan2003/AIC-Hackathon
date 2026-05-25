"""
Health check and monitoring endpoints.

GET /health       — Basic health check
GET /health/ready — Check vector store connectivity
GET /metrics      — Prometheus metrics (text format)
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse

from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns OK if the application is running.",
)
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok", "service": "zero-hallucination-rag"}


@router.get(
    "/health/ready",
    summary="Readiness check",
    description="Checks connectivity to the vector store and returns status.",
)
async def readiness_check():
    """
    Deep readiness probe — verifies that the vector store is reachable
    and both collections exist.
    """
    try:
        from core.config import get_settings
        from retrieval.vector_store.qdrant import QdrantVectorStore

        settings = get_settings()
        store = QdrantVectorStore(
            settings.retrieval.vector_store,
            embedding_dimension=settings.ingestion.embedding.dimension,
        )

        resume_count = await store.count(
            collection=settings.retrieval.vector_store.resume_collection
        )
        jd_count = await store.count(
            collection=settings.retrieval.vector_store.jd_collection
        )

        await store.close()

        return {
            "status": "ready",
            "vector_store": "connected",
            "collections": {
                "resumes": {"count": resume_count},
                "job_descriptions": {"count": jd_count},
            },
        }
    except Exception as exc:
        logger.warning("Readiness check failed", error=str(exc))
        return Response(
            content=f'{{"status": "not_ready", "error": "{exc}"}}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Returns metrics in Prometheus text exposition format.",
    response_class=PlainTextResponse,
)
async def prometheus_metrics():
    """Expose Prometheus metrics for scraping."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(
            content="# prometheus_client not installed\n",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
        )
