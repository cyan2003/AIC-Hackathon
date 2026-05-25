"""
API routers sub-package.
"""

from api.routers.ingestion import router as ingestion_router
from api.routers.retrieval import router as retrieval_router
from api.routers.health import router as health_router

__all__ = ["ingestion_router", "retrieval_router", "health_router"]
