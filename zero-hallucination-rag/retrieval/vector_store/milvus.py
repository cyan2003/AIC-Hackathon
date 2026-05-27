"""
Milvus vector store implementation (stub).

Provides the scaffolding for a Milvus-backed vector store.
Full implementation is deferred to a later phase; this stub ensures
the factory can instantiate the class and tests can exercise the
interface.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from core.config import VectorStoreConfig
from core.exceptions import VectorStoreError
from retrieval.interfaces import SearchResult, VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)


class MilvusVectorStore(VectorStore):
    """
    Milvus-backed vector store (stub implementation).

    .. note::
        This is a placeholder. The full implementation using ``pymilvus``
        will be added in a subsequent phase.

    Parameters
    ----------
    config:
        Vector store configuration.
    embedding_dimension:
        Dimensionality of the vectors to be stored.
    """

    def __init__(self, config: VectorStoreConfig, embedding_dimension: int) -> None:
        self._config = config
        self._dimension = embedding_dimension

    async def initialize(self) -> None:
        logger.warning("MilvusVectorStore.initialize() is a stub – not yet implemented")

    async def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError("Milvus upsert is not yet implemented")

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        raise NotImplementedError("Milvus search is not yet implemented")

    async def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError("Milvus delete is not yet implemented")

    async def count(self) -> int:
        raise NotImplementedError("Milvus count is not yet implemented")

    async def close(self) -> None:
        logger.info("MilvusVectorStore closed (stub)")

    async def search_jds(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        raise NotImplementedError("Milvus search_jds is not yet implemented")

    async def upsert_resumes(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError("Milvus upsert_resumes is not yet implemented")

    async def upsert_jds(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError("Milvus upsert_jds is not yet implemented")
