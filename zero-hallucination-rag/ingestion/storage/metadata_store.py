"""
In-memory metadata store.

Provides a simple dictionary-backed implementation of
:class:`~ingestion.interfaces.MetadataStore` for development and testing.
"""

from __future__ import annotations

from typing import Any, Optional

from ingestion.interfaces import MetadataStore
from utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryMetadataStore(MetadataStore):
    """
    Volatile, dictionary-backed metadata store.

    .. warning::
        All data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def store(self, document_id: str, metadata: dict[str, Any]) -> None:
        self._store[document_id] = metadata
        logger.debug("Stored metadata", document_id=document_id)

    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        return self._store.get(document_id)

    async def delete(self, document_id: str) -> bool:
        if document_id in self._store:
            del self._store[document_id]
            return True
        return False

    async def search(
        self,
        filters: dict[str, Any],
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Naive filter-based search over stored metadata.

        Each key-value pair in *filters* must match exactly.
        """
        results: list[dict[str, Any]] = []
        for doc_id, meta in self._store.items():
            if all(meta.get(k) == v for k, v in filters.items()):
                results.append({"document_id": doc_id, **meta})

        return results[offset : offset + limit]

    @property
    def count(self) -> int:
        return len(self._store)
