"""
In-memory document store.

Provides a simple dictionary-backed implementation of
:class:`~ingestion.interfaces.DocumentStore` for development and testing.
Production deployments should replace this with a persistent backend
(e.g., PostgreSQL, S3, or a document database).
"""

from __future__ import annotations

from typing import Optional

from ingestion.interfaces import DocumentStore, RawDocument
from utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryDocumentStore(DocumentStore):
    """
    Volatile, dictionary-backed document store.

    .. warning::
        All data is lost when the process exits. Use only for
        development, testing, and prototyping.
    """

    def __init__(self) -> None:
        self._store: dict[str, RawDocument] = {}

    async def store(self, document_id: str, document: RawDocument) -> None:
        self._store[document_id] = document
        logger.debug("Stored document", document_id=document_id)

    async def get(self, document_id: str) -> Optional[RawDocument]:
        return self._store.get(document_id)

    async def delete(self, document_id: str) -> bool:
        if document_id in self._store:
            del self._store[document_id]
            logger.debug("Deleted document", document_id=document_id)
            return True
        return False

    async def exists(self, document_id: str) -> bool:
        return document_id in self._store

    @property
    def count(self) -> int:
        """Number of documents currently stored."""
        return len(self._store)
