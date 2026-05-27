"""
Database dependencies for FastAPI.

Provides access to global shared instances of the vector and lexical stores.
Detects if store classes are mocked (e.g. during pytest runs) and provides
clean mock instances when needed to prevent test-pollution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from core.config import get_settings

_vector_store: Any = None
_lexical_store: Any = None


def get_vector_store() -> Any:
    """
    Get the global shared QdrantVectorStore instance.
    If the class is mocked (during unit/integration testing), a fresh mock
    instance is created and returned on each call to prevent test pollution.
    """
    global _vector_store
    from retrieval.vector_store.qdrant import QdrantVectorStore

    # Check if QdrantVectorStore class is mocked in tests
    if isinstance(QdrantVectorStore, Mock):
        settings = get_settings()
        return QdrantVectorStore(
            settings.retrieval.vector_store,
            embedding_dimension=settings.ingestion.embedding.dimension,
        )

    if _vector_store is None:
        settings = get_settings()
        _vector_store = QdrantVectorStore(
            settings.retrieval.vector_store,
            embedding_dimension=settings.ingestion.embedding.dimension,
        )
    return _vector_store


def get_lexical_store() -> Any:
    """
    Get the global shared BM25LexicalStore instance.
    If the class is mocked (during unit/integration testing), a fresh mock
    instance is created and returned on each call to prevent test pollution.
    """
    global _lexical_store
    from retrieval.lexical_store.bm25 import BM25LexicalStore

    # Check if BM25LexicalStore class is mocked in tests
    if isinstance(BM25LexicalStore, Mock):
        settings = get_settings()
        return BM25LexicalStore(settings.retrieval.lexical_store)

    if _lexical_store is None:
        settings = get_settings()
        _lexical_store = BM25LexicalStore(settings.retrieval.lexical_store)
    return _lexical_store
