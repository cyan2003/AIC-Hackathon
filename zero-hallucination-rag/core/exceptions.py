"""
Custom exception hierarchy for the Zero-Hallucination RAG Pipeline.

All domain-specific exceptions inherit from ``RAGBaseError`` to allow
callers to catch the entire family in a single ``except`` clause when needed.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class RAGBaseError(Exception):
    """Root exception for all RAG pipeline errors."""

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Ingestion exceptions
# ---------------------------------------------------------------------------


class IngestionError(RAGBaseError):
    """Raised when a generic ingestion failure occurs."""


class DocumentValidationError(IngestionError):
    """Raised when a document fails validation (format, size, encoding, etc.)."""


class DocumentParsingError(IngestionError):
    """Raised when a document cannot be parsed into text."""


class ChunkingError(IngestionError):
    """Raised when the chunking stage fails."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when generated embeddings have an unexpected dimension."""


# ---------------------------------------------------------------------------
# Retrieval exceptions
# ---------------------------------------------------------------------------


class RetrievalError(RAGBaseError):
    """Raised when a generic retrieval failure occurs."""


class VectorStoreError(RetrievalError):
    """Raised when the vector store backend reports an error."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when a connection to the vector store cannot be established."""


class LexicalStoreError(RetrievalError):
    """Raised when the lexical search backend reports an error."""


class FusionError(RetrievalError):
    """Raised when result fusion / re-ranking fails."""


class QueryProcessingError(RetrievalError):
    """Raised when query preprocessing or expansion fails."""


# ---------------------------------------------------------------------------
# Storage exceptions
# ---------------------------------------------------------------------------


class StorageError(RAGBaseError):
    """Raised when a storage operation fails."""


class DocumentNotFoundError(StorageError):
    """Raised when a requested document does not exist in the store."""


# ---------------------------------------------------------------------------
# Configuration exceptions
# ---------------------------------------------------------------------------


class ConfigurationError(RAGBaseError):
    """Raised when the application configuration is invalid or missing."""
