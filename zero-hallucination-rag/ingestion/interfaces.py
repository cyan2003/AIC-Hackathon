"""
Abstract base classes (interfaces) for the ingestion pipeline components.

Every concrete implementation in the ingestion module must implement one
of these interfaces, ensuring the pipeline orchestrator can compose
arbitrary processing stages without coupling to specific implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from core.config import DocumentType


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RawDocument:
    """
    Represents a document before any processing.

    Attributes
    ----------
    source : str
        Origin path, URL, or identifier for the document.
    content : str | bytes
        Raw content (text or binary depending on file type).
    mime_type : str
        MIME type of the source file.
    document_type : DocumentType | None
        Whether this is a resume or job description.
    metadata : dict
        Arbitrary key-value metadata supplied at ingestion time.
    """

    source: str
    content: str | bytes
    mime_type: str = "text/plain"
    document_type: Optional[DocumentType] = None
    trust_rating: float = 1.0
    created_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedChunk:
    """
    A chunk of text ready for embedding.

    Attributes
    ----------
    chunk_id : str
        Unique identifier for this chunk.
    document_id : str
        Parent document identifier.
    text : str
        Chunk text content.
    chunk_index : int
        Positional index within the parent document.
    metadata : dict
        Merged document + chunk-level metadata.
    """

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """
    A chunk with its embedding vector attached.

    Attributes
    ----------
    chunk : ProcessedChunk
        The source chunk.
    embedding : list[float]
        Dense vector representation.
    """

    chunk: ProcessedChunk
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class DocumentValidator(ABC):
    """Validates incoming documents against acceptance criteria."""

    @abstractmethod
    def validate(self, document: RawDocument) -> bool:
        """Return ``True`` if *document* is acceptable for ingestion."""
        ...

    @abstractmethod
    def validation_errors(self, document: RawDocument) -> list[str]:
        """Return a list of human-readable validation error messages."""
        ...


class DocumentCleaner(ABC):
    """Normalises and cleans raw document text."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Return cleaned text."""
        ...


class DocumentChunker(ABC):
    """Splits cleaned text into chunks."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        """Return a list of :class:`ProcessedChunk` instances."""
        ...


class MetadataExtractor(ABC):
    """Extracts structured metadata from raw documents."""

    @abstractmethod
    def extract(self, document: RawDocument) -> dict[str, Any]:
        """Return extracted metadata as a dict."""
        ...


class Embedder(ABC):
    """Generates dense vector embeddings from text."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Convenience method for a single text string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of produced embeddings."""
        ...


class DocumentStore(ABC):
    """Persists raw document content for later retrieval / audit."""

    @abstractmethod
    async def store(self, document_id: str, document: RawDocument) -> None:
        ...

    @abstractmethod
    async def get(self, document_id: str) -> Optional[RawDocument]:
        ...

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        ...

    @abstractmethod
    async def exists(self, document_id: str) -> bool:
        ...


class MetadataStore(ABC):
    """Persists extracted metadata for filtering and analytics."""

    @abstractmethod
    async def store(self, document_id: str, metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        ...

    @abstractmethod
    async def search(
        self,
        filters: dict[str, Any],
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        ...
