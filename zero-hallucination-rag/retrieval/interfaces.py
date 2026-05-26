"""
Abstract base classes (interfaces) for the retrieval pipeline components.

Defines the contracts that vector stores, lexical stores, fusion strategies,
query processors, and the retrieval pipeline itself must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """
    A single result returned by a search backend.

    Attributes
    ----------
    document_id : str
        The parent document's identifier.
    chunk_id : str
        The matched chunk's identifier.
    text : str
        The chunk text that was matched.
    score : float
        Relevance score (higher is better; scale depends on backend).
    metadata : dict
        Associated metadata for filtering / display.
    source : str
        Which backend produced this result (``"vector"``, ``"lexical"``).
    """

    document_id: str
    chunk_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class RetrievalQuery:
    """
    A fully-processed query ready for execution against search backends.

    Attributes
    ----------
    original_text : str
        The raw user query string.
    processed_text : str
        Query text after preprocessing / expansion.
    embedding : list[float] | None
        Dense vector for similarity search (populated by the pipeline).
    filters : dict
        Metadata filters to apply during search.
    top_k : int
        Maximum number of results to return.
    """

    original_text: str
    processed_text: str = ""
    embedding: Optional[list[float]] = None
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int = 10


@dataclass
class RetrievalResponse:
    """
    Aggregated response from the retrieval pipeline.

    Attributes
    ----------
    query : RetrievalQuery
        The query that was executed.
    results : list[SearchResult]
        Ranked list of results after fusion / re-ranking.
    total_candidates : int
        Number of candidates *before* final truncation to top_k.
    timings : dict
        Stage-level latencies in seconds.
    """

    query: RetrievalQuery
    results: list[SearchResult] = field(default_factory=list)
    total_candidates: int = 0
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Interface for dense vector similarity search backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create collections / indices if they do not exist."""
        ...

    @abstractmethod
    async def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        """Insert or update vectors with optional payloads."""
        ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Return the *top_k* nearest neighbours for *vector*."""
        ...

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors by ID."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors stored."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources."""
        ...


class LexicalStore(ABC):
    """Interface for keyword / lexical search backends."""

    @abstractmethod
    async def index(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        """Index texts for lexical search."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most relevant results for *query*."""
        ...

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Remove documents by ID."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of indexed documents."""
        ...


class ResultFuser(ABC):
    """Merges ranked lists from multiple backends into a single ranking."""

    @abstractmethod
    def fuse(
        self,
        result_lists: Sequence[list[SearchResult]],
    ) -> list[SearchResult]:
        """Return a fused, deduplicated, and re-scored ranking."""
        ...


class Reranker(ABC):
    """Cross-encoder or other re-ranking model interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Re-score and re-order *results* given *query*."""
        ...


class QueryProcessor(ABC):
    """Preprocesses and optionally expands user queries."""

    @abstractmethod
    def process(self, raw_query: str) -> str:
        """Return the processed query text."""
        ...


class QueryTransformer(ABC):
    """Transforms a query into backend-specific representations."""

    @abstractmethod
    def for_vector(self, query: RetrievalQuery) -> RetrievalQuery:
        """Prepare the query for vector search."""
        ...

    @abstractmethod
    def for_lexical(self, query: RetrievalQuery) -> RetrievalQuery:
        """Prepare the query for lexical search."""
        ...
