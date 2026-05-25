"""
Retrieval package — hybrid search pipeline combining vector and lexical retrieval.
"""

from retrieval.interfaces import (
    SearchResult,
    RetrievalQuery,
    RetrievalResponse,
    VectorStore,
    LexicalStore,
    ResultFuser,
    Reranker,
    QueryProcessor,
    QueryTransformer,
)
from retrieval.pipeline import RetrievalPipeline

__all__ = [
    # Data models
    "SearchResult",
    "RetrievalQuery",
    "RetrievalResponse",
    # Interfaces
    "VectorStore",
    "LexicalStore",
    "ResultFuser",
    "Reranker",
    "QueryProcessor",
    "QueryTransformer",
    # Pipeline
    "RetrievalPipeline",
]
