"""
Ingestion package — document intake, processing, and embedding pipeline.
"""

from ingestion.interfaces import (
    RawDocument,
    ProcessedChunk,
    EmbeddedChunk,
    DocumentValidator,
    DocumentCleaner,
    DocumentChunker,
    MetadataExtractor,
    Embedder,
    DocumentStore,
    MetadataStore,
)
from ingestion.pipeline import IngestionPipeline, IngestionResult, BatchIngestionResult

__all__ = [
    # Data models
    "RawDocument",
    "ProcessedChunk",
    "EmbeddedChunk",
    # Interfaces
    "DocumentValidator",
    "DocumentCleaner",
    "DocumentChunker",
    "MetadataExtractor",
    "Embedder",
    "DocumentStore",
    "MetadataStore",
    # Pipeline
    "IngestionPipeline",
    "IngestionResult",
    "BatchIngestionResult",
]
