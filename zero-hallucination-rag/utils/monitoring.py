"""
Performance monitoring and metrics collection.

Uses ``prometheus_client`` to expose counters, histograms, and gauges
that can be scraped by a Prometheus server or queried from the
``/metrics`` API endpoint.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Histogram, Gauge

# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total number of documents successfully ingested",
)

DOCUMENTS_FAILED = Counter(
    "rag_documents_failed_total",
    "Total number of documents that failed during ingestion",
)

INGESTION_DURATION = Histogram(
    "rag_ingestion_duration_seconds",
    "Time spent ingesting a single document batch",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

CHUNKS_CREATED = Counter(
    "rag_chunks_created_total",
    "Total number of text chunks created during ingestion",
)

# ---------------------------------------------------------------------------
# Embedding metrics
# ---------------------------------------------------------------------------

EMBEDDING_DURATION = Histogram(
    "rag_embedding_duration_seconds",
    "Time spent generating embeddings for a batch",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

EMBEDDINGS_GENERATED = Counter(
    "rag_embeddings_generated_total",
    "Total number of embedding vectors generated",
)

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

QUERIES_PROCESSED = Counter(
    "rag_queries_processed_total",
    "Total number of retrieval queries processed",
)

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "End-to-end retrieval query latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)

VECTOR_SEARCH_DURATION = Histogram(
    "rag_vector_search_duration_seconds",
    "Time spent in vector-similarity search",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

LEXICAL_SEARCH_DURATION = Histogram(
    "rag_lexical_search_duration_seconds",
    "Time spent in lexical (BM25) search",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

RERANKER_DURATION = Histogram(
    "rag_reranker_duration_seconds",
    "Time spent in the cross-encoder re-ranker",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# ---------------------------------------------------------------------------
# System metrics
# ---------------------------------------------------------------------------

ACTIVE_INGESTION_TASKS = Gauge(
    "rag_active_ingestion_tasks",
    "Number of ingestion tasks currently in progress",
)

VECTOR_STORE_DOCUMENT_COUNT = Gauge(
    "rag_vector_store_document_count",
    "Current number of vectors stored in the vector database",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def track_duration(histogram: Histogram) -> Generator[None, None, None]:
    """Context manager that records elapsed wall-clock time to *histogram*."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)
