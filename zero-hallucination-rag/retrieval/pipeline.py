"""
Main retrieval pipeline orchestrator.

Composes query processing → vector search → lexical search → fusion
→ optional re-ranking into a single, configurable retrieval pipeline.
"""

from __future__ import annotations

import time
from typing import Optional

from core.config import RetrievalConfig
from core.exceptions import RetrievalError
from ingestion.interfaces import Embedder
from retrieval.interfaces import (
    LexicalStore,
    QueryProcessor,
    Reranker,
    ResultFuser,
    RetrievalQuery,
    RetrievalResponse,
    SearchResult,
    VectorStore,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Hybrid retrieval pipeline combining vector and lexical search.

    Parameters
    ----------
    config:
        Retrieval configuration (top_k, thresholds, fusion params, etc.).
    vector_store:
        Dense vector search backend.
    lexical_store:
        Keyword / BM25 search backend.
    embedder:
        Embedding model (shared with ingestion) for query vectorisation.
    fuser:
        Strategy for merging results from multiple backends.
    query_processor:
        Query text preprocessing / expansion.
    reranker:
        Optional cross-encoder re-ranker.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        embedder: Embedder,
        fuser: ResultFuser,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._lexical_store = lexical_store
        self._embedder = embedder
        self._fuser = fuser
        self._query_processor = query_processor
        self._reranker = reranker

    # -- Public API -------------------------------------------------------

    async def retrieve(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> RetrievalResponse:
        """
        Execute a hybrid retrieval query.

        Stages
        ------
        1. Preprocess the raw query text.
        2. Generate a query embedding.
        3. Run vector search and lexical search in parallel (logically).
        4. Fuse results from both backends.
        5. Optionally re-rank with a cross-encoder.
        6. Apply score threshold and truncate to ``top_k``.
        """
        timings: dict[str, float] = {}
        effective_top_k = top_k or self._config.top_k

        # 1. Preprocess
        t0 = time.perf_counter()
        processed_text = (
            self._query_processor.process(query_text)
            if self._query_processor
            else query_text
        )
        timings["query_processing"] = time.perf_counter() - t0

        # 2. Embed query
        t0 = time.perf_counter()
        query_embedding = self._embedder.embed_single(processed_text)
        timings["query_embedding"] = time.perf_counter() - t0

        query = RetrievalQuery(
            original_text=query_text,
            processed_text=processed_text,
            embedding=query_embedding,
            filters=filters or {},
            top_k=effective_top_k,
        )

        # 3. Search both backends
        vector_results, lexical_results = await self._search_backends(query, timings)

        # 4. Fuse
        t0 = time.perf_counter()
        fused = self._fuser.fuse([vector_results, lexical_results])
        timings["fusion"] = time.perf_counter() - t0

        total_candidates = len(fused)

        # 5. Re-rank (optional)
        if self._reranker and self._config.enable_reranking:
            t0 = time.perf_counter()
            fused = await self._reranker.rerank(
                processed_text,
                fused,
                top_k=self._config.fusion.reranker_top_k,
            )
            timings["reranking"] = time.perf_counter() - t0

        # 6. Filter and truncate
        results = self._apply_threshold(fused)[:effective_top_k]

        logger.info(
            "Retrieval complete",
            query=query_text[:80],
            results=len(results),
            candidates=total_candidates,
        )

        return RetrievalResponse(
            query=query,
            results=results,
            total_candidates=total_candidates,
            timings=timings,
        )

    # -- Private helpers --------------------------------------------------

    async def _search_backends(
        self,
        query: RetrievalQuery,
        timings: dict[str, float],
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Run vector and lexical search sequentially (async-ready)."""
        # Vector search
        t0 = time.perf_counter()
        vector_results: list[SearchResult] = []
        if query.embedding:
            vector_results = await self._vector_store.search(
                query.embedding,
                top_k=query.top_k * 2,  # over-fetch for better fusion
                filters=query.filters or None,
            )
            for r in vector_results:
                r.source = "vector"
        timings["vector_search"] = time.perf_counter() - t0

        # Lexical search
        t0 = time.perf_counter()
        lexical_results = await self._lexical_store.search(
            query.processed_text,
            top_k=query.top_k * 2,
            filters=query.filters or None,
        )
        for r in lexical_results:
            r.source = "lexical"
        timings["lexical_search"] = time.perf_counter() - t0

        return vector_results, lexical_results

    def _apply_threshold(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove results below the configured minimum score."""
        threshold = self._config.min_score_threshold
        if threshold <= 0:
            return results
        return [r for r in results if r.score >= threshold]
