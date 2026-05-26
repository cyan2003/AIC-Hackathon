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
        fused = self._fuser.fuse({"vector": vector_results, "lexical": lexical_results})
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
        """
        Calculate confidence scores and filter results below the configured 
        minimum score and minimum confidence threshold.
        """
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc)

        # 1. Apply min_score_threshold if configured
        score_threshold = self._config.min_score_threshold
        filtered_by_score = results
        if score_threshold > 0:
            filtered_by_score = [r for r in results if r.score >= score_threshold]

        final_results = []
        for r in filtered_by_score:
            # Normalize base similarity score
            normalized_score = max(0.0, min(1.0, r.score))

            # Freshness score
            created_at_str = r.metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                except Exception:
                    created_at = current_time
            else:
                created_at = current_time

            delta = current_time - created_at
            days_since_creation = max(0.0, delta.total_seconds() / 86400.0)
            max_days = self._config.max_freshness_days
            freshness_score = max(0.0, 1.0 - (days_since_creation / max_days))

            # Trust score
            try:
                trust_rating = float(r.metadata.get("trust_rating", 1.0))
            except (ValueError, TypeError):
                trust_rating = 1.0
            trust_rating = max(0.0, min(1.0, trust_rating))

            # Combined confidence score
            w_sim = self._config.weight_similarity
            w_fresh = self._config.weight_freshness
            w_trust = self._config.weight_trust

            confidence = (w_sim * normalized_score) + (w_fresh * freshness_score) + (w_trust * trust_rating)
            r.confidence_score = float(round(confidence, 4))

            # Filter by confidence threshold
            if r.confidence_score >= self._config.min_confidence_threshold:
                final_results.append(r)

        return final_results
