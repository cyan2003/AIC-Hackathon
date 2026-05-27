"""
BM25 lexical search implementation.

Uses the ``rank-bm25`` library to provide keyword-based retrieval.
The index is held entirely in memory, which is practical for up to
a few million short documents; larger corpora should consider an
Elasticsearch or Lucene-backed alternative.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from core.config import LexicalStoreConfig
from core.exceptions import LexicalStoreError
from retrieval.interfaces import LexicalStore, SearchResult
from utils.logging import get_logger

logger = get_logger(__name__)


class BM25LexicalStore(LexicalStore):
    """
    In-memory BM25-backed lexical store.

    Parameters
    ----------
    config:
        Lexical store configuration (k1, b, max_index_size).
    """

    def __init__(self, config: LexicalStoreConfig) -> None:
        self._config = config
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None  # Rebuilt on index changes
        self._dirty = True
        self._lock = asyncio.Lock() # Ensure thread safety during rebuilds

    # -- Index management -------------------------------------------------

    async def index(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        if len(self._ids) + len(ids) > self._config.max_index_size:
            raise LexicalStoreError(
                f"Index size would exceed maximum ({self._config.max_index_size:,})"
            )

        # Offload CPU-bound tokenization to a background thread
        def _sync_index():
            for i, (doc_id, text) in enumerate(zip(ids, texts)):
                self._ids.append(doc_id)
                self._texts.append(text)
                self._payloads.append(payloads[i] if payloads else {})
                self._tokenized_corpus.append(self._tokenize(text))
            self._dirty = True

        async with self._lock:
            await asyncio.to_thread(_sync_index)
            
        logger.debug("Indexed documents", count=len(ids))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        
        async with self._lock:
            # Rebuild index in a thread if necessary
            if self._dirty:
                await asyncio.to_thread(self._rebuild_sync)

        if self._bm25 is None or not self._ids:
            return []

        # Offload the scoring and filtering process
        def _sync_search() -> list[SearchResult]:
            try:
                tokenized_query = self._tokenize(query)
                scores = self._bm25.get_scores(tokenized_query)

                # Pair scores with indices and sort descending.
                scored_indices = sorted(
                    enumerate(scores), key=lambda x: x[1], reverse=True
                )

                results: list[SearchResult] = []
                for idx, score in scored_indices:
                    if score <= 0:
                        continue
                        
                    payload = self._payloads[idx]

                    # Apply metadata filters BEFORE counting towards top_k
                    if filters and not self._matches_filter(payload, filters):
                        continue

                    results.append(
                        SearchResult(
                            document_id=payload.get("document_id", ""),
                            chunk_id=self._ids[idx],
                            text=self._texts[idx],
                            score=float(score),
                            metadata=payload,
                            source="lexical",
                        )
                    )
                    
                    # Stop once we have accumulated exactly top_k valid matches
                    if len(results) >= top_k:
                        break

                return results
            except Exception as exc:
                raise LexicalStoreError(f"BM25 search failed: {exc}") from exc

        return await asyncio.to_thread(_sync_search)

    async def delete(self, ids: Sequence[str]) -> None:
        def _sync_delete():
            id_set = set(ids)
            indices_to_keep = [
                i for i, doc_id in enumerate(self._ids) if doc_id not in id_set
            ]
            self._ids = [self._ids[i] for i in indices_to_keep]
            self._texts = [self._texts[i] for i in indices_to_keep]
            self._payloads = [self._payloads[i] for i in indices_to_keep]
            self._tokenized_corpus = [self._tokenized_corpus[i] for i in indices_to_keep]
            self._dirty = True

        async with self._lock:
            await asyncio.to_thread(_sync_delete)
            
        logger.debug("Deleted from BM25 index", count=len(ids))

    async def count(self) -> int:
        return len(self._ids)

    # -- Internals --------------------------------------------------------

    def _rebuild_sync(self) -> None:
        """Synchronous rebuild logic to be run in a thread."""
        if not self._dirty:
            return
        try:
            from rank_bm25 import BM25Okapi

            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(
                    self._tokenized_corpus,
                    k1=self._config.k1,
                    b=self._config.b,
                )
            else:
                self._bm25 = None
            self._dirty = False
        except ImportError as exc:
            raise LexicalStoreError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            ) from exc

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()

    @staticmethod
    def _matches_filter(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Return ``True`` if the payload satisfies all filters."""
        # 1. Skills filter: JD's required_skills must overlap with candidate's skills if filtered
        if "skills" in filters and filters["skills"]:
            filter_skills = {s.lower() for s in filters["skills"]}
            # required_skills is a list in JD metadata
            jd_skills = {s.lower() for s in payload.get("required_skills", [])}
            # Match if there's any overlap between candidate skills and required skills
            if not filter_skills.intersection(jd_skills):
                return False

        # 2. Experience filter: JD's required experience must be <= candidate's max experience
        if "experience_years_max" in filters and filters["experience_years_max"] is not None:
            candidate_exp = filters["experience_years_max"]
            jd_exp = payload.get("experience_years")
            if jd_exp is not None and jd_exp > candidate_exp:
                return False

        # 3. Education filter: JD's required education must be <= candidate's education level
        if "education_level" in filters and filters["education_level"]:
            EDUCATION_ORDER = {"associate": 1, "bachelor": 2, "master": 3, "phd": 4}
            candidate_edu = filters["education_level"].lower()
            jd_edu = payload.get("education_level")
            if jd_edu and candidate_edu in EDUCATION_ORDER and jd_edu.lower() in EDUCATION_ORDER:
                if EDUCATION_ORDER[jd_edu.lower()] > EDUCATION_ORDER[candidate_edu]:
                    return False

        # 4. Industry filter: exact match (case-insensitive)
        if "industry" in filters and filters["industry"]:
            candidate_ind = filters["industry"].lower()
            jd_ind = payload.get("industry")
            if not jd_ind or jd_ind.lower() != candidate_ind:
                return False

        # 5. Fallback check for any other basic filters
        for k, v in filters.items():
            if k not in ["skills", "experience_years_max", "education_level", "industry"]:
                if payload.get(k) != v:
                    return False

        return True