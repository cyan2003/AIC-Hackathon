"""
BM25 lexical search implementation.

Uses the ``rank-bm25`` library to provide keyword-based retrieval.
The index is held entirely in memory, which is practical for up to
a few million short documents; larger corpora should consider an
Elasticsearch or Lucene-backed alternative.
"""

from __future__ import annotations

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

        for i, (doc_id, text) in enumerate(zip(ids, texts)):
            self._ids.append(doc_id)
            self._texts.append(text)
            self._payloads.append(payloads[i] if payloads else {})
            self._tokenized_corpus.append(self._tokenize(text))

        self._dirty = True
        logger.debug("Indexed documents", count=len(ids))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        self._rebuild_if_dirty()
        if self._bm25 is None or not self._ids:
            return []

        try:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)

            # Pair scores with indices and sort descending.
            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )

            results: list[SearchResult] = []
            for idx, score in scored_indices[:top_k]:
                if score <= 0:
                    continue
                payload = self._payloads[idx]

                # Apply metadata filters if provided.
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

            return results
        except Exception as exc:
            raise LexicalStoreError(f"BM25 search failed: {exc}") from exc

    async def delete(self, ids: Sequence[str]) -> None:
        id_set = set(ids)
        indices_to_keep = [
            i for i, doc_id in enumerate(self._ids) if doc_id not in id_set
        ]
        self._ids = [self._ids[i] for i in indices_to_keep]
        self._texts = [self._texts[i] for i in indices_to_keep]
        self._payloads = [self._payloads[i] for i in indices_to_keep]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in indices_to_keep]
        self._dirty = True
        logger.debug("Deleted from BM25 index", count=len(ids))

    async def count(self) -> int:
        return len(self._ids)

    # -- Internals --------------------------------------------------------

    def _rebuild_if_dirty(self) -> None:
        """Re-build the BM25 model when the corpus has changed."""
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
        """Return ``True`` if all filter key-values match the payload."""
        return all(payload.get(k) == v for k, v in filters.items())
