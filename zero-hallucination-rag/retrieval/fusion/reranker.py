"""
Cross-encoder re-ranking.

Uses a HuggingFace cross-encoder model to re-score candidate results
against the original query for higher-precision ranking.
"""

from __future__ import annotations

from core.exceptions import RetrievalError
from retrieval.interfaces import Reranker, SearchResult
from utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """
    Re-ranker backed by a HuggingFace ``CrossEncoder`` model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    device:
        Torch device (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None  # Lazy-loaded

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model", model=self._model_name)
            self._model = CrossEncoder(self._model_name, device=self._device)
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers is required for cross-encoder re-ranking. "
                "Install with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load cross-encoder model '{self._model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return results

        self._load_model()

        # Build query-document pairs.
        pairs = [(query, r.text) for r in results]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RetrievalError(f"Cross-encoder prediction failed: {exc}") from exc

        # Attach reranker scores and sort.
        reranked: list[SearchResult] = []
        for result, score in zip(results, scores):
            reranked.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    metadata={**result.metadata, "original_score": result.score},
                    source=result.source,
                )
            )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
