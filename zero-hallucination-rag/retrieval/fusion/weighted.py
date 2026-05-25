"""
Weighted score fusion.

Merges results by normalising scores within each list to [0, 1] and
computing a weighted sum.  This requires that raw scores are meaningful
(i.e. higher = better) within each backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from core.config import FusionConfig
from retrieval.interfaces import ResultFuser, SearchResult


class WeightedScoreFusion(ResultFuser):
    """
    Weighted linear combination of normalised scores.

    Parameters
    ----------
    config:
        Fusion configuration containing ``vector_weight`` and ``lexical_weight``.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._weights = [config.vector_weight, config.lexical_weight]

    def fuse(
        self,
        result_lists: Sequence[list[SearchResult]],
    ) -> list[SearchResult]:
        # Normalise each list's scores to [0, 1].
        normalised_lists = [self._normalise(rlist) for rlist in result_lists]

        scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, SearchResult] = {}

        for i, rlist in enumerate(normalised_lists):
            weight = self._weights[i] if i < len(self._weights) else 1.0
            for result in rlist:
                key = result.chunk_id
                scores[key] += weight * result.score
                if key not in best_result or result.score > best_result[key].score:
                    best_result[key] = result

        fused: list[SearchResult] = []
        for key, weighted_score in sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        ):
            result = best_result[key]
            fused.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=weighted_score,
                    metadata=result.metadata,
                    source=f"weighted({result.source})",
                )
            )

        return fused

    @staticmethod
    def _normalise(results: list[SearchResult]) -> list[SearchResult]:
        """Min-max normalise scores to [0, 1]."""
        if not results:
            return results
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s
        if spread == 0:
            return [
                SearchResult(
                    document_id=r.document_id,
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=1.0,
                    metadata=r.metadata,
                    source=r.source,
                )
                for r in results
            ]
        return [
            SearchResult(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                text=r.text,
                score=(r.score - min_s) / spread,
                metadata=r.metadata,
                source=r.source,
            )
            for r in results
        ]
