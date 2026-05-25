"""
Reciprocal Rank Fusion (RRF).

Merges multiple ranked result lists by summing reciprocal ranks,
a simple and robust approach that does not require score normalisation
across backends.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods" (2009).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from core.config import FusionConfig
from retrieval.interfaces import ResultFuser, SearchResult


class ReciprocalRankFusion(ResultFuser):
    """
    Reciprocal Rank Fusion with configurable *k* parameter.

    The fused score for a document appearing at rank *r* in list *i* is::

        score += 1 / (k + r)

    Higher *k* values dampen the influence of top-ranked documents.

    Parameters
    ----------
    config:
        Fusion configuration containing ``rrf_k``.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._k = config.rrf_k

    def fuse(
        self,
        result_lists: Sequence[list[SearchResult]],
    ) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, SearchResult] = {}

        for rlist in result_lists:
            for rank, result in enumerate(rlist, start=1):
                key = result.chunk_id
                scores[key] += 1.0 / (self._k + rank)

                # Keep the result instance with the highest original score.
                if key not in best_result or result.score > best_result[key].score:
                    best_result[key] = result

        # Build fused list sorted by RRF score descending.
        fused: list[SearchResult] = []
        for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = best_result[key]
            fused.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=rrf_score,
                    metadata=result.metadata,
                    source=f"rrf({result.source})",
                )
            )

        return fused
