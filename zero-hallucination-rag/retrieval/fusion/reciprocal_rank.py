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
        results_map: dict[str, list[SearchResult]],
    ) -> list[SearchResult]:
        """
        Fuses results using RRF.
        
        Parameters
        ----------
        results_map:
            A dictionary mapping the backend name to its results.
            Example: {"vector": qdrant_results, "lexical": bm25_results}
        """
        scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, SearchResult] = {}
        sources_tracker: dict[str, set[str]] = defaultdict(set)

        for backend_name, rlist in results_map.items():
            for rank, result in enumerate(rlist, start=1):
                key = result.chunk_id
                
                # 1. Accumulate RRF Score
                scores[key] += 1.0 / (self._k + rank)
                
                # 2. Track all backends that found this chunk
                sources_tracker[key].add(result.source)

                # 3. Store the first encountered result instance to preserve metadata
                if key not in best_result:
                    best_result[key] = result

        # Build fused list sorted by RRF score descending.
        fused: list[SearchResult] = []
        for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = best_result[key]
            combined_sources = "+".join(sorted(sources_tracker[key]))
            
            fused.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=rrf_score,
                    metadata=result.metadata,
                    source=f"rrf({combined_sources})",
                )
            )

        return fused