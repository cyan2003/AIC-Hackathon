"""
Weighted score fusion.

Merges results by normalising scores within each list to [0, 1] and
computing a weighted sum. This requires that raw scores are meaningful
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
        self._config = config

    def fuse(
        self,
        result_lists: Sequence[list[SearchResult]],
    ) -> list[SearchResult]:
        """
        Fuses results using a weighted linear combination.
        
        Parameters
        ----------
        result_lists:
            A sequence of ranked result lists.
        """
        scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, SearchResult] = {}
        sources_tracker: dict[str, set[str]] = defaultdict(set)

        # Explicitly map backends to their respective configured weights
        weight_map = {
            "vector": self._config.vector_weight,
            "lexical": self._config.lexical_weight
        }

        lists_to_fuse = list(result_lists.values()) if isinstance(result_lists, dict) else result_lists

        for rlist in lists_to_fuse:
            if not rlist:
                continue
            backend_name = rlist[0].source or "vector"
            # Default to a weight of 1.0 if a new backend is added later
            weight = weight_map.get(backend_name, 1.0) 
            normalised_list = self._normalise(rlist)

            for result in normalised_list:
                key = result.chunk_id
                
                # 1. Accumulate Weighted Score
                scores[key] += weight * result.score
                
                # 2. Track all backends that found this chunk
                sources_tracker[key].add(result.source or backend_name)

                # 3. Store the first encountered result instance to preserve metadata
                if key not in best_result:
                    best_result[key] = result

        fused: list[SearchResult] = []
        for key, weighted_score in sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        ):
            result = best_result[key]
            combined_sources = "+".join(sorted(sources_tracker[key]))
            
            fused.append(
                SearchResult(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=weighted_score,
                    metadata=result.metadata,
                    source=f"weighted({combined_sources})",
                )
            )

        return fused

    @staticmethod
    def _normalise(results: list[SearchResult]) -> list[SearchResult]:
        """Min-max normalise scores to [0, 1]."""
        if not results:
            return results
        
        raw_scores = [r.score for r in results]
        min_s, max_s = min(raw_scores), max(raw_scores)
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