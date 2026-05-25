"""
Query transformation for different search backends.

Adapts a :class:`~retrieval.interfaces.RetrievalQuery` into backend-specific
representations (e.g. boosted fields for lexical, filter translations for
vector stores).
"""

from __future__ import annotations

from retrieval.interfaces import QueryTransformer, RetrievalQuery


class DefaultQueryTransformer(QueryTransformer):
    """
    Pass-through transformer that returns the query unmodified.

    Backend-specific transformations (e.g. Qdrant filter syntax,
    Elasticsearch query DSL) should be added by sub-classing this
    transformer or replacing it entirely.
    """

    def for_vector(self, query: RetrievalQuery) -> RetrievalQuery:
        """No-op: the query is used as-is for vector search."""
        return query

    def for_lexical(self, query: RetrievalQuery) -> RetrievalQuery:
        """No-op: the query is used as-is for lexical search."""
        return query
