"""
Lexical store factory.

Returns the appropriate :class:`~retrieval.interfaces.LexicalStore`
implementation based on configuration.
"""

from __future__ import annotations

from core.config import LexicalStoreBackend, LexicalStoreConfig
from retrieval.interfaces import LexicalStore


def create_lexical_store(config: LexicalStoreConfig) -> LexicalStore:
    """
    Instantiate a lexical store matching ``config.backend``.

    Raises
    ------
    ValueError
        If the configured backend is not supported.
    """
    if config.backend == LexicalStoreBackend.BM25:
        from retrieval.lexical_store.bm25 import BM25LexicalStore

        return BM25LexicalStore(config)

    raise ValueError(f"Unsupported lexical store backend: {config.backend}")
