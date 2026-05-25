"""
Embedder factory.

Returns the appropriate :class:`~ingestion.interfaces.Embedder`
implementation based on the application configuration.
"""

from __future__ import annotations

from core.config import EmbeddingConfig, EmbeddingProvider
from ingestion.interfaces import Embedder


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """
    Instantiate and return an embedder matching ``config.provider``.

    Raises
    ------
    ValueError
        If the configured provider is not supported.
    """
    if config.provider == EmbeddingProvider.HUGGINGFACE:
        from ingestion.embedders.huggingface import HuggingFaceEmbedder

        return HuggingFaceEmbedder(config)

    raise ValueError(f"Unsupported embedding provider: {config.provider}")
