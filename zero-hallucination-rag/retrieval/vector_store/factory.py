"""
Vector store factory.

Returns the appropriate :class:`~retrieval.interfaces.VectorStore`
implementation based on configuration.
"""

from __future__ import annotations

from core.config import VectorStoreBackend, VectorStoreConfig
from retrieval.interfaces import VectorStore


def create_vector_store(
    config: VectorStoreConfig,
    embedding_dimension: int,
) -> VectorStore:
    """
    Instantiate and return a vector store matching ``config.backend``.

    Parameters
    ----------
    config:
        Vector store configuration.
    embedding_dimension:
        Dimensionality of the vectors.

    Raises
    ------
    ValueError
        If the configured backend is not supported.
    """
    if config.backend == VectorStoreBackend.QDRANT:
        from retrieval.vector_store.qdrant import QdrantVectorStore

        return QdrantVectorStore(config, embedding_dimension)

    if config.backend == VectorStoreBackend.MILVUS:
        from retrieval.vector_store.milvus import MilvusVectorStore

        return MilvusVectorStore(config, embedding_dimension)

    raise ValueError(f"Unsupported vector store backend: {config.backend}")
