"""
Abstract vector store base.

Re-exports the :class:`~retrieval.interfaces.VectorStore` ABC.
"""

from retrieval.interfaces import VectorStore as BaseVectorStore

__all__ = ["BaseVectorStore"]
