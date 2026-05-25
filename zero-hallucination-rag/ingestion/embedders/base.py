"""
Abstract embedder base.

Re-exports the :class:`~ingestion.interfaces.Embedder` ABC for
convenience — all concrete embedder implementations should sub-class it.
"""

from ingestion.interfaces import Embedder as BaseEmbedder

__all__ = ["BaseEmbedder"]
