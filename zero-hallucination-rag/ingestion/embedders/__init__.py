"""
Embedders sub-package.
"""

from ingestion.embedders.factory import create_embedder
from ingestion.embedders.huggingface import HuggingFaceEmbedder

__all__ = ["create_embedder", "HuggingFaceEmbedder"]
