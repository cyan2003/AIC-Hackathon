"""
Storage sub-package.
"""

from ingestion.storage.document_store import InMemoryDocumentStore
from ingestion.storage.metadata_store import InMemoryMetadataStore

__all__ = ["InMemoryDocumentStore", "InMemoryMetadataStore"]
