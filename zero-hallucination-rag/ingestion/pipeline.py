"""
Main ingestion pipeline orchestrator.

Composes validation → cleaning → chunking → embedding → storage into a
single, configurable pipeline that can process documents in batches with
retry logic and observability hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.interfaces import VectorStore, LexicalStore

from core.config import IngestionConfig
from core.exceptions import (
    DocumentValidationError,
    EmbeddingError,
    IngestionError,
)
from ingestion.interfaces import (
    DocumentCleaner,
    DocumentChunker,
    DocumentStore,
    DocumentValidator,
    Embedder,
    EmbeddedChunk,
    MetadataExtractor,
    MetadataStore,
    ProcessedChunk,
    RawDocument,
)
from utils.helpers import batched, generate_document_id
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Outcome of ingesting a single document."""

    document_id: str
    source: str
    success: bool
    chunks_created: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchIngestionResult:
    """Aggregated outcome of a batch ingestion run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[IngestionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Parameters
    ----------
    config:
        Ingestion configuration (chunk sizes, batch size, etc.).
    validator:
        Document validation implementation.
    cleaner:
        Text cleaning implementation.
    chunker:
        Chunking strategy implementation.
    metadata_extractor:
        Metadata extraction implementation.
    embedder:
        Embedding model implementation.
    document_store:
        Storage backend for raw documents.
    metadata_store:
        Storage backend for extracted metadata.
    """

    def __init__(
        self,
        config: IngestionConfig,
        validator: DocumentValidator,
        cleaner: DocumentCleaner,
        chunker: DocumentChunker,
        metadata_extractor: MetadataExtractor,
        embedder: Embedder,
        document_store: Optional[DocumentStore] = None,
        metadata_store: Optional[MetadataStore] = None,
        vector_store: Optional[VectorStore] = None,
        lexical_store: Optional[LexicalStore] = None,
    ) -> None:
        self._config = config
        self._validator = validator
        self._cleaner = cleaner
        self._chunker = chunker
        self._metadata_extractor = metadata_extractor
        self._embedder = embedder
        self._document_store = document_store
        self._metadata_store = metadata_store
        self._vector_store = vector_store
        self._lexical_store = lexical_store

    # -- Public API -------------------------------------------------------

    async def ingest(self, document: RawDocument) -> IngestionResult:
        """
        Ingest a single document through the full pipeline.

        Stages
        ------
        1. Validate
        2. Extract metadata
        3. Clean text
        4. Chunk
        5. Embed chunks
        6. Persist document + metadata

        Returns
        -------
        IngestionResult
            Outcome including the generated ``document_id`` and chunk count.
        """
        doc_id = generate_document_id(
            document.content if isinstance(document.content, str) else document.content.decode("utf-8", errors="replace"),
            document.source,
        )
        try:
            # 1. Validate
            if not self._validator.validate(document):
                errors = self._validator.validation_errors(document)
                raise DocumentValidationError(
                    f"Document failed validation: {'; '.join(errors)}"
                )

            # 2. Extract metadata
            metadata = self._metadata_extractor.extract(document)
            metadata.update(document.metadata)
            
            # Inject trust_rating and created_at
            from datetime import datetime, timezone
            metadata["trust_rating"] = getattr(document, "trust_rating", 1.0)
            created_at_val = getattr(document, "created_at", None)
            if not created_at_val:
                created_at_val = datetime.now(timezone.utc).isoformat()
            metadata["created_at"] = created_at_val

            # 3. Clean
            text = (
                document.content
                if isinstance(document.content, str)
                else document.content.decode("utf-8", errors="replace")
            )
            cleaned = self._cleaner.clean(text)

            # 4. Chunk
            chunks: list[ProcessedChunk] = self._chunker.chunk(
                cleaned, document_id=doc_id, metadata=metadata
            )

            # 5. Embed
            embedded_chunks = self._embed_chunks(chunks)

            # 6. Persist
            if self._document_store:
                await self._document_store.store(doc_id, document)
            if self._metadata_store:
                await self._metadata_store.store(doc_id, metadata)

            # 7. Index in Vector & Lexical Stores if available
            if self._vector_store or self._lexical_store:
                chunk_texts = [c.chunk.text for c in embedded_chunks]
                chunk_ids = [c.chunk.chunk_id for c in embedded_chunks]
                payloads = [dict(c.chunk.metadata) for c in embedded_chunks]

                # Ensure payloads have 'text' and 'document_id' fields for search display
                for p, txt in zip(payloads, chunk_texts):
                    p["text"] = txt
                    p["document_id"] = doc_id

                # Generate matching UUID5 chunk IDs for both stores so they can be merged by RRF
                import uuid
                qdrant_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) for cid in chunk_ids]

                if self._vector_store:
                    vectors = [c.embedding for c in embedded_chunks]
                    from core.config import DocumentType
                    if document.document_type == DocumentType.JOB_DESCRIPTION:
                        await self._vector_store.upsert_jds(qdrant_ids, vectors, payloads)
                    else:
                        await self._vector_store.upsert_resumes(qdrant_ids, vectors, payloads)

                if self._lexical_store:
                    await self._lexical_store.index(qdrant_ids, chunk_texts, payloads)

            logger.info(
                "Ingested document",
                document_id=doc_id,
                source=document.source,
                chunks=len(embedded_chunks),
            )
            return IngestionResult(
                document_id=doc_id,
                source=document.source,
                success=True,
                chunks_created=len(embedded_chunks),
                metadata=metadata,
            )

        except IngestionError:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected ingestion error",
                document_id=doc_id,
                error=str(exc),
            )
            return IngestionResult(
                document_id=doc_id,
                source=document.source,
                success=False,
                error=str(exc),
            )

    async def ingest_batch(
        self, documents: Sequence[RawDocument]
    ) -> BatchIngestionResult:
        """
        Ingest a batch of documents, collecting per-document results.

        Documents are processed individually; a failure in one document
        does not abort the rest of the batch.
        """
        result = BatchIngestionResult(total=len(documents))
        for doc in documents:
            doc_result = await self.ingest(doc)
            result.results.append(doc_result)
            if doc_result.success:
                result.succeeded += 1
            else:
                result.failed += 1

        logger.info(
            "Batch ingestion complete",
            total=result.total,
            succeeded=result.succeeded,
            failed=result.failed,
        )
        return result

    # -- Private helpers --------------------------------------------------

    def _embed_chunks(self, chunks: list[ProcessedChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for a list of chunks in batches."""
        embedded: list[EmbeddedChunk] = []
        for batch in batched(chunks, self._config.embedding.batch_size):
            texts = [c.text for c in batch]
            try:
                vectors = self._embedder.embed(texts)
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding generation failed: {exc}"
                ) from exc

            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(chunk=chunk, embedding=vector))
        return embedded
