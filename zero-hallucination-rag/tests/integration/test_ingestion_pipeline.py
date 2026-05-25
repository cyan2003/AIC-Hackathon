"""
Integration tests for the ingestion pipeline module.
"""
import pytest
from core.config import IngestionConfig, ChunkingConfig, ChunkingStrategy
from ingestion.interfaces import RawDocument, Embedder
from ingestion.processors.cleaner import DefaultTextCleaner
from ingestion.processors.validator import DefaultDocumentValidator
from ingestion.processors.chunker import create_chunker
from ingestion.processors.resume_metadata_extractor import ResumeMetadataExtractor
from ingestion.storage.document_store import InMemoryDocumentStore
from ingestion.storage.metadata_store import InMemoryMetadataStore
from ingestion.pipeline import IngestionPipeline


class DummyEmbedder(Embedder):
    """Simple embedder returning dummy vectors for integration testing."""
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, texts):
        return [[0.1] * self._dim for _ in texts]

    def embed_single(self, text):
        return [0.1] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.mark.asyncio
async def test_ingestion_pipeline_single_resume():
    """Test full ingestion pipeline run with a single resume."""
    config = IngestionConfig(
        max_document_size_mb=1,
        supported_extensions=[".pdf", ".docx"],
    )
    
    # Components
    validator = DefaultDocumentValidator(config)
    cleaner = DefaultTextCleaner(lowercase=False)
    
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=100,
        chunk_overlap=0,
        min_chunk_size=5,
    )
    chunker = create_chunker(chunking_config)
    
    metadata_extractor = ResumeMetadataExtractor()
    embedder = DummyEmbedder()
    
    doc_store = InMemoryDocumentStore()
    meta_store = InMemoryMetadataStore()
    
    # Pipeline
    pipeline = IngestionPipeline(
        config=config,
        validator=validator,
        cleaner=cleaner,
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedder=embedder,
        document_store=doc_store,
        metadata_store=meta_store,
    )
    
    raw_resume = (
        "Alice Smith\nalice.smith@example.com\n"
        "TECHNICAL SKILLS\n"
        "Python, SQL, AWS, Docker\n"
        "EXPERIENCE\n"
        "Senior Backend Engineer with 5+ years of experience.\n"
    )
    doc = RawDocument(
        source="alice_resume.pdf",
        content=raw_resume,
        mime_type="application/pdf",
        metadata={"custom_flag": "test-run"},
    )
    
    result = await pipeline.ingest(doc)
    
    assert result.success is True
    assert result.document_id != ""
    assert result.chunks_created > 0
    
    # Check persistence
    assert doc_store.count == 1
    assert await doc_store.exists(result.document_id) is True
    
    stored_doc = await doc_store.get(result.document_id)
    assert stored_doc.source == "alice_resume.pdf"
    
    assert meta_store.count == 1
    stored_meta = await meta_store.get(result.document_id)
    assert stored_meta["candidate_name"] == "Alice Smith"
    assert stored_meta["email"] == "alice.smith@example.com"
    assert stored_meta["experience_years"] == 5
    assert stored_meta["custom_flag"] == "test-run"


@pytest.mark.asyncio
async def test_ingestion_pipeline_batch():
    """Test full ingestion pipeline run with a batch of resumes."""
    config = IngestionConfig(
        max_document_size_mb=1,
        supported_extensions=[".pdf", ".docx"],
    )
    
    validator = DefaultDocumentValidator(config)
    cleaner = DefaultTextCleaner()
    chunker = create_chunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=100, chunk_overlap=0))
    metadata_extractor = ResumeMetadataExtractor()
    embedder = DummyEmbedder()
    doc_store = InMemoryDocumentStore()
    meta_store = InMemoryMetadataStore()
    
    pipeline = IngestionPipeline(
        config=config,
        validator=validator,
        cleaner=cleaner,
        chunker=chunker,
        metadata_extractor=metadata_extractor,
        embedder=embedder,
        document_store=doc_store,
        metadata_store=meta_store,
    )
    
    docs = [
        RawDocument(
            source="res1.pdf",
            content="Bob Johnson\nbob@johnson.com\nDeveloper with 3 years exp.",
            mime_type="application/pdf",
        ),
    ]
    
    batch_result = await pipeline.ingest_batch(docs)
    
    assert batch_result.total == 1
    assert batch_result.succeeded == 1
    assert batch_result.failed == 0
    assert batch_result.results[0].success is True
    
    # Test validation error on invalid extension raises DocumentValidationError
    bad_doc = RawDocument(
        source="res2.txt",  # Unsupported extension -> will fail validation
        content="Invalid format text",
        mime_type="application/pdf",
    )
    from core.exceptions import DocumentValidationError
    with pytest.raises(DocumentValidationError):
        await pipeline.ingest(bad_doc)

