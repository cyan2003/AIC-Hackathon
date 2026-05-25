"""
System-level flow tests for the Zero-Hallucination RAG pipeline.
"""
import pytest
from qdrant_client import QdrantClient
from core.config import (
    get_settings,
    IngestionConfig,
    RetrievalConfig,
    ChunkingConfig,
    ChunkingStrategy,
    LexicalStoreConfig,
    FusionConfig,
)
from core.config import DocumentType
from ingestion.interfaces import RawDocument, Embedder
from ingestion.processors.cleaner import DefaultTextCleaner
from ingestion.processors.validator import DefaultDocumentValidator
from ingestion.processors.chunker import create_chunker
from ingestion.processors.jd_metadata_extractor import JDMetadataExtractor
from ingestion.processors.resume_metadata_extractor import ResumeMetadataExtractor
from ingestion.storage.document_store import InMemoryDocumentStore
from ingestion.storage.metadata_store import InMemoryMetadataStore
from ingestion.pipeline import IngestionPipeline
from retrieval.vector_store.qdrant import QdrantVectorStore
from retrieval.lexical_store.bm25 import BM25LexicalStore
from retrieval.fusion.reciprocal_rank import ReciprocalRankFusion
from retrieval.query.processor import DefaultQueryProcessor
from retrieval.pipeline import RetrievalPipeline


class SemanticKeywordEmbedder(Embedder):
    """
    Keyword-aware semantic mock embedder.
    Assigns higher similarity scores based on text matching.
    """
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def _get_vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text_lower = text.lower()
        if "python" in text_lower or "backend" in text_lower:
            vec[0] = 1.0
        elif "react" in text_lower or "frontend" in text_lower:
            vec[1] = 1.0
        elif "data" in text_lower or "pytorch" in text_lower:
            vec[2] = 1.0
        else:
            vec[3] = 1.0
        return vec

    def embed(self, texts):
        return [self._get_vector(t) for t in texts]

    def embed_single(self, text):
        return self._get_vector(text)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.mark.asyncio
async def test_end_to_end_system_matching_flow():
    """
    Test the full system flow:
    1. Ingest multiple job descriptions.
    2. Ingest a candidate resume.
    3. Run match retrieval querying resume → JDs.
    4. Assert correct Job Description matches with highest scores.
    """
    settings = get_settings()
    
    # 1. Setup real in-memory Qdrant client
    memory_qdrant_client = QdrantClient(location=":memory:")
    
    vector_store = QdrantVectorStore(
        settings.retrieval.vector_store,
        embedding_dimension=384,
    )
    vector_store._client = memory_qdrant_client
    await vector_store.initialize()
    
    # 2. Setup shared In-Memory Storage & BM25 Lexical Store
    doc_store = InMemoryDocumentStore()
    meta_store = InMemoryMetadataStore()
    
    lexical_store = BM25LexicalStore(
        LexicalStoreConfig(k1=1.5, b=0.75, max_index_size=100)
    )
    
    embedder = SemanticKeywordEmbedder()
    
    # 3. Setup Pipelines
    # Ingestion Pipeline for JDs
    jd_pipeline = IngestionPipeline(
        config=settings.ingestion,
        validator=DefaultDocumentValidator(settings.ingestion),
        cleaner=DefaultTextCleaner(),
        chunker=create_chunker(ChunkingConfig(strategy=ChunkingStrategy.SECTION_AWARE)),
        metadata_extractor=JDMetadataExtractor(),
        embedder=embedder,
        document_store=doc_store,
        metadata_store=meta_store,
    )
    
    # 4. Ingest 3 Job Descriptions (JDs)
    jds = [
        RawDocument(
            source="python_eng.pdf",
            content=(
                "Senior Python Developer\n"
                "RESPONSIBILITIES\n"
                "Build backend APIs using Python and FastAPI.\n"
                "REQUIREMENTS\n"
                "Expertise in Python, REST APIs, and database engineering."
            ),
            mime_type="application/pdf",
            document_type=DocumentType.JOB_DESCRIPTION,
        ),
        RawDocument(
            source="react_eng.docx",
            content=(
                "React Frontend Engineer\n"
                "RESPONSIBILITIES\n"
                "Develop responsive web interfaces with React and Tailwind.\n"
                "REQUIREMENTS\n"
                "Experience with Javascript, HTML, CSS, React, and layout design."
            ),
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            document_type=DocumentType.JOB_DESCRIPTION,
        ),
        RawDocument(
            source="data_sci.pdf",
            content=(
                "Machine Learning Scientist\n"
                "RESPONSIBILITIES\n"
                "Train deep learning models using PyTorch.\n"
                "REQUIREMENTS\n"
                "PhD in computer science with machine learning experience."
            ),
            mime_type="application/pdf",
            document_type=DocumentType.JOB_DESCRIPTION,
        ),
    ]
    
    # Run JD ingestion & also add to Vector and Lexical Indexes manually or verify
    for jd_doc in jds:
        ingest_res = await jd_pipeline.ingest(jd_doc)
        assert ingest_res.success is True
        
        # Segment-level chunks index
        text = jd_doc.content
        chunks = jd_pipeline._chunker.chunk(text, document_id=ingest_res.document_id, metadata=ingest_res.metadata)
        chunk_texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        import uuid
        qdrant_chunk_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) for cid in chunk_ids]
        vectors = embedder.embed(chunk_texts)
        payloads = [c.metadata for c in chunks]
        for p, txt in zip(payloads, chunk_texts):
            p["text"] = txt
            p["document_id"] = ingest_res.document_id
            
        await vector_store.upsert_jds(qdrant_chunk_ids, vectors, payloads)
        await lexical_store.index(chunk_ids, chunk_texts, payloads)

    # 5. Ingest a Resume
    resume_pipeline = IngestionPipeline(
        config=settings.ingestion,
        validator=DefaultDocumentValidator(settings.ingestion),
        cleaner=DefaultTextCleaner(),
        chunker=create_chunker(ChunkingConfig(strategy=ChunkingStrategy.SECTION_AWARE)),
        metadata_extractor=ResumeMetadataExtractor(),
        embedder=embedder,
        document_store=doc_store,
        metadata_store=meta_store,
    )
    
    resume_doc = RawDocument(
        source="bob_resume.pdf",
        content=(
            "Bob Vance\nbob@vance.com\n"
            "TECHNICAL SKILLS\n"
            "Python, FastAPI, Backend, PostgreSQL\n"
            "WORK EXPERIENCE\n"
            "Python Software Engineer at cooling corp. Designed backend APIs."
        ),
        mime_type="application/pdf",
        document_type=DocumentType.RESUME,
    )
    
    resume_ingest_res = await resume_pipeline.ingest(resume_doc)
    assert resume_ingest_res.success is True
    
    # 6. Perform Match Retrieval (Resume → JDs)
    retrieval_pipeline = RetrievalPipeline(
        config=settings.retrieval,
        vector_store=vector_store,
        lexical_store=lexical_store,
        embedder=embedder,
        fuser=ReciprocalRankFusion(FusionConfig(rrf_k=60)),
        query_processor=DefaultQueryProcessor(),
        reranker=None,
    )
    
    # Query with resume text
    response = await retrieval_pipeline.retrieve(
        query_text=resume_doc.content,
        top_k=3,
    )
    
    assert response is not None
    assert len(response.results) > 0
    
    # Python job should rank first because the embedder assigns vec[0]=1.0
    # to both resume and python job requirements, and lexical search matches 'python' / 'backend'
    top_result = response.results[0]
    # Find matching JD
    top_jd_meta = await meta_store.get(top_result.document_id)
    assert top_jd_meta is not None
    assert "Python" in top_jd_meta["job_title"]
    
    # Clean resources
    await vector_store.close()
