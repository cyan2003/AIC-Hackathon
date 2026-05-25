"""
Integration tests for the FastAPI service endpoints.
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
from api.main import app
from ingestion.pipeline import IngestionResult
from retrieval.interfaces import SearchResult


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns API metadata."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["service"] == "zero-hallucination-rag"
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test standard health check liveness probe."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok", "service": "zero-hallucination-rag"}


def test_metrics_endpoint(client):
    """Test metrics endpoint returns Prometheus data or custom message."""
    response = client.get("/metrics")
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_501_NOT_IMPLEMENTED]


@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
def test_readiness_endpoint(mock_qdrant_cls, client):
    """Test readiness check mocks vector store connectivity."""
    from unittest.mock import AsyncMock
    mock_store = MagicMock()
    mock_store.count = AsyncMock(side_effect=[15, 25])  # count for resumes, then jds
    mock_store.close = AsyncMock()
    mock_qdrant_cls.return_value = mock_store
    
    response = client.get("/health/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["vector_store"] == "connected"
    assert data["collections"]["resumes"]["count"] == 15
    assert data["collections"]["job_descriptions"]["count"] == 25


@patch("api.routers.ingestion._extract_text")
@patch("ingestion.pipeline.IngestionPipeline.ingest")
def test_ingest_resume(mock_ingest, mock_extract, client):
    """Test uploading a resume correctly triggers parsing and pipeline."""
    mock_extract.return_value = "Mocked parsed resume content"
    mock_ingest.return_value = IngestionResult(
        document_id="doc_res_123",
        source="resume.pdf",
        success=True,
        chunks_created=3,
        metadata={"candidate_name": "John Doe", "skills": ["Python"]},
    )
    
    # Upload fake pdf file bytes
    file_data = {"file": ("resume.pdf", io.BytesIO(b"dummy pdf bytes"), "application/pdf")}
    response = client.post("/ingest/resume", files=file_data)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["document_id"] == "doc_res_123"
    assert data["chunks_created"] == 3
    assert data["metadata"]["candidate_name"] == "John Doe"
    assert data["message"] == "Resume ingested successfully"


@patch("api.routers.ingestion._extract_text")
@patch("ingestion.pipeline.IngestionPipeline.ingest")
def test_ingest_jd(mock_ingest, mock_extract, client):
    """Test uploading a job description triggers parsing and pipeline."""
    mock_extract.return_value = "Mocked parsed JD content"
    mock_ingest.return_value = IngestionResult(
        document_id="doc_jd_456",
        source="job.docx",
        success=True,
        chunks_created=2,
        metadata={"job_title": "Python Developer"},
    )
    
    file_data = {"file": ("job.docx", io.BytesIO(b"dummy docx bytes"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/ingest/jd", files=file_data)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["document_id"] == "doc_jd_456"
    assert data["chunks_created"] == 2
    assert data["metadata"]["job_title"] == "Python Developer"
    assert data["message"] == "Job description ingested successfully"


@patch("ingestion.embedders.factory.create_embedder")
@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
@patch("retrieval.lexical_store.bm25.BM25LexicalStore")
@patch("retrieval.fusion.reranker.CrossEncoderReranker")
def test_match_endpoint(mock_reranker_cls, mock_bm25_cls, mock_qdrant_cls, mock_embedder_factory, client):
    """Test matching endpoint executes correctly with mocked retrieval pipeline."""
    from unittest.mock import AsyncMock
    # 1. Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.1] * 384
    mock_embedder_factory.return_value = mock_embedder
    
    # 2. Mock Qdrant
    mock_qdrant = MagicMock()
    mock_qdrant.search_jds = AsyncMock(return_value=[
        SearchResult(
            document_id="jd_1",
            chunk_id="chunk_1",
            text="Looking for Python backend developer.",
            score=0.9,
            metadata={"job_title": "Python Dev", "chunk_section": "requirements"},
        )
    ])
    mock_qdrant_cls.return_value = mock_qdrant
    
    # 3. Mock BM25
    mock_bm25 = MagicMock()
    mock_bm25.search = AsyncMock(return_value=[])
    mock_bm25_cls.return_value = mock_bm25
    
    # 4. Mock CrossEncoderReranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, results, top_k: results[:top_k]
    mock_reranker_cls.return_value = mock_reranker
    
    response = client.post(
        "/match",
        data={"resume_text": "Experienced Python developer skilled in SQL.", "top_k": 2},
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["document_id"] == "jd_1"
    assert data["results"][0]["job_title"] == "Python Dev"
    assert len(data["results"][0]["matched_sections"]) == 1
    assert data["results"][0]["matched_sections"][0]["section"] == "requirements"
    assert data["resume_metadata"]["candidate_name"] == "Experienced Python developer skilled in SQL."
