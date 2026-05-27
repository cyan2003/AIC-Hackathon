"""
Integration tests to verify that ingestion endpoints index into the vector and lexical stores.
"""

import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
from api.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@patch("api.routers.ingestion._extract_text")
@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
@patch("retrieval.lexical_store.bm25.BM25LexicalStore")
def test_ingest_resume_indexing_calls(mock_bm25_cls, mock_qdrant_cls, mock_extract, client):
    """Verify that ingesting a resume indexes it in both vector and lexical stores."""
    # 1. Mock text extraction
    mock_extract.return_value = "Parsed resume text with skills like Python and FastAPI."

    # 2. Mock store instances
    mock_qdrant = MagicMock()
    mock_qdrant.upsert_resumes = AsyncMock()
    mock_qdrant_cls.return_value = mock_qdrant

    mock_bm25 = MagicMock()
    mock_bm25.index = AsyncMock()
    mock_bm25_cls.return_value = mock_bm25

    # 3. Post a resume upload
    file_data = {
        "file": ("resume.pdf", io.BytesIO(b"dummy pdf content"), "application/pdf")
    }
    data = {
        "trust_rating": 0.8,
        "created_at": "2026-05-26T12:00:00Z"
    }
    
    response = client.post("/ingest/resume", files=file_data, data=data)
    
    assert response.status_code == status.HTTP_201_CREATED
    response_data = response.json()
    assert response_data["document_id"] != ""
    assert response_data["chunks_created"] > 0
    
    # 4. Assert that the vector and lexical stores were indexed
    mock_qdrant.upsert_resumes.assert_called_once()
    mock_bm25.index.assert_called_once()

    # Verify that the correct chunk IDs and payloads were forwarded
    args_qdrant, kwargs_qdrant = mock_qdrant.upsert_resumes.call_args
    # args: (ids, vectors, payloads)
    assert len(args_qdrant[0]) == response_data["chunks_created"]
    assert len(args_qdrant[1]) == response_data["chunks_created"]
    assert len(args_qdrant[2]) == response_data["chunks_created"]
    assert args_qdrant[2][0]["trust_rating"] == 0.8
    assert args_qdrant[2][0]["created_at"] == "2026-05-26T12:00:00Z"

    args_bm25, kwargs_bm25 = mock_bm25.index.call_args
    # args: (ids, texts, payloads)
    assert len(args_bm25[0]) == response_data["chunks_created"]
    assert len(args_bm25[1]) == response_data["chunks_created"]
    assert len(args_bm25[2]) == response_data["chunks_created"]


@patch("api.routers.ingestion._extract_text")
@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
@patch("retrieval.lexical_store.bm25.BM25LexicalStore")
def test_ingest_jd_indexing_calls(mock_bm25_cls, mock_qdrant_cls, mock_extract, client):
    """Verify that ingesting a job description indexes it in both vector and lexical stores."""
    # 1. Mock text extraction
    mock_extract.return_value = "Seeking a Senior Python Backend Developer with database engineering skills."

    # 2. Mock store instances
    mock_qdrant = MagicMock()
    mock_qdrant.upsert_jds = AsyncMock()
    mock_qdrant_cls.return_value = mock_qdrant

    mock_bm25 = MagicMock()
    mock_bm25.index = AsyncMock()
    mock_bm25_cls.return_value = mock_bm25

    # 3. Post a JD upload
    file_data = {
        "file": ("job.docx", io.BytesIO(b"dummy docx content"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    }
    data = {
        "trust_rating": 0.95,
        "created_at": "2026-05-26T15:30:00Z"
    }
    
    response = client.post("/ingest/jd", files=file_data, data=data)
    
    assert response.status_code == status.HTTP_201_CREATED
    response_data = response.json()
    assert response_data["document_id"] != ""
    assert response_data["chunks_created"] > 0
    
    # 4. Assert that the vector and lexical stores were indexed
    mock_qdrant.upsert_jds.assert_called_once()
    mock_bm25.index.assert_called_once()

    # Verify that the correct details were indexed
    args_qdrant, kwargs_qdrant = mock_qdrant.upsert_jds.call_args
    assert len(args_qdrant[0]) == response_data["chunks_created"]
    assert args_qdrant[2][0]["trust_rating"] == 0.95
    assert args_qdrant[2][0]["created_at"] == "2026-05-26T15:30:00Z"
