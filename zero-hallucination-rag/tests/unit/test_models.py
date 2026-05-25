"""
Unit tests for data models and Pydantic validation schemas in the pipeline.
"""
import pytest
from pydantic import ValidationError
from core.config import DocumentType
from ingestion.interfaces import RawDocument, ProcessedChunk, EmbeddedChunk
from api.models.ingestion import (
    ResumeIngestionResponse,
    JDIngestionResponse,
    IngestionErrorResponse,
)
from api.models.retrieval import (
    MatchFilters,
    MatchRequest,
    MatchedSection,
    MatchResult,
    MatchResponse,
)


def test_raw_document():
    """Test RawDocument dataclass initialization and defaults."""
    doc = RawDocument(
        source="test.pdf",
        content="hello resume content",
        document_type=DocumentType.RESUME,
    )
    assert doc.source == "test.pdf"
    assert doc.content == "hello resume content"
    assert doc.mime_type == "text/plain"
    assert doc.document_type == DocumentType.RESUME
    assert doc.metadata == {}

    doc_with_meta = RawDocument(
        source="test.docx",
        content=b"bytes",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={"author": "John Doe"},
    )
    assert doc_with_meta.content == b"bytes"
    assert doc_with_meta.metadata == {"author": "John Doe"}


def test_processed_chunk():
    """Test ProcessedChunk dataclass initialization."""
    chunk = ProcessedChunk(
        chunk_id="chunk-1",
        document_id="doc-123",
        text="Sample chunk content",
        chunk_index=2,
        metadata={"chunk_section": "experience"},
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.document_id == "doc-123"
    assert chunk.text == "Sample chunk content"
    assert chunk.chunk_index == 2
    assert chunk.metadata == {"chunk_section": "experience"}


def test_embedded_chunk():
    """Test EmbeddedChunk dataclass initialization."""
    chunk = ProcessedChunk(
        chunk_id="chunk-1",
        document_id="doc-123",
        text="Sample chunk content",
    )
    embedded = EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3])
    assert embedded.chunk == chunk
    assert embedded.embedding == [0.1, 0.2, 0.3]


def test_resume_ingestion_response():
    """Test ResumeIngestionResponse validation."""
    response = ResumeIngestionResponse(
        document_id="doc-123",
        source="resume.pdf",
        chunks_created=5,
        metadata={"candidate_name": "Alice Smith"},
    )
    assert response.document_id == "doc-123"
    assert response.source == "resume.pdf"
    assert response.chunks_created == 5
    assert response.metadata == {"candidate_name": "Alice Smith"}
    assert response.message == "Resume ingested successfully"

    # Test required fields validation
    with pytest.raises(ValidationError):
        ResumeIngestionResponse(document_id="doc-123")  # Missing other fields


def test_jd_ingestion_response():
    """Test JDIngestionResponse validation."""
    response = JDIngestionResponse(
        document_id="jd-456",
        source="jd.docx",
        chunks_created=3,
        metadata={"job_title": "Python Developer"},
    )
    assert response.document_id == "jd-456"
    assert response.source == "jd.docx"
    assert response.chunks_created == 3
    assert response.metadata == {"job_title": "Python Developer"}
    assert response.message == "Job description ingested successfully"


def test_ingestion_error_response():
    """Test IngestionErrorResponse validation."""
    err = IngestionErrorResponse(
        source="failed.pdf",
        error="Invalid file type",
    )
    assert err.source == "failed.pdf"
    assert err.error == "Invalid file type"
    assert err.message == "Ingestion failed"


def test_match_filters():
    """Test MatchFilters validation."""
    filters = MatchFilters(
        skills=["Python", "SQL"],
        experience_years_max=5,
        education_level="bachelor",
        industry="Tech",
    )
    assert filters.skills == ["Python", "SQL"]
    assert filters.experience_years_max == 5
    assert filters.education_level == "bachelor"
    assert filters.industry == "Tech"


def test_match_request():
    """Test MatchRequest validation."""
    req = MatchRequest(
        resume_text="Experienced Python Dev",
        top_k=5,
        filters=MatchFilters(skills=["Python"]),
    )
    assert req.resume_text == "Experienced Python Dev"
    assert req.top_k == 5
    assert req.filters.skills == ["Python"]

    # Test top_k validation
    with pytest.raises(ValidationError):
        MatchRequest(top_k=0)  # ge=1 validation failure
    with pytest.raises(ValidationError):
        MatchRequest(top_k=101)  # le=100 validation failure


def test_matched_section():
    """Test MatchedSection validation."""
    sec = MatchedSection(section="requirements", text="Must know Python", score=0.85)
    assert sec.section == "requirements"
    assert sec.text == "Must know Python"
    assert sec.score == 0.85


def test_match_result():
    """Test MatchResult validation."""
    res = MatchResult(
        document_id="jd-123",
        job_title="Software Engineer",
        score=0.92,
        matched_sections=[
            MatchedSection(section="requirements", text="Must know Python", score=0.85)
        ],
        metadata={"skills": ["Python"]},
    )
    assert res.document_id == "jd-123"
    assert res.job_title == "Software Engineer"
    assert res.score == 0.92
    assert len(res.matched_sections) == 1
    assert res.metadata == {"skills": ["Python"]}


def test_match_response():
    """Test MatchResponse validation."""
    resp = MatchResponse(
        results=[
            MatchResult(document_id="jd-123", score=0.92)
        ],
        total_candidates=1,
        timings={"search": 0.05, "rerank": 0.1},
        resume_metadata={"candidate_name": "Bob"},
    )
    assert len(resp.results) == 1
    assert resp.total_candidates == 1
    assert resp.timings["search"] == 0.05
    assert resp.resume_metadata == {"candidate_name": "Bob"}