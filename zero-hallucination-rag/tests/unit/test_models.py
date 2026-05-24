"""
Unit tests for Pydantic data models in the Document Ingestion pipeline.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError
from api.models.document import RawDocument, ChunkedDocument


def test_raw_document_valid():
    """Test creating a valid RawDocument."""
    doc = RawDocument(
        id="doc_123",
        content="This is a test document.",
        format="text/plain",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    assert doc.id == "doc_123"
    assert doc.content == "This is a test document."
    assert doc.format == "text/plain"
    assert isinstance(doc.created_at, datetime)
    assert isinstance(doc.updated_at, datetime)


def test_raw_document_missing_required_fields():
    """Test that RawDocument requires id, content, and format."""
    with pytest.raises(ValidationError) as exc_info:
        RawDocument()  # Missing all required fields

    errors = exc_info.value.errors()
    assert len(errors) == 3
    error_fields = {error["loc"][0] for error in errors}
    assert error_fields == {"id", "content", "format"}


def test_raw_document_invalid_format():
    """Test RawDocument with invalid format type."""
    with pytest.raises(ValidationError):
        RawDocument(
            id="doc_123",
            content="Test content",
            format=123,  # Should be string
            created_at=datetime.now(),
            updated_at=datetime.now()
        )


def test_raw_document_timestamps_optional():
    """Test that timestamps are optional in RawDocument."""
    doc = RawDocument(
        id="doc_123",
        content="Test content",
        format="text/plain"
    )

    assert doc.created_at is None
    assert doc.updated_at is None


def test_chunked_document_valid():
    """Test creating a valid ChunkedDocument."""
    chunk = ChunkedDocument(
        id="chunk_456",
        chunk_index=0,
        content="This is a test chunk.",
        parent_id="doc_123",
        embedding=[0.1, 0.2, 0.3, 0.4],
        metadata={"source": "test", "page": 1}
    )

    assert chunk.id == "chunk_456"
    assert chunk.chunk_index == 0
    assert chunk.content == "This is a test chunk."
    assert chunk.parent_id == "doc_123"
    assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]
    assert chunk.metadata == {"source": "test", "page": 1}


def test_chunked_document_missing_required_fields():
    """Test that ChunkedDocument requires id, chunk_index, content, parent_id, and embedding."""
    with pytest.raises(ValidationError) as exc_info:
        ChunkedDocument()  # Missing all required fields

    errors = exc_info.value.errors()
    assert len(errors) == 5
    error_fields = {error["loc"][0] for error in errors}
    assert error_fields == {"id", "chunk_index", "content", "parent_id", "embedding"}


def test_chunked_document_embedding_type_validation():
    """Test that embedding must be a list of numbers."""
    with pytest.raises(ValidationError):
        ChunkedDocument(
            id="chunk_456",
            chunk_index=0,
            content="Test chunk",
            parent_id="doc_123",
            embedding="not a list",  # Should be list
            metadata={}
        )

    with pytest.raises(ValidationError):
        ChunkedDocument(
            id="chunk_456",
            chunk_index=0,
            content="Test chunk",
            parent_id="doc_123",
            embedding=[0.1, "invalid", 0.3],  # Should be numbers
            metadata={}
        )


def test_chunked_document_chunk_index_non_negative():
    """Test that chunk_index must be non-negative."""
    with pytest.raises(ValidationError):
        ChunkedDocument(
            id="chunk_456",
            chunk_index=-1,  # Should be >= 0
            content="Test chunk",
            parent_id="doc_123",
            embedding=[0.1, 0.2],
            metadata={}
        )


def test_chunked_document_metadata_default():
    """Test that metadata defaults to empty dict."""
    chunk = ChunkedDocument(
        id="chunk_456",
        chunk_index=0,
        content="Test chunk",
        parent_id="doc_123",
        embedding=[0.1, 0.2]
    )

    assert chunk.metadata == {}


def test_chunked_document_metadata_optional():
    """Test that metadata can be explicitly set."""
    chunk = ChunkedDocument(
        id="chunk_456",
        chunk_index=0,
        content="Test chunk",
        parent_id="doc_123",
        embedding=[0.1, 0.2],
        metadata={"author": "John Doe", "section": "Introduction"}
    )

    assert chunk.metadata == {"author": "John Doe", "section": "Introduction"}