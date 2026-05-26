"""
Integration tests to verify Qdrant, BM25, and API endpoint metadata filtering.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from api.main import app
from retrieval.vector_store.qdrant import QdrantVectorStore
from retrieval.lexical_store.bm25 import BM25LexicalStore


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_qdrant_filter_builder():
    """Verify that QdrantVectorStore._build_filter constructs correct Qdrant Filter objects."""
    # Test 1: Skills filter
    f1 = QdrantVectorStore._build_filter({"skills": ["Python", "SQL"]})
    assert f1 is not None
    cond1 = f1.must[0]
    assert cond1.key == "required_skills"
    assert cond1.match.any == ["python", "sql"]

    # Test 2: Experience filter (lte and empty condition)
    f2 = QdrantVectorStore._build_filter({"experience_years_max": 5})
    assert f2 is not None
    cond2 = f2.must[0]
    # Should be a Filter object containing should block
    assert len(cond2.should) == 2
    assert cond2.should[0].key == "experience_years"
    assert cond2.should[0].range.lte == 5
    assert cond2.should[1].key == "experience_years"
    assert cond2.should[1].is_empty.key == "experience_years"

    # Test 3: Education filter rank matching (Master candidate matches Bachelor/Associate/Master)
    f3 = QdrantVectorStore._build_filter({"education_level": "master"})
    assert f3 is not None
    cond3 = f3.must[0]
    assert len(cond3.should) == 2
    assert cond3.should[0].key == "education_level"
    assert set(cond3.should[0].match.any) == {"associate", "bachelor", "master"}
    assert cond3.should[1].key == "education_level"
    assert cond3.should[1].is_empty.key == "education_level"

    # Test 4: Industry exact match
    f4 = QdrantVectorStore._build_filter({"industry": "Healthcare"})
    assert f4 is not None
    cond4 = f4.must[0]
    assert cond4.key == "industry"
    assert cond4.match.value == "healthcare"


def test_bm25_matches_filter():
    """Verify that BM25LexicalStore._matches_filter correctly matches payloads."""
    payload = {
        "required_skills": ["Python", "FastAPI"],
        "experience_years": 4,
        "education_level": "bachelor",
        "industry": "technology"
    }

    # 1. Skills filtering
    assert BM25LexicalStore._matches_filter(payload, {"skills": ["Python"]}) is True
    assert BM25LexicalStore._matches_filter(payload, {"skills": ["Java"]}) is False

    # 2. Experience filtering (candidate max experience >= JD required experience)
    assert BM25LexicalStore._matches_filter(payload, {"experience_years_max": 5}) is True
    assert BM25LexicalStore._matches_filter(payload, {"experience_years_max": 3}) is False
    # If JD doesn't list experience, it should pass
    empty_exp_payload = {"experience_years": None}
    assert BM25LexicalStore._matches_filter(empty_exp_payload, {"experience_years_max": 3}) is True

    # 3. Education rank filtering (candidate has Master -> matches JD Bachelor)
    assert BM25LexicalStore._matches_filter(payload, {"education_level": "master"}) is True
    # candidate has Associate -> fails JD Bachelor
    assert BM25LexicalStore._matches_filter(payload, {"education_level": "associate"}) is False
    # If JD has no education specified, it should pass
    empty_edu_payload = {"education_level": None}
    assert BM25LexicalStore._matches_filter(empty_edu_payload, {"education_level": "bachelor"}) is True

    # 4. Industry filtering
    assert BM25LexicalStore._matches_filter(payload, {"industry": "Technology"}) is True
    assert BM25LexicalStore._matches_filter(payload, {"industry": "Finance"}) is False


@patch("ingestion.embedders.factory.create_embedder")
@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
@patch("retrieval.lexical_store.bm25.BM25LexicalStore")
def test_api_match_filters_forwarding(mock_bm25_cls, mock_qdrant_cls, mock_embedder_factory, client):
    """Verify that filters passed to the API are successfully parsed and forwarded to stores."""
    # Mock search backends
    mock_qdrant = MagicMock()
    mock_qdrant.search_jds = AsyncMock(return_value=[])
    mock_qdrant_cls.return_value = mock_qdrant

    mock_bm25 = MagicMock()
    mock_bm25.search = AsyncMock(return_value=[])
    mock_bm25_cls.return_value = mock_bm25

    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.1] * 384
    mock_embedder_factory.return_value = mock_embedder

    # Send match request with filters form parameter
    filters_data = {
        "skills": ["Python"],
        "experience_years_max": 4,
        "education_level": "master",
        "industry": "technology"
    }

    response = client.post(
        "/match",
        data={
            "resume_text": "Experienced Python backend dev.",
            "top_k": 5,
            "filters": json.dumps(filters_data)
        }
    )

    assert response.status_code == status.HTTP_200_OK

    # Assert stores were queried with parsed filters dictionary
    mock_qdrant.search_jds.assert_called_once()
    args_q, kwargs_q = mock_qdrant.search_jds.call_args
    assert kwargs_q["filters"] == filters_data

    mock_bm25.search.assert_called_once()
    args_b, kwargs_b = mock_bm25.search.call_args
    assert kwargs_b["filters"] == filters_data
