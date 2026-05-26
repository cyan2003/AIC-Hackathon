"""
Integration tests for Source Confidence Scoring and Hallucination Fallback.
"""

import pytest
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from api.main import app
from core.config import RetrievalConfig, LexicalStoreConfig, FusionConfig, get_settings
from retrieval.interfaces import SearchResult
from retrieval.fusion.reciprocal_rank import ReciprocalRankFusion
from retrieval.pipeline import RetrievalPipeline


class DummyEmbedder:
    """Mock embedder returning dummy vectors."""
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed_single(self, text):
        return [0.1] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim


class MockVectorStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, vector, *, top_k=10, filters=None) -> list[SearchResult]:
        return self._results


class MockLexicalStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, query: str, *, top_k=10, filters=None) -> list[SearchResult]:
        return self._results


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_confidence_scoring_calculations():
    """Verify confidence score calculations based on similarity, freshness and trust."""
    settings = get_settings()
    
    # Configure weights and threshold
    retrieval_config = RetrievalConfig(
        min_confidence_threshold=0.4,
        max_freshness_days=180,
        weight_similarity=0.6,
        weight_freshness=0.2,
        weight_trust=0.2,
        min_score_threshold=0.0,
        enable_reranking=False,
    )
    
    # 1. Fresh document (0 days old), trusted source (1.0), similarity score (1.0)
    # Expected confidence: 0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 1.0 = 1.0
    r1 = SearchResult(
        document_id="doc_fresh",
        chunk_id="chunk1",
        text="Fresh document content",
        score=1.0,
        metadata={"created_at": datetime.now(timezone.utc).isoformat(), "trust_rating": 1.0}
    )
    
    # 2. Older document (90 days old -> freshness = 0.5), trusted source (1.0), similarity (0.5)
    # Expected confidence: 0.6 * 0.5 + 0.2 * 0.5 + 0.2 * 1.0 = 0.3 + 0.1 + 0.2 = 0.6
    ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    r2 = SearchResult(
        document_id="doc_older",
        chunk_id="chunk2",
        text="Older document content",
        score=0.5,
        metadata={"created_at": ninety_days_ago, "trust_rating": 1.0}
    )

    # 3. Low trust rating (0.0), fresh document, similarity (0.5)
    # Expected confidence: 0.6 * 0.5 + 0.2 * 1.0 + 0.2 * 0.0 = 0.3 + 0.2 + 0.0 = 0.5
    r3 = SearchResult(
        document_id="doc_low_trust",
        chunk_id="chunk3",
        text="Low trust document",
        score=0.5,
        metadata={"created_at": datetime.now(timezone.utc).isoformat(), "trust_rating": 0.0}
    )

    # 4. Old document (180 days old -> freshness = 0.0), low trust (0.0), similarity (0.0)
    # Expected confidence: 0.6 * 0.0 + 0.2 * 0.0 + 0.2 * 0.0 = 0.0
    # (Should be filtered out because 0.0 < 0.4 threshold)
    hundred_eighty_days_ago = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    r4 = SearchResult(
        document_id="doc_stale",
        chunk_id="chunk4",
        text="Stale document",
        score=0.0,
        metadata={"created_at": hundred_eighty_days_ago, "trust_rating": 0.0}
    )

    results = [r1, r2, r3, r4]
    
    pipeline = RetrievalPipeline(
        config=retrieval_config,
        vector_store=MockVectorStore(results),
        lexical_store=MockLexicalStore([]),
        embedder=DummyEmbedder(),
        fuser=ReciprocalRankFusion(FusionConfig(rrf_k=60)),
    )
    
    fused_results = pipeline._apply_threshold(results)
    
    # Assertions
    assert len(fused_results) == 3  # r4 filtered out
    
    # Check r1
    res_r1 = next(r for r in fused_results if r.document_id == "doc_fresh")
    assert res_r1.confidence_score == 1.0
    
    # Check r2
    res_r2 = next(r for r in fused_results if r.document_id == "doc_older")
    assert res_r2.confidence_score == 0.6

    # Check r3
    res_r3 = next(r for r in fused_results if r.document_id == "doc_low_trust")
    assert res_r3.confidence_score == 0.5


@patch("retrieval.vector_store.qdrant.QdrantVectorStore")
@patch("retrieval.lexical_store.bm25.BM25LexicalStore")
def test_match_endpoint_fallback_trigger(mock_bm25_cls, mock_qdrant_cls, client):
    """Test /match endpoint triggers the LLM bypass fallback when confidence is below threshold."""
    # 1. Setup mock stores to return stale, low trust search results
    # RRF / Similarity score = 0.2
    # Freshness score = 0.0 (180 days old)
    # Trust rating = 0.2
    # Combined confidence: 0.6 * 0.2 + 0.2 * 0.0 + 0.2 * 0.2 = 0.12 + 0.0 + 0.04 = 0.16 (well below 0.4 threshold)
    stale_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    mock_results = [
        SearchResult(
            document_id="doc_low_conf",
            chunk_id="chunk_low_conf",
            text="Low confidence matching chunk",
            score=0.2,
            metadata={"created_at": stale_date, "trust_rating": 0.2, "job_title": "Software Developer"}
        )
    ]
    
    mock_qdrant = MagicMock()
    mock_qdrant.search_jds = AsyncMock(return_value=mock_results)
    mock_qdrant.close = AsyncMock()
    mock_qdrant_cls.return_value = mock_qdrant
    
    mock_bm25 = MagicMock()
    mock_bm25.search = AsyncMock(return_value=[])
    mock_bm25_cls.return_value = mock_bm25

    # 2. Invoke match endpoint
    response = client.post("/match", data={"resume_text": "dummy resume content", "top_k": "5"})
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # 3. Assert fallback triggered correctly
    assert len(data["results"]) == 0
    assert data["assessment"] is not None
    assert data["assessment"]["recommendation"] == "No Match"
    assert "Insufficient evidence found" in data["assessment"]["summary"]
