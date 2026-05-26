"""
Unit tests for the retrieval module components.
"""
import pytest
from unittest.mock import MagicMock, patch
from core.config import LexicalStoreConfig, FusionConfig
from retrieval.interfaces import SearchResult, RetrievalQuery
from retrieval.lexical_store.bm25 import BM25LexicalStore
from retrieval.fusion.reciprocal_rank import ReciprocalRankFusion
from retrieval.fusion.weighted import WeightedScoreFusion
from retrieval.fusion.reranker import CrossEncoderReranker
from retrieval.query.processor import DefaultQueryProcessor
from retrieval.query.transformer import DefaultQueryTransformer


@pytest.mark.asyncio
async def test_bm25_lexical_store():
    """Test BM25 database index, search, filters, and deletion."""
    config = LexicalStoreConfig(k1=1.5, b=0.75, max_index_size=100)
    store = BM25LexicalStore(config)
    
    # Empty index search
    results = await store.search("python")
    assert results == []
    
    # Index documents - include extra docs so that term frequency does not cancel out the BM25 IDF score
    ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    texts = [
        "Senior Python Software Engineer with Flask experience",
        "React frontend developer styling expert",
        "Data scientist using Python and pandas",
        "Ruby on Rails programmer web architect",
        "C++ systems engineer low latency",
        "Java spring boot developer backend",
    ]
    payloads = [
        {"document_id": "job1", "experience": "senior"},
        {"document_id": "job2", "experience": "mid"},
        {"document_id": "job3", "experience": "senior"},
        {"document_id": "job4", "experience": "senior"},
        {"document_id": "job5", "experience": "senior"},
        {"document_id": "job6", "experience": "senior"},
    ]
    
    await store.index(ids, texts, payloads)
    assert await store.count() == 6
    
    # Simple search
    results = await store.search("python", top_k=5)
    assert len(results) == 2  # doc1 and doc3 contain Python
    assert results[0].chunk_id in ["doc1", "doc3"]
    assert results[0].score > 0
    assert results[0].source == "lexical"

    # Search with filter
    results_filtered = await store.search(
        "python",
        filters={"experience": "senior"},
    )
    assert len(results_filtered) == 2
    
    results_filtered_mid = await store.search(
        "python",
        filters={"experience": "mid"},
    )
    assert len(results_filtered_mid) == 0

    # Deletion
    await store.delete(["doc1"])
    assert await store.count() == 5
    
    results_after_delete = await store.search("python")
    assert len(results_after_delete) == 1
    assert results_after_delete[0].chunk_id == "doc3"


def test_reciprocal_rank_fusion():
    """Test Reciprocal Rank Fusion calculation and sorting."""
    config = FusionConfig(rrf_k=60)
    fuser = ReciprocalRankFusion(config)
    
    # Candidate lists
    vector_results = [
        SearchResult(document_id="docA", chunk_id="chunkA1", text="Text A", score=0.9, source="vector"),
        SearchResult(document_id="docB", chunk_id="chunkB1", text="Text B", score=0.8, source="vector"),
    ]
    lexical_results = [
        SearchResult(document_id="docB", chunk_id="chunkB1", text="Text B", score=10.0, source="lexical"),
        SearchResult(document_id="docC", chunk_id="chunkC1", text="Text C", score=5.0, source="lexical"),
    ]
    
    fused = fuser.fuse([vector_results, lexical_results])
    
    # RRF score calculation:
    # chunkB1: ranks 2 (vector) and 1 (lexical) -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # chunkA1: ranks 1 (vector) and not in lexical -> 1/(60+1) = 0.016393
    # chunkC1: ranks 2 (lexical) and not in vector -> 1/(60+2) = 0.016129
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunkB1"
    assert fused[1].chunk_id == "chunkA1"
    assert fused[2].chunk_id == "chunkC1"
    assert fused[0].score > fused[1].score


def test_weighted_score_fusion():
    """Test weighted linear combination with normalized scores."""
    config = FusionConfig(vector_weight=0.7, lexical_weight=0.3)
    fuser = WeightedScoreFusion(config)
    
    vector_results = [
        SearchResult(document_id="docA", chunk_id="chunkA1", text="Text A", score=0.9, source="vector"),
        SearchResult(document_id="docB", chunk_id="chunkB1", text="Text B", score=0.7, source="vector"),
    ]  # Normalizes to 1.0 (A) and 0.0 (B)
    lexical_results = [
        SearchResult(document_id="docB", chunk_id="chunkB1", text="Text B", score=10.0, source="lexical"),
        SearchResult(document_id="docA", chunk_id="chunkA1", text="Text A", score=5.0, source="lexical"),
    ]  # Normalizes to 1.0 (B) and 0.0 (A)
    
    fused = fuser.fuse([vector_results, lexical_results])
    
    # chunkA1 normalized score: vector=1.0, lexical=0.0 -> Weighted = 1.0*0.7 + 0.0*0.3 = 0.7
    # chunkB1 normalized score: vector=0.0, lexical=1.0 -> Weighted = 0.0*0.7 + 1.0*0.3 = 0.3
    assert len(fused) == 2
    assert fused[0].chunk_id == "chunkA1"
    assert fused[0].score == pytest.approx(0.7)
    assert fused[1].chunk_id == "chunkB1"
    assert fused[1].score == pytest.approx(0.3)


@pytest.mark.asyncio
@patch("sentence_transformers.CrossEncoder")
async def test_cross_encoder_reranker(mock_cross_encoder_cls):
    """Test CrossEncoderReranker with mock sentence-transformers model."""
    mock_instance = MagicMock()
    # Mock prediction scores
    mock_instance.predict.return_value = [0.1, 0.9, 0.5]
    mock_cross_encoder_cls.return_value = mock_instance
    
    reranker = CrossEncoderReranker(model_name="mock-model", device="cpu")
    
    results = [
        SearchResult(document_id="doc1", chunk_id="c1", text="text1", score=0.5, source="rrf"),
        SearchResult(document_id="doc2", chunk_id="c2", text="text2", score=0.4, source="rrf"),
        SearchResult(document_id="doc3", chunk_id="c3", text="text3", score=0.3, source="rrf"),
    ]
    
    reranked = await reranker.rerank("query text", results, top_k=2)
    
    assert len(reranked) == 2
    # doc2 gets score 0.9 (rank 1), doc3 gets score 0.5 (rank 2), doc1 gets score 0.1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 0.9
    assert reranked[0].metadata["original_score"] == 0.4
    assert reranked[1].chunk_id == "c3"
    assert reranked[1].score == 0.5


def test_default_query_processor():
    """Test query whitespace normalization, casing, and stopwords removal."""
    processor = DefaultQueryProcessor(lowercase=True, remove_stopwords=True)
    
    raw = "  Python Developer   and  SQL developer needed  "
    processed = processor.process(raw)
    
    # "and", "needed" (if needed is not in stopwords list, wait, let's see. 'needed' is not in stopword list)
    # stopwords: "a", "an", "and", "are", "as", "at", "be", "but", "by", ...
    # "and" should be removed.
    assert "and" not in processed
    assert "python developer sql developer needed" in processed

    processor_no_stop = DefaultQueryProcessor(lowercase=False, remove_stopwords=False)
    processed_no_stop = processor_no_stop.process(raw)
    assert processed_no_stop == "Python Developer and SQL developer needed"


def test_default_query_transformer():
    """Test pass-through query transformer."""
    transformer = DefaultQueryTransformer()
    q = RetrievalQuery(original_text="python flask", top_k=5)
    
    assert transformer.for_vector(q) == q
    assert transformer.for_lexical(q) == q
