"""
Integration tests for the retrieval pipeline module.
"""
import pytest
from core.config import RetrievalConfig, LexicalStoreConfig, FusionConfig
from ingestion.interfaces import Embedder
from retrieval.interfaces import VectorStore, SearchResult
from retrieval.lexical_store.bm25 import BM25LexicalStore
from retrieval.fusion.reciprocal_rank import ReciprocalRankFusion
from retrieval.query.processor import DefaultQueryProcessor
from retrieval.pipeline import RetrievalPipeline


class DummyEmbedder(Embedder):
    """Simple embedder returning dummy vectors for query vectorisation."""
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, texts):
        return [[0.1] * self._dim for _ in texts]

    def embed_single(self, text):
        return [0.1] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim


class MockVectorStore(VectorStore):
    """Mock VectorStore returning fixed results."""
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def initialize(self) -> None:
        pass

    async def upsert(self, ids, vectors, payloads=None) -> None:
        pass

    async def search(self, vector, *, top_k=10, filters=None) -> list[SearchResult]:
        return self._results

    async def delete(self, ids) -> None:
        pass

    async def count(self) -> int:
        return len(self._results)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_retrieval_pipeline_hybrid_search():
    """Test hybrid retrieval pipeline executing vector & BM25 searches and fusing them."""
    # 1. Configuration
    retrieval_config = RetrievalConfig(
        top_k=2,
        min_score_threshold=0.0,
        enable_reranking=False,
    )
    
    # 2. Vector Store Setup (Mocked)
    vector_results = [
        SearchResult(
            document_id="doc_vector_only",
            chunk_id="chunk_vector_only",
            text="Vector search matched candidate with Python experience.",
            score=0.92,
        ),
        SearchResult(
            document_id="doc_shared",
            chunk_id="chunk_shared",
            text="Shared candidate matched by both systems.",
            score=0.88,
        ),
    ]
    vector_store = MockVectorStore(vector_results)
    
    # 3. Lexical Store Setup (BM25 - Real)
    lexical_config = LexicalStoreConfig(k1=1.5, b=0.75, max_index_size=100)
    lexical_store = BM25LexicalStore(lexical_config)
    
    await lexical_store.index(
        ids=["chunk_shared", "chunk_lexical_only", "chunk_dummy1", "chunk_dummy2"],
        texts=[
            "Shared candidate matched by both systems.",
            "Lexical search matched candidate text.",
            "Dummy job detail three",
            "Dummy job detail four",
        ],
        payloads=[
            {"document_id": "doc_shared"},
            {"document_id": "doc_lexical_only"},
            {"document_id": "doc_dummy1"},
            {"document_id": "doc_dummy2"},
        ]
    )
    
    # 4. Shared components
    embedder = DummyEmbedder()
    fuser = ReciprocalRankFusion(FusionConfig(rrf_k=60))
    query_processor = DefaultQueryProcessor(lowercase=True, remove_stopwords=True)
    
    # 5. Pipeline instantiation
    pipeline = RetrievalPipeline(
        config=retrieval_config,
        vector_store=vector_store,
        lexical_store=lexical_store,
        embedder=embedder,
        fuser=fuser,
        query_processor=query_processor,
        reranker=None,
    )
    
    # Run retrieval
    response = await pipeline.retrieve("candidate systems", top_k=2)
    
    assert response is not None
    assert response.query.original_text == "candidate systems"
    assert len(response.results) <= 2
    assert response.total_candidates > 0
    
    # Verify timings are present
    assert "query_processing" in response.timings
    assert "query_embedding" in response.timings
    assert "vector_search" in response.timings
    assert "lexical_search" in response.timings
    assert "fusion" in response.timings
