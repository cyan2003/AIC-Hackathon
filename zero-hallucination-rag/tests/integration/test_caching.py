"""
Integration tests to verify embedding and LLM assessment caching behavior.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from core.config import get_settings
from core.cache import get_cache_instance
from ingestion.embedders.huggingface import HuggingFaceEmbedder
from agent.assessor import CandidateAssessor
from agent.schemas import CandidateAssessment


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure the SQLite cache tables are cleared before every test."""
    cache = get_cache_instance()
    cache.clear()
    yield
    cache.clear()


def test_embedding_caching():
    """Verify that HuggingFaceEmbedder caches outputs and only calls model on misses."""
    settings = get_settings()
    embedder = HuggingFaceEmbedder(settings.ingestion.embedding)

    # Mock the raw embedding model encoder method
    with patch.object(embedder, "_run_raw_embed", return_value=[[0.1] * 384]) as mock_raw:
        # First call: cache miss
        vec1 = embedder.embed(["Hello, World!"])
        assert len(vec1) == 1
        assert vec1[0] == [0.1] * 384
        mock_raw.assert_called_once_with(["Hello, World!"])

        # Second call: cache hit
        mock_raw.reset_mock()
        vec2 = embedder.embed(["Hello, World!"])
        assert len(vec2) == 1
        assert vec2[0] == [0.1] * 384
        mock_raw.assert_not_called()


@pytest.mark.asyncio
async def test_llm_assessment_caching():
    """Verify that CandidateAssessor caches outputs and returns cache_hit=True on hits."""
    settings = get_settings()
    
    # Enable cache in settings config (in case it is disabled in test config environment)
    settings.cache.enable_llm_cache = True
    
    # We must patch the AsyncOpenAI client call to avoid external network requests
    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        # Setup mock OpenAI completion response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # DeepSeek R1 structured candidate assessment response format
        mock_message.content = """
        <think>Some reasoning...</think>
        ```json
        {
            "overall_score": 0.85,
            "recommendation": "Strong Match",
            "summary": "Excellent fit.",
            "strengths": ["Python expert"],
            "gaps": ["None"],
            "cited_evidence": []
        }
        ```
        """
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_client

        assessor = CandidateAssessor(settings.agent)
        # Ensure api_key is set to allow running
        assessor._config.api_key = "mock-key"

        resume_text = "Experienced software engineer."
        resume_metadata = {"candidate_name": "Alice"}
        match_results = [{"job_title": "Python Dev", "score": 0.9}]

        # First call: cache miss, calls LLM
        res1 = await assessor.assess(resume_text, resume_metadata, match_results)
        assert res1.overall_score == 0.85
        assert res1.recommendation == "Strong Match"
        assert res1.cache_hit is False
        mock_client.chat.completions.create.assert_called_once()

        # Second call: cache hit, retrieves from SQLite instantly
        mock_client.chat.completions.create.reset_mock()
        res2 = await assessor.assess(resume_text, resume_metadata, match_results)
        assert res2.overall_score == 0.85
        assert res2.recommendation == "Strong Match"
        assert res2.cache_hit is True
        # Verify the OpenAI client was not invoked again
        mock_client.chat.completions.create.assert_not_called()
