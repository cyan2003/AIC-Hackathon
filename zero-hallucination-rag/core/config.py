"""
Configuration management for the Zero-Hallucination RAG Pipeline.

Uses pydantic-settings for environment-based configuration with validation,
type coercion, and sensible defaults for all pipeline components.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DocumentType(str, Enum):
    """Types of documents handled by the pipeline."""

    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""

    HUGGINGFACE = "huggingface"


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""

    QDRANT = "qdrant"
    MILVUS = "milvus"


class LexicalStoreBackend(str, Enum):
    """Supported lexical search backends."""

    BM25 = "bm25"


class ChunkingStrategy(str, Enum):
    """Supported document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    SECTION_AWARE = "section_aware"


class FusionStrategy(str, Enum):
    """Supported retrieval fusion strategies."""

    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED = "weighted"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class EmbeddingConfig(BaseSettings):
    """Configuration for the embedding subsystem."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    max_seq_length: int = 512
    device: str = "cpu"
    normalize: bool = True
    cache_dir: Optional[str] = None


class ChunkingConfig(BaseSettings):
    """Configuration for the document chunking subsystem."""

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    max_chunk_size: int = 2048

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class VectorStoreConfig(BaseSettings):
    """Configuration for the vector store backend."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_")

    backend: VectorStoreBackend = VectorStoreBackend.QDRANT
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    resume_collection: str = "resumes"
    jd_collection: str = "job_descriptions"
    prefer_grpc: bool = True
    api_key: Optional[str] = None
    timeout: float = 30.0


class LexicalStoreConfig(BaseSettings):
    """Configuration for the lexical search backend."""

    model_config = SettingsConfigDict(env_prefix="LEXICAL_STORE_")

    backend: LexicalStoreBackend = LexicalStoreBackend.BM25
    k1: float = 1.5
    b: float = 0.75
    max_index_size: int = 10_000_000


class FusionConfig(BaseSettings):
    """Configuration for result fusion and re-ranking."""

    model_config = SettingsConfigDict(env_prefix="FUSION_")

    strategy: FusionStrategy = FusionStrategy.RECIPROCAL_RANK
    rrf_k: int = 60
    vector_weight: float = 0.6
    lexical_weight: float = 0.4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 50


class IngestionConfig(BaseSettings):
    """Aggregate configuration for the ingestion pipeline."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    batch_size: int = 100
    max_workers: int = 4
    max_document_size_mb: float = 50.0
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx"]
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class RetrievalConfig(BaseSettings):
    """Aggregate configuration for the retrieval pipeline."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = 10
    min_score_threshold: float = 0.0
    min_confidence_threshold: float = 0.4
    max_freshness_days: int = 180
    weight_similarity: float = 0.6
    weight_freshness: float = 0.2
    weight_trust: float = 0.2
    enable_reranking: bool = True
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class AgentConfig(BaseSettings):
    """Configuration for the LLM-powered assessment agent."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    api_key: Optional[str] = Field(None, alias="CHUTES_API_KEY")
    base_url: str = "https://llm.chutes.ai/v1"
    model: str = "deepseek-ai/DeepSeek-R1-0528"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: float = 120.0


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root application settings.

    Reads from environment variables and ``.env`` file at the project root.
    All sub-configs are instantiated with sensible defaults that can be
    overridden individually via env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application --
    app_name: str = "zero-hallucination-rag"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # -- Sub-configs --
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def get_settings() -> Settings:
    """Instantiate and return the application settings (singleton-friendly)."""
    return Settings()
