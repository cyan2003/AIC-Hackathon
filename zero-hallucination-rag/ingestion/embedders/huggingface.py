"""
HuggingFace sentence-transformers embedder.

Wraps the ``sentence-transformers`` library to produce dense vector
embeddings.  Supports configurable model, device, batch size, and
normalisation.
"""

from __future__ import annotations

from typing import Sequence

from core.config import EmbeddingConfig
from core.exceptions import EmbeddingError
from ingestion.embedders.base import BaseEmbedder
from utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedder backed by a HuggingFace ``SentenceTransformer`` model.

    Parameters
    ----------
    config:
        Embedding configuration (model name, device, batch size, etc.).
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None  # Lazy-loaded
        self._dimension: int = config.dimension

    # -- Lazy model loading ------------------------------------------------

    def _load_model(self):
        """Load the sentence-transformer model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model",
                model=self._config.model_name,
                device=self._config.device,
            )
            self._model = SentenceTransformer(
                self._config.model_name,
                device=self._config.device,
                cache_folder=self._config.cache_dir,
            )

            if self._config.max_seq_length:
                self._model.max_seq_length = self._config.max_seq_length

            # Auto-detect dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()

            logger.info(
                "Embedding model loaded",
                dimension=self._dimension,
                max_seq_length=self._model.max_seq_length,
            )
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is required but not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self._config.model_name}': {exc}"
            ) from exc

    # -- Public API --------------------------------------------------------

    def _run_raw_embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Runs the raw sentence transformer embedding encoding."""
        try:
            embeddings = self._model.encode(
                list(texts),
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts, utilizing the SQLite cache if enabled."""
        from core.config import get_settings
        settings = get_settings()
        cache_enabled = getattr(settings, "cache", None) and settings.cache.enable_embedding_cache

        if not cache_enabled:
            self._load_model()
            return self._run_raw_embed(texts)

        from core.cache import get_cache_instance
        cache = get_cache_instance()

        results: list[Optional[list[float]]] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached_vec = cache.get_embedding(text)
            if cached_vec is not None:
                results[i] = cached_vec
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            self._load_model()
            miss_embeddings = self._run_raw_embed(miss_texts)
            for idx, text, emb in zip(miss_indices, miss_texts, miss_embeddings):
                cache.set_embedding(text, emb)
                results[idx] = emb

        return results  # type: ignore

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        result = self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimension
