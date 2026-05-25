"""
Document chunking strategies.

Provides multiple strategies for splitting cleaned text into chunks
suitable for embedding. Each strategy is a concrete implementation of
the :class:`~ingestion.interfaces.DocumentChunker` interface.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from core.config import ChunkingConfig, ChunkingStrategy
from ingestion.interfaces import DocumentChunker, ProcessedChunk
from utils.helpers import generate_chunk_id


class FixedSizeChunker(DocumentChunker):
    """
    Split text into fixed-size character windows with configurable overlap.

    This is the simplest and most predictable chunking strategy, suitable
    for homogeneous text where semantic boundaries are less important.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size
        self._overlap = config.chunk_overlap
        self._min_size = config.min_chunk_size

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        metadata = metadata or {}
        chunks: list[ProcessedChunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._size
            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= self._min_size:
                chunks.append(
                    ProcessedChunk(
                        chunk_id=generate_chunk_id(document_id, idx),
                        document_id=document_id,
                        text=chunk_text.strip(),
                        chunk_index=idx,
                        metadata={**metadata, "chunk_strategy": "fixed_size"},
                    )
                )
                idx += 1

            start += self._size - self._overlap

        return chunks


class SentenceChunker(DocumentChunker):
    """
    Group sentences into chunks that stay within the configured size limit.

    Sentence boundaries are detected via a simple regex heuristic.
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size
        self._overlap = config.chunk_overlap
        self._min_size = config.min_chunk_size

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        metadata = metadata or {}
        sentences = self._SENTENCE_SPLIT.split(text)
        chunks: list[ProcessedChunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_len + len(sentence) > self._size and current_sentences:
                chunk_text = " ".join(current_sentences)
                if len(chunk_text) >= self._min_size:
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=generate_chunk_id(document_id, idx),
                            document_id=document_id,
                            text=chunk_text,
                            chunk_index=idx,
                            metadata={**metadata, "chunk_strategy": "sentence"},
                        )
                    )
                    idx += 1

                # Keep overlap sentences for context continuity.
                overlap_chars = 0
                overlap_start = len(current_sentences)
                for i in range(len(current_sentences) - 1, -1, -1):
                    overlap_chars += len(current_sentences[i])
                    if overlap_chars >= self._overlap:
                        overlap_start = i
                        break
                current_sentences = current_sentences[overlap_start:]
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_len += len(sentence)

        # Flush remaining sentences.
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text) >= self._min_size:
                chunks.append(
                    ProcessedChunk(
                        chunk_id=generate_chunk_id(document_id, idx),
                        document_id=document_id,
                        text=chunk_text,
                        chunk_index=idx,
                        metadata={**metadata, "chunk_strategy": "sentence"},
                    )
                )

        return chunks


class ParagraphChunker(DocumentChunker):
    """
    Split on paragraph boundaries (double newlines), merging short paragraphs
    until the chunk size limit is reached.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size
        self._min_size = config.min_chunk_size

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        metadata = metadata or {}
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[ProcessedChunk] = []
        buffer: list[str] = []
        buffer_len = 0
        idx = 0

        for para in paragraphs:
            if buffer_len + len(para) > self._size and buffer:
                chunk_text = "\n\n".join(buffer)
                if len(chunk_text) >= self._min_size:
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=generate_chunk_id(document_id, idx),
                            document_id=document_id,
                            text=chunk_text,
                            chunk_index=idx,
                            metadata={**metadata, "chunk_strategy": "paragraph"},
                        )
                    )
                    idx += 1
                buffer = []
                buffer_len = 0

            buffer.append(para)
            buffer_len += len(para)

        if buffer:
            chunk_text = "\n\n".join(buffer)
            if len(chunk_text) >= self._min_size:
                chunks.append(
                    ProcessedChunk(
                        chunk_id=generate_chunk_id(document_id, idx),
                        document_id=document_id,
                        text=chunk_text,
                        chunk_index=idx,
                        metadata={**metadata, "chunk_strategy": "paragraph"},
                    )
                )

        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_chunker(config: ChunkingConfig) -> DocumentChunker:
    """
    Factory function returning the appropriate chunker for the configured strategy.

    For ``SECTION_AWARE`` strategy, returns a :class:`ResumeSectionChunker`
    by default.  Use :func:`create_chunker_for_document_type` to get the
    correct domain-specific chunker.
    """
    from ingestion.processors.section_chunker import ResumeSectionChunker

    _registry: dict[ChunkingStrategy, type[DocumentChunker]] = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.SECTION_AWARE: ResumeSectionChunker,
    }
    cls = _registry.get(config.strategy)
    if cls is None:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")
    return cls(config)


def create_chunker_for_document_type(
    config: ChunkingConfig,
    document_type: str,
) -> DocumentChunker:
    """
    Return the section-aware chunker appropriate for *document_type*.

    Parameters
    ----------
    config:
        Chunking configuration.
    document_type:
        ``"resume"`` or ``"job_description"``.
    """
    from ingestion.processors.section_chunker import (
        JDSectionChunker,
        ResumeSectionChunker,
    )

    if document_type == "job_description":
        return JDSectionChunker(config)
    return ResumeSectionChunker(config)

