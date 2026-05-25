"""
Section-aware chunking for resumes and job descriptions.

Detects domain-specific section headers (e.g. "Experience", "Skills",
"Requirements") and splits documents on section boundaries, preserving
section labels in chunk metadata.  Falls back to paragraph chunking
when no recognisable sections are found.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from core.config import ChunkingConfig
from core.constants import JD_SECTION_HEADERS, RESUME_SECTION_HEADERS
from ingestion.interfaces import DocumentChunker, ProcessedChunk
from utils.helpers import generate_chunk_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_header_pattern(headers: list[str]) -> re.Pattern:
    """
    Build a compiled regex that matches any of the given section headers.

    Matches lines that start with the header text (case-insensitive),
    optionally followed by a colon, and preceded by a newline or
    start-of-string.  Also matches ALL-CAPS headers and underlined
    headers (=== or ---).
    """
    escaped = [re.escape(h) for h in sorted(headers, key=len, reverse=True)]
    alternatives = "|".join(escaped)
    # Match: optional bullet/number prefix, header text, optional colon/dash
    pattern = (
        rf"(?:^|\n)"
        rf"[\s]*"
        rf"(?:\d+[\.\)]\s*)?"
        rf"(?P<header>{alternatives})"
        rf"[\s]*[:\-–—]?"
        rf"[\s]*$"
    )
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def _split_by_sections(
    text: str,
    header_pattern: re.Pattern,
    *,
    min_chunk_size: int = 50,
    max_chunk_size: int = 2048,
    document_id: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> list[ProcessedChunk]:
    """
    Split *text* into chunks based on detected section headers.

    Each chunk includes the section header label in its metadata under
    the key ``chunk_section``.
    """
    metadata = metadata or {}
    chunks: list[ProcessedChunk] = []

    # Find all section header positions.
    matches = list(header_pattern.finditer(text))

    if not matches:
        # No sections detected – return the whole text as a single chunk.
        stripped = text.strip()
        if len(stripped) >= min_chunk_size:
            chunks.append(
                ProcessedChunk(
                    chunk_id=generate_chunk_id(document_id, 0),
                    document_id=document_id,
                    text=stripped[:max_chunk_size],
                    chunk_index=0,
                    metadata={
                        **metadata,
                        "chunk_strategy": "section_aware",
                        "chunk_section": "unknown",
                    },
                )
            )
        return chunks

    # Handle text before the first section header (e.g. name/contact info).
    preamble = text[: matches[0].start()].strip()
    idx = 0
    if preamble and len(preamble) >= min_chunk_size:
        chunks.append(
            ProcessedChunk(
                chunk_id=generate_chunk_id(document_id, idx),
                document_id=document_id,
                text=preamble[:max_chunk_size],
                chunk_index=idx,
                metadata={
                    **metadata,
                    "chunk_strategy": "section_aware",
                    "chunk_section": "header",
                },
            )
        )
        idx += 1

    # Extract each section.
    for i, match in enumerate(matches):
        section_label = match.group("header").strip().lower()
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()

        if len(section_text) < min_chunk_size:
            continue

        # If section is too long, split into sub-chunks.
        if len(section_text) <= max_chunk_size:
            chunks.append(
                ProcessedChunk(
                    chunk_id=generate_chunk_id(document_id, idx),
                    document_id=document_id,
                    text=section_text,
                    chunk_index=idx,
                    metadata={
                        **metadata,
                        "chunk_strategy": "section_aware",
                        "chunk_section": section_label,
                    },
                )
            )
            idx += 1
        else:
            # Sub-split on paragraph boundaries within the section.
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
            buffer: list[str] = []
            buffer_len = 0

            for para in paragraphs:
                if buffer_len + len(para) > max_chunk_size and buffer:
                    chunk_text = "\n\n".join(buffer)
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=generate_chunk_id(document_id, idx),
                            document_id=document_id,
                            text=chunk_text,
                            chunk_index=idx,
                            metadata={
                                **metadata,
                                "chunk_strategy": "section_aware",
                                "chunk_section": section_label,
                            },
                        )
                    )
                    idx += 1
                    buffer = []
                    buffer_len = 0

                buffer.append(para)
                buffer_len += len(para)

            if buffer:
                chunk_text = "\n\n".join(buffer)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=generate_chunk_id(document_id, idx),
                            document_id=document_id,
                            text=chunk_text,
                            chunk_index=idx,
                            metadata={
                                **metadata,
                                "chunk_strategy": "section_aware",
                                "chunk_section": section_label,
                            },
                        )
                    )
                    idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Concrete chunkers
# ---------------------------------------------------------------------------


class ResumeSectionChunker(DocumentChunker):
    """
    Section-aware chunker tuned for resume documents.

    Detects typical resume sections (Summary, Experience, Education,
    Skills, Certifications, Projects, etc.) and produces one chunk per
    section with the ``chunk_section`` metadata field.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._min_size = config.min_chunk_size
        self._max_size = config.max_chunk_size
        self._pattern = _build_header_pattern(RESUME_SECTION_HEADERS)

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        return _split_by_sections(
            text,
            self._pattern,
            min_chunk_size=self._min_size,
            max_chunk_size=self._max_size,
            document_id=document_id,
            metadata=metadata,
        )


class JDSectionChunker(DocumentChunker):
    """
    Section-aware chunker tuned for job description documents.

    Detects typical JD sections (Responsibilities, Requirements,
    Qualifications, Benefits, etc.) and produces one chunk per section
    with the ``chunk_section`` metadata field.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._min_size = config.min_chunk_size
        self._max_size = config.max_chunk_size
        self._pattern = _build_header_pattern(JD_SECTION_HEADERS)

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ProcessedChunk]:
        return _split_by_sections(
            text,
            self._pattern,
            min_chunk_size=self._min_size,
            max_chunk_size=self._max_size,
            document_id=document_id,
            metadata=metadata,
        )
