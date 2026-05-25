"""
Application-wide constants for the Zero-Hallucination RAG Pipeline.

All magic numbers, default limits, and sentinel values should be
defined here so they can be referenced consistently across modules.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ingestion constants
# ---------------------------------------------------------------------------

#: Maximum number of documents that can be ingested in a single API call.
MAX_BATCH_INGEST_SIZE: int = 500

#: Default encoding assumed when no encoding metadata is available.
DEFAULT_TEXT_ENCODING: str = "utf-8"

#: Supported MIME types for document ingestion (PDF and DOCX only).
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)

# ---------------------------------------------------------------------------
# Collection constants
# ---------------------------------------------------------------------------

#: Default Qdrant collection name for resume vectors.
RESUME_COLLECTION_NAME: str = "resumes"

#: Default Qdrant collection name for job description vectors.
JD_COLLECTION_NAME: str = "job_descriptions"

# ---------------------------------------------------------------------------
# Resume section headers (case-insensitive matching)
# ---------------------------------------------------------------------------

RESUME_SECTION_HEADERS: list[str] = [
    "summary",
    "professional summary",
    "executive summary",
    "objective",
    "career objective",
    "profile",
    "about me",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "education",
    "academic background",
    "academic qualifications",
    "skills",
    "technical skills",
    "core competencies",
    "key skills",
    "competencies",
    "certifications",
    "certificates",
    "licenses",
    "projects",
    "personal projects",
    "key projects",
    "achievements",
    "accomplishments",
    "awards",
    "publications",
    "references",
    "languages",
    "volunteer",
    "volunteer experience",
    "interests",
    "hobbies",
]

# ---------------------------------------------------------------------------
# Job description section headers (case-insensitive matching)
# ---------------------------------------------------------------------------

JD_SECTION_HEADERS: list[str] = [
    "about",
    "about us",
    "about the company",
    "about the role",
    "about the position",
    "overview",
    "job overview",
    "role overview",
    "description",
    "job description",
    "role description",
    "responsibilities",
    "key responsibilities",
    "duties",
    "what you'll do",
    "what you will do",
    "requirements",
    "job requirements",
    "minimum requirements",
    "qualifications",
    "required qualifications",
    "preferred qualifications",
    "minimum qualifications",
    "desired qualifications",
    "who you are",
    "what we're looking for",
    "what we are looking for",
    "skills",
    "required skills",
    "technical skills",
    "nice to have",
    "preferred skills",
    "benefits",
    "perks",
    "compensation",
    "what we offer",
    "why join us",
    "location",
    "work arrangement",
    "experience",
    "education",
]

# ---------------------------------------------------------------------------
# Embedding constants
# ---------------------------------------------------------------------------

#: Sentinel dimension value indicating "auto-detect from model".
EMBEDDING_DIM_AUTO: int = -1

#: Maximum tokens passed to the embedding model in a single request.
EMBEDDING_MAX_TOKENS: int = 512

# ---------------------------------------------------------------------------
# Retrieval constants
# ---------------------------------------------------------------------------

#: Default number of results to return from a retrieval query.
DEFAULT_TOP_K: int = 10

#: Reciprocal Rank Fusion default k-parameter.
RRF_DEFAULT_K: int = 60

#: Maximum number of candidates passed to the re-ranker.
RERANKER_MAX_CANDIDATES: int = 100

#: Minimum similarity score to include a result (0.0 = no filtering).
MIN_SIMILARITY_SCORE: float = 0.0

# ---------------------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------------------

#: HNSW index build parameter – number of neighbours during construction.
HNSW_M: int = 16

#: HNSW index build parameter – ef during construction.
HNSW_EF_CONSTRUCT: int = 200

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

#: Maximum request body size in bytes (50 MB).
MAX_REQUEST_BODY_BYTES: int = 50 * 1024 * 1024

#: Default pagination page size.
DEFAULT_PAGE_SIZE: int = 20

#: Maximum pagination page size.
MAX_PAGE_SIZE: int = 100
