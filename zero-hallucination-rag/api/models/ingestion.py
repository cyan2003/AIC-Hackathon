"""
Pydantic models for ingestion request/response payloads.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ResumeIngestionResponse(BaseModel):
    """Response after successfully ingesting a resume."""

    document_id: str = Field(..., description="Unique identifier for the ingested resume")
    source: str = Field(..., description="Original filename or path")
    chunks_created: int = Field(..., description="Number of section chunks created")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted metadata (candidate_name, skills, etc.)",
    )
    message: str = "Resume ingested successfully"


class JDIngestionResponse(BaseModel):
    """Response after successfully ingesting a job description."""

    document_id: str = Field(..., description="Unique identifier for the ingested JD")
    source: str = Field(..., description="Original filename or path")
    chunks_created: int = Field(..., description="Number of section chunks created")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted metadata (job_title, required_skills, etc.)",
    )
    message: str = "Job description ingested successfully"


class IngestionErrorResponse(BaseModel):
    """Error response when ingestion fails."""

    source: str
    error: str
    message: str = "Ingestion failed"
