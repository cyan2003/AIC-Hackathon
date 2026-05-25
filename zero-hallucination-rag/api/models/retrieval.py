"""
Pydantic models for retrieval / matching request/response payloads.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MatchFilters(BaseModel):
    """Optional filters to narrow job matching results."""

    skills: Optional[list[str]] = Field(
        None, description="Required skills the JD must mention"
    )
    experience_years_max: Optional[int] = Field(
        None, description="Maximum years of experience required by the JD"
    )
    education_level: Optional[str] = Field(
        None, description="Minimum education level (associate, bachelor, master, phd)"
    )
    industry: Optional[str] = Field(
        None, description="Target industry"
    )


class MatchRequest(BaseModel):
    """
    Request body for the ``POST /match`` endpoint.

    Either ``resume_text`` or a file upload (handled by the router) is required.
    """

    resume_text: Optional[str] = Field(
        None,
        description="Plain-text resume content (alternative to file upload)",
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of JDs to return")
    filters: Optional[MatchFilters] = Field(
        None, description="Optional metadata filters"
    )


class MatchedSection(BaseModel):
    """A section of the matched JD that is relevant to the resume."""

    section: str = Field(..., description="Section label (e.g. 'requirements')")
    text: str = Field(..., description="Section text content")
    score: float = Field(..., description="Relevance score for this section")


class MatchResult(BaseModel):
    """A single matched job description."""

    document_id: str = Field(..., description="JD document identifier")
    job_title: Optional[str] = Field(None, description="Extracted job title")
    score: float = Field(..., description="Overall relevance score")
    matched_sections: list[MatchedSection] = Field(
        default_factory=list,
        description="Relevant sections from the JD",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Full JD metadata",
    )


class MatchResponse(BaseModel):
    """Response from the ``POST /match`` endpoint."""

    results: list[MatchResult] = Field(
        default_factory=list,
        description="Ranked list of matching job descriptions",
    )
    total_candidates: int = Field(
        0, description="Number of candidates before final truncation"
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage latencies in seconds",
    )
    resume_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata extracted from the uploaded resume",
    )
