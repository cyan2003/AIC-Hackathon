"""
Pydantic models for the LLM agent's structured output.

The CandidateAssessment schema enforces zero-hallucination by requiring
cited_evidence for every claim the LLM makes about a candidate.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CitedEvidence(BaseModel):
    """A piece of evidence cited from the source documents."""

    source_section: str = Field(
        ...,
        description="Section label the evidence comes from (e.g. 'TECHNICAL SKILLS', 'REQUIREMENTS')",
    )
    source_text: str = Field(
        ...,
        description="Exact quote from the source document chunk",
    )
    relevance: str = Field(
        ...,
        description="Why this evidence supports the assessment claim",
    )


class CandidateAssessment(BaseModel):
    """
    Structured LLM assessment of a candidate against matched job descriptions.

    Every field is designed to be directly renderable in the Streamlit UI.
    """

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall match score between 0.0 and 1.0",
    )
    recommendation: str = Field(
        ...,
        description="One of: Strong Match, Good Match, Weak Match, No Match",
    )
    summary: str = Field(
        ...,
        description="2-3 sentence executive summary of the candidate's fit",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Key strengths of the candidate relative to the matched JDs",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Gaps or missing qualifications relative to the matched JDs",
    )
    cited_evidence: list[CitedEvidence] = Field(
        default_factory=list,
        description="Evidence from source documents supporting the assessment",
    )
    cache_hit: Optional[bool] = Field(
        None,
        description="Internal cache hit indicator",
    )


def fallback_assessment(reason: str = "Assessment unavailable") -> CandidateAssessment:
    """Return a safe fallback assessment when the LLM is unreachable."""
    return CandidateAssessment(
        overall_score=0.0,
        recommendation="Assessment Unavailable",
        summary=reason,
        strengths=[],
        gaps=[],
        cited_evidence=[],
    )
