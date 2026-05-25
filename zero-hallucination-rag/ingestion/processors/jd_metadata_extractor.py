"""
Job description-specific metadata extraction.

Uses regex heuristics to extract structured fields from job description
text: job title, required skills, experience, education, industry.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ingestion.interfaces import MetadataExtractor, RawDocument
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_YEARS_EXP_PATTERN = re.compile(
    r"(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)

_DEGREE_PATTERNS: dict[str, re.Pattern] = {
    "phd": re.compile(r"\b(?:ph\.?d|doctor(?:ate)?)\b", re.IGNORECASE),
    "master": re.compile(
        r"\b(?:master(?:'?s)?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|m\.?eng)\b",
        re.IGNORECASE,
    ),
    "bachelor": re.compile(
        r"\b(?:bachelor(?:'?s)?|b\.?s\.?|b\.?a\.?|b\.?eng|b\.?sc)\b",
        re.IGNORECASE,
    ),
    "associate": re.compile(r"\b(?:associate(?:'?s)?|a\.?s\.?|a\.?a\.?)\b", re.IGNORECASE),
}

# Section headers that typically precede requirements / skills
_REQUIREMENTS_HEADER = re.compile(
    r"(?:^|\n)\s*(?:requirements|qualifications|required skills|"
    r"what we(?:'re| are) looking for|who you are|skills)\s*[:\-–—]?\s*\n",
    re.IGNORECASE,
)

# Industry keyword patterns
_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "technology": ["software", "saas", "tech", "IT", "engineering", "developer", "cloud"],
    "finance": ["finance", "banking", "fintech", "investment", "trading", "accounting"],
    "healthcare": ["healthcare", "medical", "clinical", "health", "pharma", "biotech"],
    "education": ["education", "teaching", "academic", "university", "school"],
    "retail": ["retail", "e-commerce", "ecommerce", "shopping", "consumer"],
    "manufacturing": ["manufacturing", "production", "supply chain", "logistics"],
    "consulting": ["consulting", "advisory", "strategy"],
    "marketing": ["marketing", "advertising", "digital marketing", "brand"],
}


def _extract_job_title(text: str) -> Optional[str]:
    """
    Heuristic: the job title is typically the first prominent heading
    or the first short, title-cased line.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        # Skip very long lines
        if len(line) > 80:
            continue
        # Skip lines that look like company descriptions
        if any(word in line.lower() for word in ["about us", "about the company", "overview"]):
            continue
        # A good title candidate: short, mostly alpha
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio > 0.7 and 3 <= len(line) <= 80:
            return line
    return None


def _extract_required_skills(text: str) -> list[str]:
    """
    Find the Requirements/Qualifications section and extract skill-like items.
    """
    match = _REQUIREMENTS_HEADER.search(text)
    if not match:
        return []

    start = match.end()
    # Find next section header
    next_header = re.search(r"\n\s*[A-Z][a-zA-Z\s]+[:\-–—]\s*\n", text[start:])
    end = start + next_header.start() if next_header else min(start + 2000, len(text))
    section_text = text[start:end]

    # Extract bullet-pointed items
    items = re.split(r"[•·▪►\n]+", section_text)
    skills: list[str] = []
    for item in items:
        cleaned = item.strip().strip("-–—*").strip()
        if cleaned and 3 <= len(cleaned) <= 100:
            skills.append(cleaned)
    return skills


def _extract_experience_years(text: str) -> Optional[int]:
    """Extract minimum years of experience required."""
    matches = _YEARS_EXP_PATTERN.findall(text)
    if matches:
        # Return the minimum requirement (usually the first match)
        return min(int(m) for m in matches)
    return None


def _extract_education_level(text: str) -> Optional[str]:
    """Return the minimum education level required."""
    # Check in order from highest to lowest
    for level, pattern in _DEGREE_PATTERNS.items():
        if pattern.search(text):
            return level
    return None


def _detect_industry(text: str) -> Optional[str]:
    """Simple keyword-based industry detection."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[industry] = score

    if scores:
        return max(scores, key=scores.get)
    return None


# ---------------------------------------------------------------------------
# Concrete extractor
# ---------------------------------------------------------------------------


class JDMetadataExtractor(MetadataExtractor):
    """
    Regex-based metadata extractor for job description documents.

    Extracts: job_title, required_skills, experience_years,
    education_level, industry, file_path, source_file_type, embedding_date.
    """

    def extract(self, document: RawDocument) -> dict[str, Any]:
        text = (
            document.content
            if isinstance(document.content, str)
            else document.content.decode("utf-8", errors="replace")
        )

        metadata: dict[str, Any] = {
            "document_type": "job_description",
            "file_path": document.source,
            "source_file_type": Path(document.source).suffix.lower(),
            "embedding_date": datetime.now(timezone.utc).isoformat(),
        }

        # Job title
        title = _extract_job_title(text)
        if title:
            metadata["job_title"] = title

        # Required skills
        skills = _extract_required_skills(text)
        if skills:
            metadata["required_skills"] = skills

        # Experience years
        exp = _extract_experience_years(text)
        if exp is not None:
            metadata["experience_years"] = exp

        # Education level
        edu = _extract_education_level(text)
        if edu:
            metadata["education_level"] = edu

        # Industry
        industry = _detect_industry(text)
        if industry:
            metadata["industry"] = industry

        return metadata
