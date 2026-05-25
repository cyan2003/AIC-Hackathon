"""
Resume-specific metadata extraction.

Uses regex heuristics for simple fields (candidate name, email) and
defines an LLM interface for complex fields (skills, experience level).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from ingestion.interfaces import MetadataExtractor, RawDocument
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM interface (to be wired later)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for an LLM client used in metadata extraction.

    Implement this protocol with any LLM provider (NVIDIA NIM, OpenAI, etc.)
    and pass it to :class:`LLMResumeMetadataExtractor`.
    """

    def complete(self, prompt: str, *, system: str = "") -> str:
        """Return the LLM completion as a string."""
        ...


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)
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

# Section header pattern for finding the Skills section
_SKILLS_HEADER = re.compile(
    r"(?:^|\n)\s*(?:skills|technical skills|core competencies|key skills|competencies)\s*[:\-–—]?\s*\n",
    re.IGNORECASE,
)


def _extract_candidate_name(text: str) -> Optional[str]:
    """
    Heuristic: the candidate's name is typically the first non-empty,
    non-email, non-phone line of a resume.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        # Skip lines that look like emails, phone numbers, or URLs
        if _EMAIL_PATTERN.search(line):
            continue
        if _PHONE_PATTERN.search(line):
            continue
        if line.startswith("http"):
            continue
        # Skip very long lines (likely paragraphs, not names)
        if len(line) > 60:
            continue
        # Name-like: mostly alpha characters with spaces
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio > 0.8:
            return line
    return None


def _extract_skills_from_section(text: str) -> list[str]:
    """
    Find the Skills section and parse comma/bullet-separated items.
    """
    match = _SKILLS_HEADER.search(text)
    if not match:
        return []

    # Get text after the Skills header until next section or end.
    start = match.end()
    # Find next section header (line that looks like a heading)
    next_header = re.search(r"\n\s*[A-Z][a-zA-Z\s]+[:\-–—]\s*\n", text[start:])
    end = start + next_header.start() if next_header else len(text)
    skills_text = text[start:end]

    # Split by common delimiters: comma, bullet, pipe, semicolon, newline
    items = re.split(r"[,;|•·▪►\n]+", skills_text)
    skills = []
    for item in items:
        cleaned = item.strip().strip("-–—•*").strip()
        if cleaned and 2 <= len(cleaned) <= 50:
            skills.append(cleaned)
    return skills


def _extract_experience_years(text: str) -> Optional[int]:
    """Extract years of experience from patterns like '5+ years experience'."""
    matches = _YEARS_EXP_PATTERN.findall(text)
    if matches:
        return max(int(m) for m in matches)
    return None


def _extract_education_level(text: str) -> Optional[str]:
    """Return the highest detected education level."""
    for level, pattern in _DEGREE_PATTERNS.items():
        if pattern.search(text):
            return level
    return None


def _extract_job_title(text: str) -> Optional[str]:
    """
    Heuristic: look for a title-like line near the top of the resume
    (after the name), or the first job title in the Experience section.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Check lines 2-5 for a short, title-like line
    for line in lines[1:6]:
        if 3 <= len(line) <= 60 and not _EMAIL_PATTERN.search(line):
            # Looks like a title if it doesn't have too many special characters
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(
                len(line), 1
            )
            if alpha_ratio > 0.7:
                return line
    return None


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------


class ResumeMetadataExtractor(MetadataExtractor):
    """
    Regex-based metadata extractor for resume documents.

    Extracts: candidate_name, email, phone, skills, experience_years,
    education_level, job_title, file_path, source_file_type, embedding_date.
    """

    def extract(self, document: RawDocument) -> dict[str, Any]:
        text = (
            document.content
            if isinstance(document.content, str)
            else document.content.decode("utf-8", errors="replace")
        )

        metadata: dict[str, Any] = {
            "document_type": "resume",
            "file_path": document.source,
            "source_file_type": Path(document.source).suffix.lower(),
            "embedding_date": datetime.now(timezone.utc).isoformat(),
        }

        # Candidate name
        name = _extract_candidate_name(text)
        if name:
            metadata["candidate_name"] = name

        # Email
        email_match = _EMAIL_PATTERN.search(text)
        if email_match:
            metadata["email"] = email_match.group()

        # Phone
        phone_match = _PHONE_PATTERN.search(text)
        if phone_match:
            metadata["phone"] = phone_match.group()

        # Skills
        skills = _extract_skills_from_section(text)
        if skills:
            metadata["skills"] = skills

        # Experience years
        exp = _extract_experience_years(text)
        if exp is not None:
            metadata["experience_years"] = exp

        # Education level
        edu = _extract_education_level(text)
        if edu:
            metadata["education_level"] = edu

        # Job title
        title = _extract_job_title(text)
        if title:
            metadata["job_title"] = title

        return metadata


class LLMResumeMetadataExtractor(MetadataExtractor):
    """
    LLM-backed metadata extractor for resumes.

    Uses a provided :class:`LLMClient` to extract structured metadata
    from resume text via prompted JSON output.

    .. note::
        The LLM provider is not wired yet. Pass any implementation of
        the :class:`LLMClient` protocol to use this extractor.

    Parameters
    ----------
    llm_client:
        An LLM client implementing the ``complete`` method.
    fallback:
        Optional regex-based extractor to use as fallback on LLM failure.
    """

    _SYSTEM_PROMPT = (
        "You are a resume parsing assistant. Extract structured metadata "
        "from the following resume text. Return ONLY valid JSON with these fields: "
        "candidate_name, job_title, skills (list of strings), "
        "experience_years (integer), education_level (one of: phd, master, bachelor, associate), "
        "industry (string). If a field cannot be determined, set it to null."
    )

    def __init__(
        self,
        llm_client: LLMClient,
        fallback: Optional[MetadataExtractor] = None,
    ) -> None:
        self._llm = llm_client
        self._fallback = fallback or ResumeMetadataExtractor()

    def extract(self, document: RawDocument) -> dict[str, Any]:
        text = (
            document.content
            if isinstance(document.content, str)
            else document.content.decode("utf-8", errors="replace")
        )

        # Always get regex-based fields first
        base_metadata = self._fallback.extract(document)

        try:
            import json

            # Truncate for LLM context window
            truncated = text[:4000]
            response = self._llm.complete(
                f"Resume text:\n\n{truncated}",
                system=self._SYSTEM_PROMPT,
            )

            # Parse JSON from response
            llm_data = json.loads(response)

            # Merge LLM results (LLM takes precedence for complex fields)
            if llm_data.get("candidate_name"):
                base_metadata["candidate_name"] = llm_data["candidate_name"]
            if llm_data.get("skills"):
                base_metadata["skills"] = llm_data["skills"]
            if llm_data.get("experience_years") is not None:
                base_metadata["experience_years"] = llm_data["experience_years"]
            if llm_data.get("education_level"):
                base_metadata["education_level"] = llm_data["education_level"]
            if llm_data.get("job_title"):
                base_metadata["job_title"] = llm_data["job_title"]
            if llm_data.get("industry"):
                base_metadata["industry"] = llm_data["industry"]

        except Exception as exc:
            logger.warning(
                "LLM metadata extraction failed, using regex fallback",
                error=str(exc),
            )

        return base_metadata
