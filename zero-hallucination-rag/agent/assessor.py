"""
LLM-powered candidate assessor.

Uses the Chutes AI API (OpenAI-compatible) to generate structured
candidate assessments with citations, enforcing zero-hallucination.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import AgentConfig
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.schemas import CandidateAssessment, fallback_assessment
from utils.logging import get_logger

logger = get_logger(__name__)


class CandidateAssessor:
    """
    LLM-powered candidate assessment agent.

    Calls the Chutes AI API (via OpenAI SDK) to evaluate how well
    a candidate's resume matches retrieved job descriptions.

    Parameters
    ----------
    config:
        Agent configuration with API key, model, and parameters.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = None

    def _get_client(self):
        """Lazy-initialise the OpenAI client pointing at Chutes AI."""
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout,
            )
            return self._client
        except ImportError:
            raise RuntimeError(
                "openai package is required. Install with: pip install openai"
            )

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """
        Strip <think>...</think> reasoning blocks from DeepSeek R1 output.

        DeepSeek R1 models emit chain-of-thought inside <think> tags
        before providing the final answer.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from LLM output that may contain markdown fences
        or other wrapper text.
        """
        # Try to find JSON inside ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try to find raw JSON object
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        return text.strip()

    async def assess(
        self,
        resume_text: str,
        resume_metadata: dict[str, Any],
        match_results: list[dict[str, Any]],
    ) -> CandidateAssessment:
        """
        Generate a structured candidate assessment using the LLM.

        Parameters
        ----------
        resume_text:
            The cleaned resume text content.
        resume_metadata:
            Metadata extracted from the resume (skills, name, etc.).
        match_results:
            List of matched JD results from the retrieval pipeline.

        Returns
        -------
        CandidateAssessment
            Structured assessment with scores, strengths, gaps, and citations.
            Returns a fallback assessment if the LLM call fails.
        """
        if not self._config.api_key:
            logger.warning("No Chutes API key configured — skipping LLM assessment")
            return fallback_assessment("No LLM API key configured. Set CHUTES_API_KEY in .env")

        try:
            client = self._get_client()
            user_prompt = build_user_prompt(resume_text, resume_metadata, match_results)

            logger.info(
                "Calling LLM for candidate assessment",
                model=self._config.model,
            )

            response = await client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            raw_output = response.choices[0].message.content or ""
            logger.debug("Raw LLM output length", length=len(raw_output))

            # Strip DeepSeek R1 thinking blocks
            cleaned = self._strip_think_blocks(raw_output)

            # Extract JSON from potential markdown fences
            json_str = self._extract_json(cleaned)

            # Parse into Pydantic model
            data = json.loads(json_str)
            assessment = CandidateAssessment(**data)

            logger.info(
                "LLM assessment complete",
                score=assessment.overall_score,
                recommendation=assessment.recommendation,
                evidence_count=len(assessment.cited_evidence),
            )

            return assessment

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON output", error=str(exc))
            return fallback_assessment(f"LLM returned invalid JSON: {exc}")

        except Exception as exc:
            logger.error("LLM assessment failed", error=str(exc))
            return fallback_assessment(f"LLM assessment error: {exc}")
