"""
Prompt templates for the LLM-powered candidate assessor.

The system prompt enforces structured JSON output and zero-hallucination
by requiring every claim to cite evidence from the provided documents.
"""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """\
You are an expert senior technical recruiter AI. Your task is to assess how \
well a candidate's resume matches against retrieved job descriptions.

RULES — you MUST follow these strictly:
1. Output ONLY valid JSON matching the schema below. No markdown, no commentary.
2. Every strength or gap you mention MUST be backed by cited_evidence from the \
   source documents provided. Do NOT fabricate or infer information not present \
   in the provided text.
3. The overall_score must be between 0.0 (no match) and 1.0 (perfect match).
4. recommendation must be exactly one of: "Strong Match", "Good Match", \
   "Weak Match", "No Match".
5. Keep the summary to 2-3 sentences.
6. Each cited_evidence entry must include the exact source_text quote from the \
   provided documents.

OUTPUT JSON SCHEMA:
{
  "overall_score": <float 0.0-1.0>,
  "recommendation": "<Strong Match|Good Match|Weak Match|No Match>",
  "summary": "<2-3 sentence executive summary>",
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "gaps": ["<gap 1>", "<gap 2>", ...],
  "cited_evidence": [
    {
      "source_section": "<section label>",
      "source_text": "<exact quote from source>",
      "relevance": "<why this evidence matters>"
    }
  ]
}
"""


def build_user_prompt(
    resume_text: str,
    resume_metadata: dict[str, Any],
    matched_jds: list[dict[str, Any]],
) -> str:
    """
    Build the user prompt containing the resume and matched JD data.

    Parameters
    ----------
    resume_text:
        The cleaned resume text.
    resume_metadata:
        Extracted metadata from the resume (name, skills, experience, etc.).
    matched_jds:
        List of matched JD results, each containing at minimum
        ``job_title``, ``score``, ``text``, and ``metadata``.
    """
    # Format resume section
    meta_lines = []
    for key, value in resume_metadata.items():
        if value and key not in ("document_type",):
            meta_lines.append(f"  {key}: {value}")
    meta_block = "\n".join(meta_lines) if meta_lines else "  (no metadata extracted)"

    # Format matched JDs
    jd_blocks = []
    for i, jd in enumerate(matched_jds, 1):
        title = jd.get("job_title") or jd.get("metadata", {}).get("job_title", "Unknown")
        score = jd.get("score", 0.0)
        text = jd.get("text", "")
        sections = jd.get("matched_sections", [])

        jd_text = f"--- JD #{i}: {title} (retrieval score: {score:.3f}) ---\n"
        if sections:
            for sec in sections:
                jd_text += f"[{sec.get('section', 'unknown')}]\n{sec.get('text', '')}\n\n"
        elif text:
            jd_text += text[:2000] + "\n"

        # Add JD metadata
        jd_meta = jd.get("metadata", {})
        if jd_meta.get("required_skills"):
            jd_text += f"Required Skills: {jd_meta['required_skills']}\n"
        if jd_meta.get("experience_required"):
            jd_text += f"Experience Required: {jd_meta['experience_required']}\n"

        jd_blocks.append(jd_text)

    jds_section = "\n".join(jd_blocks) if jd_blocks else "(no matching JDs found)"

    return f"""\
=== CANDIDATE RESUME ===
{resume_text[:3000]}

=== EXTRACTED RESUME METADATA ===
{meta_block}

=== TOP MATCHING JOB DESCRIPTIONS ===
{jds_section}

Assess how well this candidate matches the job descriptions above. \
Respond with the JSON assessment only.
"""
