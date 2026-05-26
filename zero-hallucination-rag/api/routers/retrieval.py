"""
Retrieval / matching API endpoints.

POST /match — Upload a resume → get top-10 matching job descriptions
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status

from api.models.retrieval import (
    MatchedSection,
    MatchRequest,
    MatchResponse,
    MatchResult,
)
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["matching"])


@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match resume to job descriptions",
    description=(
        "Upload a resume (PDF/DOCX) or provide resume text to find the "
        "top matching job descriptions from the indexed collection."
    ),
)
async def match_resume_to_jds(
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    top_k: int = Form(10),
    filters: Optional[str] = Form(None),
):
    """
    Match a resume against indexed job descriptions.

    Accepts either a file upload or plain text. The resume is:
    1. Parsed (if PDF/DOCX)
    2. Cleaned and embedded
    3. Searched against the JD collection
    4. Results are fused (RRF) and re-ranked
    5. Top-k matches returned with relevant sections
    """
    filters_dict = {}
    if filters:
        try:
            import json
            filters_dict = json.loads(filters)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format for 'filters' parameter: {exc}",
            )

    timings: dict[str, float] = {}

    # -- 1. Get resume text -----------------------------------------------
    t0 = time.perf_counter()
    if file and file.filename:
        try:
            from api.routers.ingestion import _read_upload, _extract_text

            content, filename, mime_type = await _read_upload(file)
            text = await _extract_text(content, mime_type)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse resume file: {exc}",
            )
    elif resume_text:
        text = resume_text
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a file upload or resume_text is required.",
        )
    timings["parse"] = time.perf_counter() - t0

    try:
        from core.config import get_settings
        from ingestion.processors.resume_metadata_extractor import ResumeMetadataExtractor
        from ingestion.processors.cleaner import DefaultTextCleaner
        from ingestion.embedders.factory import create_embedder
        from ingestion.interfaces import RawDocument
        from retrieval.fusion.reciprocal_rank import ReciprocalRankFusion
        from retrieval.fusion.reranker import CrossEncoderReranker
        from retrieval.query.processor import DefaultQueryProcessor
        from retrieval.vector_store.qdrant import QdrantVectorStore
        from retrieval.lexical_store.bm25 import BM25LexicalStore

        settings = get_settings()

        # -- 2. Extract resume metadata ----------------------------------
        t0 = time.perf_counter()
        doc = RawDocument(source="upload", content=text)
        extractor = ResumeMetadataExtractor()
        resume_metadata = extractor.extract(doc)
        timings["metadata_extraction"] = time.perf_counter() - t0

        # -- 3. Clean and embed the resume --------------------------------
        t0 = time.perf_counter()
        cleaner = DefaultTextCleaner()
        cleaned_text = cleaner.clean(text)
        timings["cleaning"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        embedder = create_embedder(settings.ingestion.embedding)
        resume_embedding = embedder.embed_single(cleaned_text)
        timings["embedding"] = time.perf_counter() - t0

        # -- 4. Search the JD collection ----------------------------------
        from api.dependencies.database import get_vector_store, get_lexical_store

        # -- 4. Search the JD collection ----------------------------------
        t0 = time.perf_counter()
        vector_store = get_vector_store()
        jd_results = await vector_store.search_jds(
            resume_embedding,
            top_k=top_k * 3,  # Over-fetch for fusion
            filters=filters_dict,
        )
        timings["vector_search"] = time.perf_counter() - t0

        # -- 5. BM25 lexical search (if index is populated) ---------------
        t0 = time.perf_counter()
        lexical_store = get_lexical_store()
        lexical_results = await lexical_store.search(
            cleaned_text[:500],  # Use first 500 chars as keyword query
            top_k=top_k * 3,
            filters=filters_dict,
        )
        timings["lexical_search"] = time.perf_counter() - t0

        # -- 6. Fuse results (RRF) ----------------------------------------
        t0 = time.perf_counter()
        fuser = ReciprocalRankFusion(settings.retrieval.fusion)
        fused = fuser.fuse({"vector": jd_results, "lexical": lexical_results})
        total_candidates = len(fused)
        timings["fusion"] = time.perf_counter() - t0

        # -- 7. Re-rank with cross-encoder --------------------------------
        if settings.retrieval.enable_reranking and fused:
            t0 = time.perf_counter()
            reranker = CrossEncoderReranker(
                model_name=settings.retrieval.fusion.reranker_model,
            )
            fused = await reranker.rerank(
                cleaned_text[:500],
                fused,
                top_k=top_k,
            )
            timings["reranking"] = time.perf_counter() - t0

        # -- 7.5 Calculate Source Confidence Scores and Filter -----------
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc)
        
        confidence_threshold = settings.retrieval.min_confidence_threshold
        max_days = settings.retrieval.max_freshness_days

        fused_with_confidence = []
        for r in fused:
            normalized_score = max(0.0, min(1.0, r.score))

            created_at_str = r.metadata.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                except Exception:
                    created_at = current_time
            else:
                created_at = current_time

            delta = current_time - created_at
            days_since_creation = max(0.0, delta.total_seconds() / 86400.0)
            freshness_score = max(0.0, 1.0 - (days_since_creation / max_days))

            try:
                trust_rating = float(r.metadata.get("trust_rating", 1.0))
            except (ValueError, TypeError):
                trust_rating = 1.0
            trust_rating = max(0.0, min(1.0, trust_rating))

            w_sim = settings.retrieval.weight_similarity
            w_fresh = settings.retrieval.weight_freshness
            w_trust = settings.retrieval.weight_trust

            confidence = (w_sim * normalized_score) + (w_fresh * freshness_score) + (w_trust * trust_rating)
            r.confidence_score = float(round(confidence, 4))

            if r.confidence_score >= confidence_threshold:
                fused_with_confidence.append(r)

        fused = fused_with_confidence

        # Determine max confidence score
        max_confidence = max([r.confidence_score for r in fused]) if fused else 0.0

        # If max confidence is less than the threshold, bypass LLM and return fallback
        if max_confidence < confidence_threshold:
            from agent.schemas import CandidateAssessment
            assessment = CandidateAssessment(
                overall_score=0.0,
                recommendation="No Match",
                summary="Insufficient evidence found: confidence score is below threshold.",
                strengths=[],
                gaps=[],
                cited_evidence=[]
            )
            return MatchResponse(
                results=[],
                total_candidates=total_candidates,
                timings=timings,
                resume_metadata=resume_metadata,
                assessment=assessment,
            )

        # -- 8. Build response -------------------------------------------
        results: list[MatchResult] = []
        for r in fused[:top_k]:
            sections = []
            if r.metadata.get("chunk_section"):
                sections.append(
                    MatchedSection(
                        section=r.metadata["chunk_section"],
                        text=r.text[:500],
                        score=r.score,
                    )
                )

            results.append(
                MatchResult(
                    document_id=r.document_id,
                    job_title=r.metadata.get("job_title"),
                    score=r.score,
                    confidence_score=r.confidence_score,
                    matched_sections=sections,
                    metadata=r.metadata,
                )
            )

        # -- 9. LLM Agent Assessment (graceful degradation) ---------------
        assessment = None
        try:
            from agent.assessor import CandidateAssessor

            assessor = CandidateAssessor(settings.agent)
            jd_dicts = [
                {
                    "job_title": r.job_title,
                    "score": r.score,
                    "text": r.metadata.get("text", ""),
                    "matched_sections": [
                        {"section": s.section, "text": s.text}
                        for s in r.matched_sections
                    ],
                    "metadata": r.metadata,
                }
                for r in results
            ]

            t0 = time.perf_counter()
            assessment = await assessor.assess(
                resume_text=cleaned_text,
                resume_metadata=resume_metadata,
                match_results=jd_dicts,
            )
            timings["assessment"] = time.perf_counter() - t0
        except Exception as exc:
            logger.warning("LLM assessment failed (returning results without it)", error=str(exc))

        cache_hit = False
        if assessment and getattr(assessment, "cache_hit", None) is not None:
            cache_hit = assessment.cache_hit

        return MatchResponse(
            results=results,
            total_candidates=total_candidates,
            timings=timings,
            resume_metadata=resume_metadata,
            assessment=assessment,
            llm_cache_hit=cache_hit,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Matching failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during matching: {exc}",
        )
