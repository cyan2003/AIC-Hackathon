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
        t0 = time.perf_counter()
        vector_store = QdrantVectorStore(
            settings.retrieval.vector_store,
            embedding_dimension=settings.ingestion.embedding.dimension,
        )
        jd_results = await vector_store.search_jds(
            resume_embedding,
            top_k=top_k * 3,  # Over-fetch for fusion
        )
        timings["vector_search"] = time.perf_counter() - t0

        # -- 5. BM25 lexical search (if index is populated) ---------------
        t0 = time.perf_counter()
        lexical_store = BM25LexicalStore(settings.retrieval.lexical_store)
        lexical_results = await lexical_store.search(
            cleaned_text[:500],  # Use first 500 chars as keyword query
            top_k=top_k * 3,
        )
        timings["lexical_search"] = time.perf_counter() - t0

        # -- 6. Fuse results (RRF) ----------------------------------------
        t0 = time.perf_counter()
        fuser = ReciprocalRankFusion(settings.retrieval.fusion)
        fused = fuser.fuse([jd_results, lexical_results])
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
                    matched_sections=sections,
                    metadata=r.metadata,
                )
            )

        return MatchResponse(
            results=results,
            total_candidates=total_candidates,
            timings=timings,
            resume_metadata=resume_metadata,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Matching failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during matching: {exc}",
        )
