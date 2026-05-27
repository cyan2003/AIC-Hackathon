"""
Streamlit UI for the Zero-Hallucination Recruiter AI.

Connects to the FastAPI backend for real document ingestion,
hybrid retrieval matching, and LLM-powered candidate assessments.
Also embeds Arize Phoenix for live telemetry and observability.
"""

import streamlit as st
import httpx
import os
import json
import pandas as pd
import altair as alt
import psutil
import asyncio
import time
from datetime import datetime, timezone
from streamlit_extras.metric_cards import style_metric_cards

# ---------------------------------------------------------------------------
# Phoenix Telemetry Initialization
# ---------------------------------------------------------------------------
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(skip_dep_check=True)
except Exception as e:
    pass

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zero-Hallucination Recruiter AI",
    page_icon="💼",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Assessment card */
    .assessment-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-strong { background: rgba(34,197,94,0.2); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .score-good { background: rgba(59,130,246,0.2); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
    .score-weak { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .score-none { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

    /* Evidence card */
    .evidence-card {
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Timing pills */
    .timing-pill {
        display: inline-block;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        margin: 0.15rem;
        font-size: 0.75rem;
        color: #a5b4fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=64,
    )
    st.title("⚙️ Run Settings")

    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://localhost:8000",
        help="The base URL of the running FastAPI server",
    )

    st.divider()

    st.header("📋 Document Parameters")
    doc_trust = st.slider(
        "Ingestion Trust Rating",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Trust score to assign to new documents (1.0 = fully verified source)",
    )
    
    custom_date = st.checkbox("Set Custom Creation Date", value=False)
    created_at_str = None
    if custom_date:
        created_date = st.date_input("Document Created Date", value=datetime.now())
        created_time = st.time_input("Document Created Time", value=datetime.now().time())
        created_at_str = datetime.combine(created_date, created_time).isoformat()

    st.divider()

    st.header("🔍 Match Controls")
    top_k = st.slider("Top-K Matches", min_value=1, max_value=20, value=5)

    st.markdown("### 🎯 Advanced Filters")
    enable_filters = st.checkbox("Enable Metadata Filtering", value=False)
    
    filters_dict = {}
    if enable_filters:
        filter_ind = st.selectbox(
            "Filter by Industry",
            options=["", "Technology", "Finance", "Healthcare", "Education", "Retail", "Manufacturing", "Consulting", "Marketing"],
            index=0
        )
        if filter_ind:
            filters_dict["industry"] = filter_ind.lower()
            
        filter_exp = st.number_input(
            "Max Experience Required (Years)",
            min_value=0,
            max_value=30,
            value=5,
            step=1
        )
        filters_dict["experience_years_max"] = filter_exp
        
        filter_edu = st.selectbox(
            "Candidate Degree Level",
            options=["", "Associate", "Bachelor", "Master", "PhD"],
            index=0
        )
        if filter_edu:
            filters_dict["education_level"] = filter_edu.lower()
            
        filter_skills = st.text_input(
            "Candidate Skills (comma separated)",
            placeholder="e.g. Python, Docker, SQL"
        )
        if filter_skills:
            filters_dict["skills"] = [s.strip() for s in filter_skills.split(",") if s.strip()]

    process_btn = st.button("🚀 Run Recruiter Agent", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _recommendation_badge(rec: str) -> str:
    """Return an HTML badge for the recommendation level."""
    css = "score-none"
    if "Strong" in rec:
        css = "score-strong"
    elif "Good" in rec:
        css = "score-good"
    elif "Weak" in rec:
        css = "score-weak"
    return f'<span class="score-badge {css}">{rec}</span>'


async def ingest_file(client: httpx.AsyncClient, endpoint: str, file, trust_rating: float, created_at: str = None) -> dict:
    """Upload a single file to an ingestion endpoint with trust/freshness metadata."""
    content = file.read()
    file.seek(0)
    
    data = {"trust_rating": str(trust_rating)}
    if created_at:
        data["created_at"] = created_at

    response = await client.post(
        endpoint,
        files={"file": (file.name, content, "application/octet-stream")},
        data=data,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


async def match_resume(client: httpx.AsyncClient, file, top_k: int, filters: dict = None) -> dict:
    """Send a resume file to the /match endpoint."""
    content = file.read()
    file.seek(0)
    data = {"top_k": str(top_k)}
    if filters:
        data["filters"] = json.dumps(filters)
    response = await client.post(
        "/match",
        files={"file": (file.name, content, "application/octet-stream")},
        data=data,
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()

# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------
tab_match, tab_ingest, tab_telemetry = st.tabs([
    "🎯 Match Dashboard",
    "📂 Ingestion Center",
    "📊 System Telemetry",
])

# ---------------------------------------------------------------------------
# Tab 1: Match Dashboard
# ---------------------------------------------------------------------------
with tab_match:
    st.title("💼 AI Recruitment Agent")
    st.caption("Zero-Hallucination Resume Screening & Hybrid Retrieval — Powered by DeepSeek R1")

    uploaded_resumes = st.file_uploader(
        "Upload Candidate Resumes (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    uploaded_jds = st.file_uploader(
        "Upload Job Descriptions to Index (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="jd_uploader",
    )

    st.divider()

    if process_btn:
        if not uploaded_resumes:
            st.warning("⚠️ Please upload at least one resume to proceed.")
        else:
            async def run_pipeline():
                async with httpx.AsyncClient(base_url=api_url) as client:
                    # Check backend health
                    try:
                        health = await client.get("/health", timeout=5.0)
                        health.raise_for_status()
                    except Exception as e:
                        st.error(f"❌ Cannot reach backend at `{api_url}`. Is the FastAPI server running?\n\n`{e}`")
                        return

                    # Ingest JDs if provided
                    if uploaded_jds:
                        with st.status("📋 Ingesting Job Descriptions...", expanded=True) as jd_status:
                            for jd_file in uploaded_jds:
                                try:
                                    result = await ingest_file(
                                        client, "/ingest/jd", jd_file, doc_trust, created_at_str
                                    )
                                    st.write(f"✅ **{jd_file.name}** — {result.get('chunks_created', 0)} chunks, Trust: `{doc_trust}`, ID: `{result.get('document_id', 'N/A')[:12]}...`")
                                except Exception as e:
                                    st.write(f"❌ **{jd_file.name}** — {e}")
                            jd_status.update(label=f"✅ {len(uploaded_jds)} JDs ingested", state="complete")

                    # Process each resume
                    for resume_file in uploaded_resumes:
                        st.subheader(f"📄 Results for: {resume_file.name}")

                        with st.status("🔄 Processing resume...", expanded=True) as resume_status:
                            st.write("🔄 Parsing and embedding resume...")
                            st.write("🔍 Running Hybrid Retrieval (Vector + BM25) & Confidence Scoring...")
                            st.write("🤖 Generating LLM Assessment...")

                            try:
                                result = await match_resume(client, resume_file, top_k, filters=filters_dict if enable_filters else None)
                                resume_status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                            except Exception as e:
                                resume_status.update(label="❌ Analysis Failed", state="error")
                                st.error(f"Matching failed: {e}")
                                continue

                        # Display Results
                        matches = result.get("results", [])
                        assessment = result.get("assessment")
                        timings = result.get("timings", {})
                        resume_meta = result.get("resume_metadata", {})

                        # Check if fallback triggered
                        is_fallback = (
                            assessment is not None and 
                            "Insufficient evidence found" in assessment.get("summary", "")
                        )

                        if is_fallback:
                            st.error("⚠️ **Hallucination Guardrail Triggered:** Insufficient confidence in retrieved context.")
                            st.info("The match scores or trust ratings for the indexed job descriptions were below the minimum confidence threshold. Bypassed LLM assessor to guarantee zero hallucination.")

                        if result.get("llm_cache_hit"):
                            st.success("⚡ **Cache Hit**: LLM assessment retrieved from persistent SQLite cache (0 tokens consumed, response returned instantly).")

                        # Timings bar
                        if timings:
                            timing_html = " ".join(
                                f'<span class="timing-pill">{k}: {v*1000:.0f}ms</span>'
                                for k, v in timings.items()
                            )
                            st.markdown(timing_html, unsafe_allow_html=True)

                        # Resume metadata
                        if resume_meta:
                            with st.expander("📋 Extracted Resume Metadata", expanded=False):
                                for k, v in resume_meta.items():
                                    if v and k != "document_type":
                                        st.write(f"**{k.replace('_', ' ').title()}:** {v}")

                        # LLM Assessment Card
                        if assessment and not is_fallback and assessment.get("recommendation") != "Assessment Unavailable":
                            st.markdown("### 🤖 AI Assessment")
                            score = assessment.get("overall_score", 0)
                            rec = assessment.get("recommendation", "Unknown")

                            a_col1, a_col2 = st.columns([1, 3])
                            with a_col1:
                                score_pct = int(score * 100)
                                score_color = "#22c55e" if score >= 0.7 else "#3b82f6" if score >= 0.5 else "#f59e0b" if score >= 0.3 else "#ef4444"
                                st.markdown(
                                    f"""
                                    <div style="text-align:center; padding:1rem;">
                                        <div style="font-size:3rem; font-weight:700; color:{score_color};">{score_pct}%</div>
                                        <div>{_recommendation_badge(rec)}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            with a_col2:
                                st.markdown(f"**Summary:** {assessment.get('summary', 'N/A')}")

                                strengths = assessment.get("strengths", [])
                                gaps = assessment.get("gaps", [])

                                s_col, g_col = st.columns(2)
                                with s_col:
                                    st.markdown("**✅ Strengths**")
                                    for s in strengths:
                                        st.markdown(f"- {s}")
                                with g_col:
                                    st.markdown("**⚠️ Gaps**")
                                    for g in gaps:
                                        st.markdown(f"- {g}")

                            # Cited Evidence
                            evidence = assessment.get("cited_evidence", [])
                            if evidence:
                                with st.expander(f"📎 Cited Evidence ({len(evidence)} citations)", expanded=False):
                                    for ev in evidence:
                                        st.markdown(
                                            f"""<div class="evidence-card">
                                            <strong>[{ev.get('source_section', 'N/A')}]</strong><br/>
                                            <em>"{ev.get('source_text', '')}"</em><br/>
                                            <small>→ {ev.get('relevance', '')}</small>
                                            </div>""",
                                            unsafe_allow_html=True,
                                        )

                        # Matched JDs table
                        if matches:
                            st.markdown("### 📊 Matched Job Descriptions")

                            df = pd.DataFrame([
                                {
                                    "Rank": i + 1,
                                    "Job Title": m.get("job_title") or "Unknown",
                                    "Similarity Score": round(m.get("score", 0), 4),
                                    "Confidence Score": f"{round(m.get('confidence_score', 0) * 100, 1)}%" if m.get("confidence_score") is not None else "N/A",
                                    "Doc ID": m.get("document_id", "")[:16] + "...",
                                }
                                for i, m in enumerate(matches)
                            ])
                            st.dataframe(df, use_container_width=True, hide_index=True)

                            # Score chart
                            chart = (
                                alt.Chart(df)
                                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                                .encode(
                                    x=alt.X("Job Title:N", sort="-y", title="Job Description"),
                                    y=alt.Y("Similarity Score:Q", title="Match Score", scale=alt.Scale(domain=[0, 1])),
                                    color=alt.Color(
                                        "Similarity Score:Q",
                                        scale=alt.Scale(scheme="viridis"),
                                        legend=None,
                                    ),
                                    tooltip=["Rank", "Job Title", "Similarity Score", "Confidence Score", "Doc ID"],
                                )
                                .properties(height=250)
                            )
                            st.altair_chart(chart, use_container_width=True)

                            # Expandable sections per match
                            for i, m in enumerate(matches):
                                sections = m.get("matched_sections", [])
                                if sections:
                                    conf_percent = f"{round(m.get('confidence_score', 0) * 100, 1)}%" if m.get('confidence_score') is not None else "N/A"
                                    with st.expander(f"📑 #{i+1} {m.get('job_title', 'Unknown')} — Matched Chunks (Confidence: {conf_percent})"):
                                        for sec in sections:
                                            st.markdown(f"**[{sec.get('section', '')}]** (similarity score: {sec.get('score', 0):.3f})")
                                            st.text(sec.get("text", ""))
                        else:
                            if not is_fallback:
                                st.info("No matching job descriptions found. Try ingesting some JDs first.")

                        st.divider()

            asyncio.run(run_pipeline())
    else:
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 2rem; opacity:0.6;">
                <p style="font-size:3rem;">💼</p>
                <h3>Upload resumes and job descriptions to get started</h3>
                <p>The agent will match candidates to jobs using hybrid retrieval<br/>
                and generate AI-powered assessments with confidence checking.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab 2: Ingestion Center
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingestion Center")
    st.caption("Upload documents individually to configure specific Trust and Date parameters.")

    ingest_type = st.radio("Ingestion Type", ["Job Description", "Resume"])
    uploaded_ingest_files = st.file_uploader(
        f"Upload {ingest_type}s (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="direct_uploader",
    )

    if uploaded_ingest_files:
        if st.button("🚀 Ingest Documents", key="direct_ingest_btn"):
            async def run_direct_ingest():
                async with httpx.AsyncClient(base_url=api_url) as client:
                    endpoint = "/ingest/jd" if ingest_type == "Job Description" else "/ingest/resume"
                    for file in uploaded_ingest_files:
                        try:
                            result = await ingest_file(
                                client, endpoint, file, doc_trust, created_at_str
                            )
                            st.success(f"Successfully Ingested: **{file.name}**")
                            st.json(result)
                        except Exception as e:
                            st.error(f"Failed to Ingest **{file.name}**: {e}")

            asyncio.run(run_direct_ingest())

# ---------------------------------------------------------------------------
# Tab 3: System Telemetry
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.subheader("📊 System Telemetry & AI Observability")
    
    col1, col2 = st.columns(2)
    col1.metric("Host CPU Overhead", f"{psutil.cpu_percent()}%")
    col2.metric("System RAM Utilization", f"{psutil.virtual_memory().percent}%")
    
    st.divider()
    
    st.markdown("### 🔍 Live Agent Operational Graph (Arize Phoenix)")
    st.caption("Inspect live node execution chains, prompt layers, and retrieval latency metrics below:")
    
    st.components.v1.iframe(
        src="http://localhost:6006", 
        height=750, 
        scrolling=True
    )

# Style metric cards
style_metric_cards(
    background_color="#1E293B",
    border_left_color="#3B82F6",
    border_color="#334155"
)
