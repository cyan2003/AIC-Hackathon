import streamlit as st
import httpx
import os
import pandas as pd
import altair as alt
import psutil
import asyncio
import time
from streamlit_extras.metric_cards import style_metric_cards

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. Direct OpenTelemetry to look at your local running Phoenix instance
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# 2. Automatically intercept and instrument LangChain / LangGraph calls
# The second your teammate plugs their pipeline loops into your UI, 
# it will auto-record every single node execution step!
try:
    LangChainInstrumentor().instrument(skip_dep_check=True)
except Exception as e:
    print(f"Telemetry warning: {e}")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zero‑Hallucination Recruiter AI",
    page_icon="🤖",
    layout="wide",
)

# Global CSS for utility classes (Tailwind‑like)
st.markdown(
    """
    <style>
    .bg-card   { background-color: #f9fafb; }
    @media (prefers-color-scheme: dark) {
        .bg-card { background-color: #1e293b; }
    }
    .rounded   { border-radius: .5rem; }
    .shadow    { box-shadow: 0 1px 3px rgba(0,0,0,.1); }
    .p-4       { padding: 1rem; }
    .grid      { display: grid; gap: 1rem; }
    .grid-cols-3 { grid-template-columns: repeat(3, minmax(0,1fr)); }
    @media (max-width:600px) {
        .grid-cols-3 { grid-template-columns: 1fr; }
    }
    .card:focus-visible { outline: 2px solid #2563eb; outline-offset: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – job description and execution trigger
# ---------------------------------------------------------------------------
st.sidebar.header("Job Specification")
job_description = st.sidebar.text_area(
    "Paste Job Description Here",
    height=250,
    placeholder="Looking for a Python Developer with experience in RAG...",
)
run_btn = st.sidebar.button("Run Recruiter Agent", type="primary")

# ---------------------------------------------------------------------------
# Tab navigation
# ---------------------------------------------------------------------------
tab_match, tab_ingest, tab_telemetry = st.tabs([
    "🎯 Match Dashboard",
    "📂 Ingestion Center",
    "📊 System Telemetry",
])

# ---------------------------------------------------------------------------
# Helper: mock async streaming log generator
# ---------------------------------------------------------------------------
async def mock_stream(status_container: st.status):
    nodes = [
        "Node: rewrite_query",
        "Node: hybrid_retrieve (Qdrant + BM25)",
        "Node: rerank_cross_encoder",
        "Node: hallucination_guardrail",
    ]
    for node in nodes:
        status_container.write(f"🔄 {node} …")
        await asyncio.sleep(0.8)  # simulate work
    status_container.update(label="Analysis Complete!", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# Tab 1 – Match Dashboard
# ---------------------------------------------------------------------------
with tab_match:
    st.subheader("Match Dashboard")
    if run_btn:
        # Real‑time streaming block
        with st.status("Agent analyzing profiles...", expanded=True) as status:
            # Run mock stream in the event loop
            asyncio.run(mock_stream(status))
        st.success("Analysis ready! Review the matched candidates below.")

        # Placeholder results – replace with real API call when backend is ready
        candidates = pd.DataFrame({
            "candidate_id": ["C001", "C002", "C003"],
            "overall_score": [0.92, 0.85, 0.78],
            "technical_fit": [0.94, 0.80, 0.75],
            "experience_match": [0.88, 0.86, 0.77],
        })

        for _, row in candidates.iterrows():
            with st.expander(f"Candidate {row['candidate_id']} – Score {row['overall_score']:.0%}"):
                # Metric cards
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall", f"{row['overall_score']:.0%}")
                col2.metric("Technical Fit", f"{row['technical_fit']:.0%}")
                col3.metric("Experience", f"{row['experience_match']:.0%}")

                # Verification expander
                with st.expander("Verify Core Claims (Zero‑Hallucination Evidence)"):
                    # Mock evidence block – in a real app this would be populated from backend
                    st.write(
                        "**Claim:** Candidate demonstrates experience with vector databases."
                    )
                    st.write("**Source Chunk:** `...vector search implementation...`")
                    st.write("**Confidence:** 0.96")
                    st.write("---")
                    st.write(
                        "**Claim:** Proven track record of building RAG pipelines."
                    )
                    st.write("**Source Chunk:** `...RAG pipeline architecture...`")
                    st.write("**Confidence:** 0.93")
    else:
        st.info("Configure a job description and click **Run Recruiter Agent** to see matches.")

# ---------------------------------------------------------------------------
# Tab 2 – Ingestion Center
# ---------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingestion Center")
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # 1. Mock data placeholders
        total_files = len(uploaded_files)
        avg_latency_ms = 12.3  
        success_rate = 0.98  

        # 2. Use Native Streamlit Columns instead of standard HTML divs
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Files Ingested", value=total_files)

        with col2:
            st.metric(label="Avg Latency", value=f"{avg_latency_ms:.1f} ms/kB")

        with col3:
            st.metric(label="Success Rate", value=f"{success_rate:.0%}")

        # 3. Apply your premium custom styling automatically to the native metric blocks
        from streamlit_extras.metric_cards import style_metric_cards
        
        # This automatically assigns clean borders, proper text contrast, and rounded corners
        style_metric_cards(
            background_color="#1E293B",  # Slate background matching your dark theme base
            border_left_color="#3B82F6",  # Clean accent indicator line
            border_color="#334155"
        )

        st.markdown("### Pipeline Progress")

        # Mock pipeline progress table
        pipeline_df = pd.DataFrame(
            {
                "File": [os.path.basename(f.name) for f in uploaded_files],
                "Cleaned": ["🟢"] * total_files,
                "Chunked": ["🟢"] * total_files,
                "Embedded": ["🟢"] * total_files,
                "Indexed": ["🟢"] * total_files,
            }
        )
        st.subheader("Pipeline Progress")
        st.dataframe(pipeline_df)
    else:
        st.info("Upload PDF resumes to see ingestion metrics.")

# ---------------------------------------------------------------------------
# Tab 3 – System Telemetry
# ---------------------------------------------------------------------------
with tab_telemetry:
    st.subheader("📊 System Telemetry & AI Observability")
    
    # Render basic machine parameters
    import psutil
    col1, col2 = st.columns(2)
    col1.metric("Host CPU Overhead", f"{psutil.cpu_percent()}%")
    col2.metric("System RAM Utilization", f"{psutil.virtual_memory().percent}%")
    
    st.divider()
    
    st.markdown("### 🔍 Live Agent Operational Graph (Arize Phoenix)")
    st.caption("Inspect live node execution chains, prompt layers, and retrieval latency metrics below:")
    
    # Natively embed the local running Phoenix service inside your tab view!
    st.components.v1.iframe(
        src="http://localhost:6006", 
        height=750, 
        scrolling=True
    )

# Apply consistent styling to metric cards
style_metric_cards(
    background_color="#1E1E1E" if st.get_option("theme.base") == "dark" else "#F0F2F6"
)
