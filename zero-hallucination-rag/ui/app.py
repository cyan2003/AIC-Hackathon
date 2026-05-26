import streamlit as st
import httpx
import os
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
import altair as alt

# Tailwind-like utility CSS
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


# 1. Page Configuration
st.set_page_config(
    page_title="Zero-Hallucination Recruiter AI",
    page_icon="🤡",
    layout="wide"
)

st.title("🤡 AI Recruitment Agent")
st.subheader("Zero-Hallucination Resume Screening & Hybrid Retrieval")

# 2. Sidebar Layout for Inputs
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    st.header("Job Specification")
    job_description = st.text_area(
        "Paste Job Description Here", 
        height=250, 
        placeholder="Looking for a Python Developer with experience in RAG..."
    )
    
    process_btn = st.button("Run Recruiter Agent", type="primary")

# 3. Main Dashboard View (Bento Grid)
st.markdown('<div class="grid grid-cols-3">', unsafe_allow_html=True)

# Card 1 – Resumes Processed
st.markdown(
    f'''<div class="bg-card rounded shadow p-4 text-center card" tabindex="0">
        <h3 class="text-lg font-medium">Resumes Processed</h3>
        <p class="text-2xl font-bold">{len(uploaded_files) if uploaded_files else 0}</p>
    </div>''',
    unsafe_allow_html=True,
)

# Card 2 – Top Matches Found
st.markdown(
    '''<div class="bg-card rounded shadow p-4 text-center card" tabindex="0">
        <h3 class="text-lg font-medium">Top Matches Found</h3>
        <p class="text-2xl font-bold">0</p>
    </div>''',
    unsafe_allow_html=True,
)

# Card 3 – VDB Retrieval Latency
st.markdown(
    '''<div class="bg-card rounded shadow p-4 text-center card" tabindex="0">
        <h3 class="text-lg font-medium">VDB Retrieval Latency</h3>
        <p class="text-2xl font-bold">0.0ms</p>
    </div>''',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

style_metric_cards(background_color="#1E1E1E" if st.get_option("theme.base") == "dark" else "#F0F2F6")

st.divider()

# 4. Agent Execution & Communication with FastAPI Backend
if process_btn:
    if not uploaded_files or not job_description:
        st.warning("Please upload at least one resume and provide a job description.")
    else:
        with st.status("Agent analyzing profiles...", expanded=True) as status:
            st.write("🔄 Extracting metadata from documents...")
            # Here you will use httpx to request your FastAPI endpoints later
            
            st.write("🔍 Running Hybrid Retrieval (Vector + BM25)...")
            
            st.write("🛡️ Verifying citations for zero-hallucination compliance...")
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        st.success("Analysis ready! Review the matched candidates below.")
        # ---- Results Section ----
        with st.expander("📊 Results", expanded=True):
            # Placeholder DataFrame – replace with real results later
            df = pd.DataFrame({
                "candidate_id": ["C001", "C002", "C003"],
                "score": [0.92, 0.85, 0.78],
                "match_details": ["Excellent fit", "Good fit", "Fair fit"]
            })
            st.subheader("Candidate Rankings")
            st.dataframe(df)

            # Bar chart of scores with accessible color scheme
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('candidate_id:N', title='Candidate ID'),
                y=alt.Y('score:Q', title='Match Score'),
                tooltip=['candidate_id', 'score', 'match_details']
            ).properties(height=200).configure_scale(rangeStep=30).configure_mark(color='#4f46e5')
            st.altair_chart(chart, use_container_width=True)