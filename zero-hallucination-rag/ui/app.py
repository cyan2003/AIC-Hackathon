import streamlit as st
import httpx
import os
from streamlit_extras.metric_cards import style_metric_cards

# 1. Page Configuration
st.set_page_config(
    page_title="Zero-Hallucination Recruiter AI",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Recruitment Agent")
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

# 3. Main Dashboard View
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Resumes Processed", value=len(uploaded_files) if uploaded_files else 0)
with col2:
    st.metric(label="Top Matches Found", value=0)
with col3:
    st.metric(label="VDB Retrieval Latency", value="0.0ms")

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
        # Your results ranking and citation visualization components go here