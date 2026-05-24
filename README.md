# Intelligent Recruiter Agent 6️⃣7️⃣😭

## Overview
An AI-powered recruitment agent that matches 
candidates to job descriptions using RAG, 
hybrid retrieval and LangGraph orchestration.

## Team
- Member 1 — Data Pipeline
- Member 2 — Retrieval + Reranking
- Member 3 — LangGraph Agent
- Member 4 — UI + Observability

## Tech Stack
- Python 3.11
- LangChain + LangGraph
- ChromaDB
- Sentence Transformers
- Streamlit
- Chutes AI
- Hermes CLI

## Architecture


## Setup Instructions

### 1. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### 2. Install Libraries
pip install -r requirements.txt

### 3. Configure Environment
cp .env.example .env
# Add your API keys inside .env

### 4. Run Application
streamlit run ui/app.py

## How It Works
1. Upload resumes (PDF/DOCX)
2. Paste job description
3. Agent retrieves + ranks candidates
4. View match scores + reasoning
5. Every claim cited back to resume

## Project Structure
intelligent-recruiter/
├── ingest/
├── retrieval/
├── agent/
├── ui/
├── cache/
├── observability/
├── evals/
├── data/
├── .env
├── requirements.txt
└── README.md

## References
- LangChain Documentation
- ChromaDB Documentation
- Direct Corpus Interaction Paper (DCI)
- Chutes AI Documentation
- Hermes CLI Documentation

