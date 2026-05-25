# Intelligent Recruiter Agent 6️⃣7️⃣😭

## Overview
An AI-powered recruitment agent that matches candidates to job descriptions using a Zero-Hallucination RAG pipeline, hybrid retrieval, and advanced data processing. The system implements a two-collection Qdrant architecture to keep resumes and job descriptions separate, combining semantic vector search with BM25 lexical search and Reciprocal Rank Fusion (RRF) to produce highly precise matching results.

## Team
- Member 1 — Data Pipeline (Ingestion, Validation, Section-Aware Chunking, Metadata Extraction)
- Member 2 — Retrieval + Reranking (Vector Search, BM25, Fusion, Cross-Encoder Reranking)
- Member 3 — API Layer & Orchestration (FastAPI Web Service & Lifespan Hooks)
- Member 4 — UI, Testing, & Observability (Streamlit Dashboard, Pytest Suite, Prometheus Metrics)

## Tech Stack
- Python 3.12
- FastAPI (REST API layer)
- Qdrant (Vector search DB, supporting `:memory:` local testing & remote server deployment)
- Hugging Face Sentence Transformers (Semantic embeddings & Cross-Encoder reranking)
- rank-bm25 (Lexical store indexing & search)
- Pytest & Pytest-asyncio (Unit, Integration, and System testing)
- Pydantic v2 (Data validation & schemas)
- Prometheus (Performance monitoring & metrics)

## Architecture
```mermaid
graph TD
    A[Raw Document: Resume/JD] --> B[DefaultDocumentValidator]
    B --> C[DefaultTextCleaner]
    C --> D[Section-Aware Chunker]
    D --> E[Metadata Extractor: Heuristic/LLM]
    E --> F[Semantic Embedder]
    F --> G[(Qdrant Vector Store: Two-Collection)]
    E --> H[(BM25 Lexical Store)]
    
    I[Query: Resume Text] --> J[Query Processor]
    J --> K[Vector Search - JD Collection]
    J --> L[Lexical Search - BM25 Store]
    K --> M[Fusion: RRF / Weighted]
    L --> M
    M --> N[Cross-Encoder Reranking]
    N --> O[Final Matched JDs + Citations]
```

The system separates files into two distinct paths:
1. **Ingestion Pipeline**: Processes, validates, cleans, chunks (using regex section-aware headers), extracts key metadata, embeds, and indexes documents into both the BM25 Lexical Store and Qdrant Vector Store.
2. **Retrieval Pipeline**: Performs hybrid search across both vector and lexical backends, merges results using Reciprocal Rank Fusion (RRF) or Weighted Score Fusion, and optionally re-ranks the candidate matches using a Cross-Encoder model.

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
```

### 2. Install Libraries
```bash
cd zero-hallucination-rag
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` inside the `zero-hallucination-rag/` directory and configure the environment variables:
```bash
cp .env.example .env
```

### 4. Run Application
Run the FastAPI web backend:
```bash
uvicorn api.main:app --reload
```
You can access the interactive API docs at `http://localhost:8000/docs`.

### 5. Run the Test Suite
To verify the system end-to-end and run the 39 tests:
```bash
python -m pytest tests/
```

## How It Works
1. **Ingest Resumes & Job Descriptions**: Resumes and JDs are uploaded via `POST /ingest/resume` and `POST /ingest/jd`. They are validated (ensuring they are PDF/DOCX), cleaned, and segmented by section headers (e.g. Work Experience, Education, Skills).
2. **Indexing**: Chunks are embedded and indexed into separate collections in Qdrant (a `resumes` collection and a `job_descriptions` collection) and indexed in BM25.
3. **Match Retrieval**: A request to `POST /match` evaluates a candidate's resume against all ingested JDs by querying the JD database collection using hybrid vector and keyword search.
4. **Fusion & Reranking**: Results are fused using reciprocal rank fusion (RRF) and re-ranked using a Cross-Encoder to return the top matching JDs.
5. **Citations & Verification**: Citations are generated back to the source document chunks, ensuring zero hallucination.

## Project Structure
```text
zero-hallucination-rag/
├── api/                       # FastAPI router, models, and dependencies
│   ├── models/                # Pydantic schemas for request/response validation
│   └── routers/               # Endpoints (/ingest, /match, /health)
├── core/                      # Configuration, constants, and custom exceptions
├── ingestion/                 # Document ingestion engine
│   ├── embedders/             # Embedding generation wrappers (HuggingFace)
│   ├── processors/            # Validator, cleaner, section-aware chunkers, extractors
│   └── storage/               # Document and metadata stores (in-memory)
├── retrieval/                 # Search, fusion, and reranking engine
│   ├── fusion/                # Reciprocal Rank Fusion & Weighted Score Fusion
│   ├── lexical_store/         # BM25 keyword search implementation
│   ├── query/                 # Query processors and transformations
│   └── vector_store/          # Qdrant client implementation (in-memory & remote)
├── tests/                     # Unit, integration, and system tests
│   ├── unit/                  # Tests for models, processors, retrieval components
│   ├── integration/           # Tests for API endpoints and pipeline coordination
│   └── system/                # End-to-end flow testing with local Qdrant memory
└── utils/                     # Logging, monitoring, and general helper scripts
```

## References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qdrant Vector DB Documentation](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Direct Corpus Interaction Paper (DCI)](https://arxiv.org/)
- [Chutes AI Documentation](https://chutes.ai/)
- [Hermes CLI Documentation](https://hermes.dev/)

