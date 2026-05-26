import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const projectRoot = 'C:/Users/Damien/OneDrive/Desktop/AIC-Hackathon';
const base = join(projectRoot, '.understand-anything');
const scan = JSON.parse(readFileSync(join(base, 'intermediate/scan-result.json'), 'utf8'));
const imports = JSON.parse(readFileSync(join(base, 'tmp/ua-import-map-output.json'), 'utf8'));

const result = {
  name: 'zero-hallucination-rag',
  description:
    'AI-powered recruitment agent that matches candidates to job descriptions using a Zero-Hallucination RAG pipeline with hybrid vector (Qdrant) and BM25 retrieval, RRF fusion, and cross-encoder reranking.',
  languages: ['config', 'markdown', 'python', 'txt'],
  frameworks: [
    'FastAPI',
    'Pydantic',
    'Qdrant',
    'Sentence Transformers',
    'rank-bm25',
    'Pytest',
    'Prometheus',
    'Streamlit',
  ],
  files: scan.files,
  totalFiles: scan.totalFiles,
  filteredByIgnore: scan.filteredByIgnore,
  estimatedComplexity: scan.estimatedComplexity,
  importMap: imports.importMap,
};

writeFileSync(join(base, 'intermediate/scan-result.json'), JSON.stringify(result, null, 2));
