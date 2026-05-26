import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const projectRoot = 'C:/Users/Damien/OneDrive/Desktop/AIC-Hackathon';
const base = join(projectRoot, '.understand-anything');
const assembled = JSON.parse(readFileSync(join(base, 'intermediate/assembled-graph.json'), 'utf8'));
const scan = JSON.parse(readFileSync(join(base, 'intermediate/scan-result.json'), 'utf8'));
const commit = '9ba17ca099f5beff8b8e06b712c63d58001ffbcb';

const FILE_TYPES = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const fileNodes = assembled.nodes.filter((n) => FILE_TYPES.has(n.type));

function layerFor(node) {
  const p = node.filePath || '';
  if (p === 'README.md' || p === 'PLAN.md') return 'layer:documentation';
  if (p.includes('.env') || p.includes('requirements') || p.includes('pytest.ini')) return 'layer:configuration';
  if (p.includes('/api/')) return 'layer:api';
  if (p.includes('/ingestion/')) return 'layer:ingestion';
  if (p.includes('/retrieval/')) return 'layer:retrieval';
  if (p.includes('/tests/')) return 'layer:testing';
  if (p.includes('/ui/') || p.includes('monitoring')) return 'layer:ui-observability';
  if (p.includes('/core/') || p.includes('/utils/')) return 'layer:core';
  return 'layer:misc';
}

const layerDefs = {
  'layer:api': {
    id: 'layer:api',
    name: 'API Layer',
    description: 'FastAPI routes, Pydantic models, and HTTP orchestration for ingest and match endpoints.',
  },
  'layer:ingestion': {
    id: 'layer:ingestion',
    name: 'Ingestion Pipeline',
    description: 'Document validation, cleaning, section-aware chunking, metadata extraction, embedding, and storage.',
  },
  'layer:retrieval': {
    id: 'layer:retrieval',
    name: 'Retrieval Pipeline',
    description: 'Hybrid vector + BM25 search, fusion (RRF/weighted), reranking, and query processing.',
  },
  'layer:core': {
    id: 'layer:core',
    name: 'Core & Utilities',
    description: 'Shared configuration, exceptions, helpers, and logging used across pipelines.',
  },
  'layer:testing': {
    id: 'layer:testing',
    name: 'Tests',
    description: 'Unit, integration, and system tests for models, processors, retrieval, and API.',
  },
  'layer:ui-observability': {
    id: 'layer:ui-observability',
    name: 'UI & Observability',
    description: 'Streamlit dashboard and Prometheus metrics instrumentation.',
  },
  'layer:documentation': {
    id: 'layer:documentation',
    name: 'Documentation',
    description: 'README and project planning documents.',
  },
  'layer:configuration': {
    id: 'layer:configuration',
    name: 'Configuration',
    description: 'Environment variables, dependencies, and test runner settings.',
  },
  'layer:misc': {
    id: 'layer:misc',
    name: 'Miscellaneous',
    description: 'Root-level and ancillary project files.',
  },
};

const layersMap = {};
for (const n of fileNodes) {
  const lid = layerFor(n);
  if (!layersMap[lid]) layersMap[lid] = { ...layerDefs[lid], nodeIds: [] };
  layersMap[lid].nodeIds.push(n.id);
}

const layers = Object.values(layersMap).filter((l) => l.nodeIds.length);

const tour = [
  {
    order: 1,
    title: 'Project Overview',
    description:
      'Start with the README to understand the Intelligent Recruiter Agent, team roles, and hybrid RAG architecture.',
    nodeIds: ['document:README.md'],
  },
  {
    order: 2,
    title: 'Configuration & Settings',
    description:
      'Review centralized Pydantic settings that drive embedding models, Qdrant collections, chunking, and fusion strategies.',
    nodeIds: ['file:zero-hallucination-rag/core/config.py', 'config:zero-hallucination-rag/.env'],
  },
  {
    order: 3,
    title: 'API Entry Point',
    description:
      'Explore the FastAPI app lifespan, routers for health, ingestion, and resume-to-JD matching.',
    nodeIds: [
      'file:zero-hallucination-rag/api/main.py',
      'file:zero-hallucination-rag/api/routers/ingestion.py',
      'file:zero-hallucination-rag/api/routers/retrieval.py',
    ],
  },
  {
    order: 4,
    title: 'Ingestion Pipeline',
    description:
      'Follow how resumes and JDs are validated, section-chunked, embedded, and indexed into vector and lexical stores.',
    nodeIds: [
      'file:zero-hallucination-rag/ingestion/pipeline.py',
      'file:zero-hallucination-rag/ingestion/interfaces.py',
      'file:zero-hallucination-rag/ingestion/processors/section_chunker.py',
    ],
  },
  {
    order: 5,
    title: 'Retrieval & Fusion',
    description:
      'See hybrid search across Qdrant and BM25, merged via RRF or weighted fusion with optional cross-encoder reranking.',
    nodeIds: [
      'file:zero-hallucination-rag/retrieval/pipeline.py',
      'file:zero-hallucination-rag/retrieval/vector_store/qdrant.py',
      'file:zero-hallucination-rag/retrieval/lexical_store/bm25.py',
      'file:zero-hallucination-rag/retrieval/fusion/reciprocal_rank.py',
    ],
  },
  {
    order: 6,
    title: 'End-to-End Verification',
    description:
      'Run through system and API integration tests that exercise the full ingest → index → match flow.',
    nodeIds: [
      'file:zero-hallucination-rag/tests/system/test_system_flow.py',
      'file:zero-hallucination-rag/tests/integration/test_api.py',
    ],
  },
];

const nodeIds = new Set(assembled.nodes.map((n) => n.id));
for (const layer of layers) {
  layer.nodeIds = layer.nodeIds.filter((id) => nodeIds.has(id));
}
for (const step of tour) {
  step.nodeIds = step.nodeIds.filter((id) => nodeIds.has(id));
}

const graph = {
  version: '1.0.0',
  project: {
    name: scan.name,
    languages: scan.languages,
    frameworks: scan.frameworks,
    description: scan.description,
    analyzedAt: new Date().toISOString(),
    gitCommitHash: commit,
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers,
  tour,
};

writeFileSync(join(base, 'intermediate/assembled-graph.json'), JSON.stringify(graph, null, 2));
writeFileSync(join(base, 'knowledge-graph.json'), JSON.stringify(graph, null, 2));

const stats = {
  nodes: graph.nodes.length,
  edges: graph.edges.length,
  layers: layers.length,
  tourSteps: tour.length,
};
writeFileSync(join(base, 'intermediate/review.json'), JSON.stringify({ issues: [], warnings: [], stats }, null, 2));
writeFileSync(
  join(base, 'meta.json'),
  JSON.stringify({
    lastAnalyzedAt: new Date().toISOString(),
    gitCommitHash: commit,
    version: '1.0.0',
    analyzedFiles: scan.totalFiles,
  }, null, 2)
);

console.log(JSON.stringify(stats, null, 2));
