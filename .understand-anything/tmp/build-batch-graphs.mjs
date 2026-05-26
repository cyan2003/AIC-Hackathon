import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const projectRoot = 'C:/Users/Damien/OneDrive/Desktop/AIC-Hackathon';
const base = join(projectRoot, '.understand-anything');
const batches = JSON.parse(readFileSync(join(base, 'intermediate/batches.json'), 'utf8'));

const SUMMARIES = {
  'zero-hallucination-rag/api/main.py':
    'FastAPI application entry point wiring ingestion, matching, and health routers with lifespan hooks for Qdrant initialization.',
  'zero-hallucination-rag/ingestion/pipeline.py':
    'Orchestrates validate → clean → chunk → embed → store for resumes and job descriptions with batch support.',
  'zero-hallucination-rag/retrieval/pipeline.py':
    'Hybrid retrieval orchestrator combining vector search, BM25, fusion (RRF/weighted), and optional cross-encoder reranking.',
  'zero-hallucination-rag/retrieval/vector_store/qdrant.py':
    'Qdrant vector store with separate resumes and job_descriptions collections for semantic search.',
  'zero-hallucination-rag/retrieval/lexical_store/bm25.py':
    'BM25 lexical index for keyword search over ingested document chunks.',
  'zero-hallucination-rag/core/config.py':
    'Pydantic settings for ingestion, retrieval, embedding, chunking, fusion, and store backends.',
  'zero-hallucination-rag/ingestion/interfaces.py':
    'Core ingestion contracts: RawDocument, ProcessedChunk, Embedder, validators, chunkers, and stores.',
  'zero-hallucination-rag/retrieval/interfaces.py':
    'Retrieval contracts: VectorStore, LexicalStore, ResultFuser, Reranker, and query types.',
  'README.md':
    'Project overview, architecture diagram, setup, and team responsibilities for the Intelligent Recruiter Agent.',
  'PLAN.md': 'Development plan and milestone tracking for the hackathon RAG pipeline.',
  'zero-hallucination-rag/ui/app.py': 'Streamlit dashboard for uploading documents and viewing match results.',
  'zero-hallucination-rag/utils/monitoring.py': 'Prometheus metrics helpers for tracking pipeline duration and throughput.',
};

function nodeTypeFor(fileCategory, path) {
  if (fileCategory === 'config') return 'config';
  if (fileCategory === 'docs') return 'document';
  return 'file';
}

function nodeId(type, path, name) {
  if (type === 'file') return `file:${path}`;
  if (type === 'config') return `config:${path}`;
  if (type === 'document') return `document:${path}`;
  if (type === 'function') return `function:${path}:${name}`;
  if (type === 'class') return `class:${path}:${name}`;
  return `${type}:${path}`;
}

function complexity(nonEmpty) {
  if (nonEmpty < 50) return 'simple';
  if (nonEmpty < 200) return 'moderate';
  return 'complex';
}

function tagsFor(path, fileCategory) {
  const t = [];
  if (path.includes('/tests/') || path.includes('test_')) t.push('test');
  if (path.includes('api/')) t.push('api-handler');
  if (path.includes('pipeline')) t.push('orchestration');
  if (path.includes('interfaces')) t.push('type-definition');
  if (path.includes('config')) t.push('configuration');
  if (path.includes('fusion')) t.push('retrieval');
  if (path.includes('vector_store')) t.push('vector-store');
  if (path.includes('lexical')) t.push('lexical-search');
  if (path.includes('embedder')) t.push('embedding');
  if (path.includes('processor')) t.push('data-pipeline');
  if (path.endsWith('README.md')) t.push('documentation', 'entry-point');
  if (fileCategory === 'config') t.push('configuration');
  if (!t.length) t.push('utility');
  return [...new Set(t)].slice(0, 5);
}

function summaryFor(path, fileCategory) {
  if (SUMMARIES[path]) return SUMMARIES[path];
  const base = path.split('/').pop();
  if (fileCategory === 'docs') return `Documentation file ${base} describing project scope or plans.`;
  if (fileCategory === 'config') return `Configuration for ${base.replace(/\.[^.]+$/, '')} settings.`;
  if (path.includes('__init__.py')) return `Package barrel exporting ${path.split('/').slice(-2, -1)[0]} module symbols.`;
  if (path.includes('test')) return `Tests validating ${path.replace(/.*tests\//, '').replace(/\.py$/, '')} behavior.`;
  return `Python module ${base} supporting the zero-hallucination RAG recruitment pipeline.`;
}

for (const batch of batches.batches) {
  const idx = batch.batchIndex;
  const extract = JSON.parse(
    readFileSync(join(base, `tmp/ua-file-extract-results-${idx}.json`), 'utf8')
  );
  const nodes = [];
  const edges = [];
  const nodeSet = new Set();

  for (const r of extract.results) {
    const ftype = nodeTypeFor(r.fileCategory, r.path);
    const fid = nodeId(ftype, r.path);
    if (!nodeSet.has(fid)) {
      nodeSet.add(fid);
      nodes.push({
        id: fid,
        type: ftype,
        name: r.path.split('/').pop(),
        filePath: r.path,
        summary: summaryFor(r.path, r.fileCategory),
        tags: tagsFor(r.path, r.fileCategory),
        complexity: complexity(r.nonEmptyLines || 0),
      });
    }

    for (const cls of r.classes || []) {
      const lineSpan = (cls.endLine || 0) - (cls.startLine || 0);
      const methods = cls.methods?.length || 0;
      if (lineSpan < 20 && methods < 2) continue;
      const cid = nodeId('class', r.path, cls.name);
      if (!nodeSet.has(cid)) {
        nodeSet.add(cid);
        nodes.push({
          id: cid,
          type: 'class',
          name: cls.name,
          filePath: r.path,
          summary: `Class ${cls.name} in ${r.path.split('/').pop()} with ${methods} methods.`,
          tags: tagsFor(r.path, r.fileCategory).concat('data-model').slice(0, 5),
          complexity: lineSpan > 80 ? 'complex' : 'moderate',
        });
        edges.push({
          source: fid,
          target: cid,
          type: 'contains',
          weight: 1.0,
        });
      }
      for (const m of cls.methods || []) {
        const mid = nodeId('function', r.path, `${cls.name}.${m}`);
        if (!nodeSet.has(mid)) {
          nodeSet.add(mid);
          nodes.push({
            id: mid,
            type: 'function',
            name: m,
            filePath: r.path,
            summary: `Method ${m} on ${cls.name}.`,
            tags: ['method'],
            complexity: 'simple',
          });
          edges.push({ source: cid, target: mid, type: 'contains', weight: 1.0 });
        }
      }
    }

    for (const fn of r.functions || []) {
      const lineSpan = (fn.endLine || 0) - (fn.startLine || 0);
      if (lineSpan < 10 && !fn.name?.startsWith('test_')) continue;
      const fnid = nodeId('function', r.path, fn.name);
      if (!nodeSet.has(fnid)) {
        nodeSet.add(fnid);
        nodes.push({
          id: fnid,
          type: 'function',
          name: fn.name,
          filePath: r.path,
          summary: `Function ${fn.name} in ${r.path.split('/').pop()}.`,
          tags: tagsFor(r.path, r.fileCategory),
          complexity: lineSpan > 40 ? 'moderate' : 'simple',
        });
        edges.push({ source: fid, target: fnid, type: 'contains', weight: 1.0 });
      }
    }
  }

  const imports = batch.batchImportData || {};
  for (const [from, targets] of Object.entries(imports)) {
    const fromId = nodeId(nodeTypeFor(
      batch.files.find((f) => f.path === from)?.fileCategory || 'code',
      from
    ), from);
    for (const to of targets) {
      const toFile = batch.files.find((f) => f.path === to);
      const toType = nodeTypeFor(toFile?.fileCategory || 'code', to);
      const toId = nodeId(toType, to);
      edges.push({ source: fromId, target: toId, type: 'imports', weight: 0.7 });
    }
  }

  for (const [from, targets] of Object.entries(imports)) {
    if (!from.includes('test')) continue;
    const fromId = nodeId('file', from);
    for (const to of targets) {
      if (!to.includes('test')) {
        edges.push({
          source: from.replace(/^zero-hallucination-rag\//, 'file:zero-hallucination-rag/').startsWith('file:')
            ? fromId
            : nodeId('file', from),
          target: nodeId(
            nodeTypeFor(batch.files.find((f) => f.path === to)?.fileCategory || 'code', to),
            to
          ),
          type: 'tested_by',
          weight: 0.5,
        });
      }
    }
  }

  // Fix tested_by direction: production -> test
  const fixedEdges = edges.map((e) => {
    if (e.type !== 'tested_by') return e;
    return { ...e, source: e.target, target: e.source };
  });

  writeFileSync(
    join(base, `intermediate/batch-${idx}.json`),
    JSON.stringify({ nodes, edges: fixedEdges }, null, 2)
  );
  console.log(`Wrote batch-${idx}.json: ${nodes.length} nodes, ${fixedEdges.length} edges`);
}
