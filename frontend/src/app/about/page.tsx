'use client'

import Link from 'next/link'
import { ArrowLeft, BrainCircuit, ChevronRight } from 'lucide-react'

// ── Section wrapper ──────────────────────────────────────────────────────────

function Section({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <section id={id} className="py-12 border-t border-zinc-800/60 scroll-mt-20">
      {children}
    </section>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-heading text-xl font-medium text-zinc-100 mb-6">{children}</h2>
  )
}

function Prose({ children }: { children: React.ReactNode }) {
  return <div className="space-y-4 text-zinc-400 text-sm leading-relaxed">{children}</div>
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-xs text-zinc-300 bg-zinc-900 border border-zinc-800 rounded px-1.5 py-0.5">
      {children}
    </code>
  )
}

// ── Inline table ─────────────────────────────────────────────────────────────

function RagasTable() {
  const rows = [
    {
      phase: '1',
      addition: 'Naive RAG (256 tok, raw embed)',
      hit5: '—', mrr: '—', faith: '—', rel: '—', recall: '—', prec: '—',
      note: 'No eval yet',
    },
    {
      phase: '1.5a',
      addition: 'Trimodal RRF (dense+BM25+graph)',
      hit5: '83%', mrr: '0.518', faith: '—', rel: '—', recall: '—', prec: '—',
      note: 'Graph hurting precision',
    },
    {
      phase: '1.5c',
      addition: 'Dense + BM25 (graph demoted)',
      hit5: '97%', mrr: '0.737', faith: '—', rel: '—', recall: '—', prec: '—',
      note: 'Retrieval target hit',
      highlight: true,
    },
    {
      phase: '2',
      addition: 'Naive RAG baseline (Haiku, n=56)',
      hit5: '97%', mrr: '0.737', faith: '0.901', rel: '0.584', recall: '0.858', prec: '0.857',
      note: 'Low relevancy from 20 adversarial cross-paper Qs',
    },
    {
      phase: '3a',
      addition: '+ Cohere reranking',
      hit5: '97%', mrr: '0.737', faith: '0.861', rel: '0.544', recall: '0.801', prec: '0.850',
      note: 'Neutral on answerable Qs; adversarial unchanged',
    },
    {
      phase: '4a',
      addition: 'LangGraph agent v1 (Sonnet, n=56)',
      hit5: '97%', mrr: '0.737', faith: '0.956', rel: '0.760', recall: '0.958', prec: '0.614',
      note: 'Faith+Recall up; planner over-decomposes',
      highlight: true,
    },
  ]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono border-collapse" data-testid="ragas-table">
        <thead>
          <tr className="border-b border-zinc-800">
            {['Phase', 'Addition', 'Hit@5', 'MRR', 'Faithfulness', 'Relevancy', 'Recall', 'Precision', 'Notes'].map((h) => (
              <th key={h} className="text-left text-zinc-600 pb-2 pr-4 font-normal whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.phase}
              className={`border-b border-zinc-800/40 ${r.highlight ? 'text-zinc-200' : 'text-zinc-500'}`}
            >
              <td className="py-2.5 pr-4 text-zinc-600">{r.phase}</td>
              <td className="py-2.5 pr-4 whitespace-nowrap max-w-[220px] truncate">{r.addition}</td>
              <td className="py-2.5 pr-4">{r.hit5}</td>
              <td className="py-2.5 pr-4">{r.mrr}</td>
              <td className={`py-2.5 pr-4 ${r.faith && r.faith !== '—' && parseFloat(r.faith) >= 0.9 ? 'text-green-400' : ''}`}>{r.faith}</td>
              <td className={`py-2.5 pr-4 ${r.rel && r.rel !== '—' && parseFloat(r.rel) >= 0.75 ? 'text-green-400' : ''}`}>{r.rel}</td>
              <td className={`py-2.5 pr-4 ${r.recall && r.recall !== '—' && parseFloat(r.recall) >= 0.9 ? 'text-green-400' : ''}`}>{r.recall}</td>
              <td className={`py-2.5 pr-4 ${r.prec && r.prec !== '—' && parseFloat(r.prec) < 0.7 ? 'text-amber-500' : ''}`}>{r.prec}</td>
              <td className="py-2.5 text-zinc-600 max-w-[200px]">{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Pipeline diagram (text-based) ────────────────────────────────────────────

function PipelineDiagram() {
  const nodes = [
    { name: 'classifier', desc: 'Haiku classifies query complexity → sets max sub-query budget (2 / 4 / 6)' },
    { name: 'planner', desc: 'Sonnet decomposes query into N independent sub-queries, capped by classifier' },
    { name: 'retrieve_one ×N', desc: 'Parallel fan-out — each sub-query runs trimodal hybrid retrieval (dense + BM25 + RRF)' },
    { name: 'rerank', desc: 'Deduplication + score sort across all retrieved chunks, capped at 20' },
    { name: 'gate', desc: 'Haiku sufficiency judge — checks paper diversity and entity coverage' },
    { name: 'replan (optional)', desc: 'If gate fails, routes back to planner with fresh angles (max 2 replans)' },
    { name: 'synthesize', desc: 'Haiku produces cited answer using only provided chunks, outputs confidence JSON' },
  ]

  return (
    <div className="space-y-1" data-testid="pipeline-diagram">
      {nodes.map((n, i) => (
        <div key={n.name} className="flex items-start gap-4">
          <div className="flex flex-col items-center flex-shrink-0 mt-1">
            <div className="w-2 h-2 rounded-full bg-zinc-600" />
            {i < nodes.length - 1 && <div className="w-px h-full min-h-[28px] bg-zinc-800 mt-1" />}
          </div>
          <div className="pb-4">
            <span className="font-mono text-xs text-zinc-300">{n.name}</span>
            <p className="text-xs text-zinc-600 mt-0.5 leading-relaxed">{n.desc}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Decision card ─────────────────────────────────────────────────────────────

function DecisionCard({
  phase,
  title,
  children,
}: {
  phase: string
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-5" data-testid={`decision-card-${phase}`}>
      <div className="flex items-center gap-3 mb-3">
        <span className="font-mono text-xs text-zinc-600 bg-zinc-950 border border-zinc-800 rounded px-2 py-0.5">
          Phase {phase}
        </span>
        <h3 className="font-heading text-sm font-medium text-zinc-200">{title}</h3>
      </div>
      <div className="text-sm text-zinc-500 leading-relaxed space-y-2">{children}</div>
    </div>
  )
}

// ── TOC item ──────────────────────────────────────────────────────────────────

const TOC = [
  { id: 'overview', label: 'Overview' },
  { id: 'ingestion', label: 'Ingestion Pipeline' },
  { id: 'retrieval', label: 'Retrieval Strategy' },
  { id: 'agent', label: 'Agentic Pipeline' },
  { id: 'synthesis', label: 'Synthesis & Attribution' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'decisions', label: 'Design Decisions' },
  { id: 'tradeoffs', label: 'Known Tradeoffs' },
]

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-[#09090B]">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-[#09090B]/90 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              data-testid="about-nav-home"
              className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-300 transition-colors text-xs"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Home
            </Link>
            <span className="text-zinc-800">/</span>
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-zinc-500" />
              <span className="text-sm text-zinc-200 font-medium">About</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/library"
              data-testid="about-nav-library"
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Library
            </Link>
            <Link
              href="/chat"
              data-testid="about-nav-chat"
              className="flex items-center gap-1.5 bg-white text-black text-xs font-medium rounded-md px-3.5 py-1.5 hover:bg-zinc-200 transition-colors"
            >
              Open chat
              <ChevronRight className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12 flex gap-16">
        {/* Sidebar TOC */}
        <aside className="hidden lg:block w-48 flex-shrink-0">
          <nav className="sticky top-24 space-y-1" data-testid="about-toc">
            <p className="font-mono text-xs text-zinc-700 mb-3 uppercase tracking-widest">Contents</p>
            {TOC.map(({ id, label }) => (
              <a
                key={id}
                href={`#${id}`}
                className="block text-xs text-zinc-600 hover:text-zinc-300 transition-colors py-1"
              >
                {label}
              </a>
            ))}
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 max-w-3xl" data-testid="about-main">

          {/* Page title */}
          <div className="mb-2">
            <p className="font-mono text-xs text-zinc-600 mb-3">architecture & evaluation</p>
            <h1 className="font-heading text-3xl font-semibold text-zinc-50 mb-4">
              How PaperMind works
            </h1>
            <p className="text-zinc-400 text-base leading-relaxed">
              A full walkthrough of the ingestion pipeline, retrieval strategy, agentic reasoning loop,
              and the RAGAS evaluation results that guided every architectural decision.
            </p>
          </div>

          {/* Overview */}
          <Section id="overview">
            <SectionTitle>Overview</SectionTitle>
            <Prose>
              <p>
                PaperMind is an agentic RAG system built specifically for deep, cross-document synthesis of academic
                research papers. The core problem it solves is attribution loss: when researchers manually synthesize
                information from multiple PDFs, the thread from claim back to source is broken. PaperMind keeps that
                thread intact by grounding every sentence of every answer in a specific chunk from a specific paper.
              </p>
              <p>
                The system is deliberately narrow in scope. It operates on a curated corpus of up to 10 papers — not
                the open web. This constraint allows it to build deep knowledge of each paper's structure, entities,
                and relationships, rather than optimizing for broad recall across a noisy index.
              </p>
              <p>
                The pipeline is built in Python using{' '}
                <Code>LangGraph</Code>,{' '}
                <Code>Qdrant</Code> for vector storage,{' '}
                <Code>BM25</Code> for lexical retrieval,{' '}
                <Code>Neo4j</Code> for the citation and entity graph, and{' '}
                <Code>Claude Haiku</Code> for synthesis. All architectural decisions are backed by RAGAS evaluation
                results documented below.
              </p>
            </Prose>
          </Section>

          {/* Ingestion Pipeline */}
          <Section id="ingestion">
            <SectionTitle>Ingestion Pipeline</SectionTitle>
            <Prose>
              <p>
                When a paper is uploaded, it passes through a 7-step ingestion pipeline before it can be queried:
              </p>
            </Prose>

            <div className="mt-6 space-y-3">
              {[
                {
                  step: '01',
                  title: 'PDF Parsing',
                  body: 'LlamaParse converts each PDF to structured markdown, preserving section headings, tables, and math expressions. Results are cached by file hash to avoid re-parsing on ingestion reruns.',
                },
                {
                  step: '02',
                  title: 'Section-aware chunking',
                  body: 'A custom PaperChunker splits markdown at section boundaries first, then by token count (512 tokens, 64-token overlap). Chunks never cross section boundaries — section context is preserved.',
                },
                {
                  step: '03',
                  title: 'Metadata extraction',
                  body: 'Regex-based extraction pulls title, authors, year, venue, and bibliography from each paper. No LLM call required — faster and more reliable on structured academic text.',
                },
                {
                  step: '04',
                  title: 'Contextual embedding → Qdrant',
                  body: 'Each chunk is prefixed with its paper title, authors, year, and section before embedding (contextual retrieval). This makes embeddings more discriminative — "Results show 15% improvement" is meaningless; "Paper X, Results section: shows 15% improvement" is not.',
                },
                {
                  step: '05',
                  title: 'BM25 index',
                  body: 'A BM25 index is built over the same contextualized chunks. BM25 captures exact-match signals that dense embeddings miss — paper-specific terminology, numeric values, and author names.',
                },
                {
                  step: '06',
                  title: 'Entity extraction → Neo4j',
                  body: 'Haiku extracts (entity, relation, entity) triples from a sample of chunks per paper. Entity types: methods, datasets, metrics, authors. Relation types: PROPOSES, EVALUATES_ON, CITES, OUTPERFORMS. These are written to Neo4j with chunk-level provenance.',
                },
                {
                  step: '07',
                  title: 'Citation graph',
                  body: 'The bibliography of each paper is parsed and written to Neo4j as CITES edges between papers. This enables graph traversal for questions about citation relationships.',
                },
              ].map((s) => (
                <div
                  key={s.step}
                  className="flex gap-4 bg-zinc-900 border border-zinc-800 rounded-lg px-5 py-4"
                  data-testid={`ingestion-step-${s.step}`}
                >
                  <span className="font-mono text-xs text-zinc-700 flex-shrink-0 mt-0.5 w-6">{s.step}</span>
                  <div>
                    <p className="text-sm font-medium text-zinc-200 mb-1">{s.title}</p>
                    <p className="text-xs text-zinc-500 leading-relaxed">{s.body}</p>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Retrieval Strategy */}
          <Section id="retrieval">
            <SectionTitle>Retrieval Strategy</SectionTitle>
            <Prose>
              <p>
                Retrieval uses a trimodal hybrid approach: dense vector similarity, BM25 lexical matching, and
                graph traversal. Results are fused with Reciprocal Rank Fusion (RRF) before reranking.
              </p>
              <p>
                However, the weight of each modality is not fixed. After evaluating each combination, the graph
                modality was <strong className="text-zinc-300">demoted by default</strong> — enabling it only when
                the query explicitly involves entity relationships or citation chains. The dense+BM25 combination
                alone hit the retrieval target of Hit@5 = 97%, and graph inclusion was hurting precision on
                single-paper factoid questions.
              </p>
            </Prose>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                {
                  label: 'Dense',
                  sub: 'OpenAI text-embedding-3-small → Qdrant',
                  note: 'Captures semantic similarity. Strong on paraphrase and conceptual questions.',
                },
                {
                  label: 'BM25',
                  sub: 'bm25s on disk index',
                  note: 'Captures exact-match signals: numeric values, specific terms, author names. Anchors precision.',
                },
                {
                  label: 'Graph',
                  sub: 'Neo4j traversal (demoted)',
                  note: 'Enabled for multi-hop questions only. Useful for entity-relationship queries; harmful on simple factoids.',
                },
              ].map((m) => (
                <div
                  key={m.label}
                  className="bg-zinc-950 border border-zinc-800 rounded-lg p-4"
                  data-testid={`retrieval-modality-${m.label.toLowerCase()}`}
                >
                  <p className="font-mono text-xs text-zinc-300 mb-1">{m.label}</p>
                  <p className="font-mono text-xs text-zinc-600 mb-2">{m.sub}</p>
                  <p className="text-xs text-zinc-500 leading-relaxed">{m.note}</p>
                </div>
              ))}
            </div>
          </Section>

          {/* Agentic Pipeline */}
          <Section id="agent">
            <SectionTitle>Agentic Pipeline</SectionTitle>
            <Prose>
              <p>
                A standard single-pass RAG pipeline cannot answer cross-paper questions — it can only retrieve from
                whichever paper scores highest for the query. The agentic pipeline solves this by decomposing
                queries into independent sub-queries and retrieving in parallel fan-out, one sub-query per branch.
              </p>
              <p>
                The graph is implemented in{' '}
                <Code>LangGraph</Code> and follows the Adaptive-RAG pattern: a lightweight classifier node
                first estimates query complexity, setting a budget on how many sub-queries the planner is allowed
                to generate. This prevents over-decomposition on simple factoid questions.
              </p>
            </Prose>

            <div className="mt-6">
              <PipelineDiagram />
            </div>

            <Prose>
              <p className="mt-2">
                The <strong className="text-zinc-300">gate node</strong> is the quality control step. It uses an
                LLM sufficiency judge (Haiku) rather than the heuristic entity-overlap check used in v1 — which
                was systematically failing on cross-paper questions because entities from different papers only
                appeared in their respective sub-query windows and never reached the coverage threshold together.
              </p>
            </Prose>
          </Section>

          {/* Synthesis & Attribution */}
          <Section id="synthesis">
            <SectionTitle>Synthesis & Attribution</SectionTitle>
            <Prose>
              <p>
                Synthesis is performed by Claude Haiku with a strict system prompt that enforces three rules:
              </p>
            </Prose>

            <div className="mt-5 space-y-2">
              {[
                {
                  rule: '1',
                  title: 'Ground every claim',
                  body: 'Every factual statement must be immediately followed by a citation in the format [Short Title, §Section Name]. Claims without supporting chunks are not permitted.',
                },
                {
                  rule: '2',
                  title: 'Flag contradictions explicitly',
                  body: 'If chunks from different papers disagree on a point, the answer must flag this: "Note: [Paper A] states X while [Paper B] states Y." The model is not permitted to silently resolve contradictions.',
                },
                {
                  rule: '3',
                  title: '"I don\'t know" over hallucination',
                  body: 'If the retrieved context is insufficient, the model must say so rather than extrapolate. The confidence JSON at the end of every response — {"confidence": "high"|"medium"|"low"} — is parsed to gate low-confidence responses in the UI.',
                },
              ].map((r) => (
                <div
                  key={r.rule}
                  className="flex gap-4 border-l-2 border-zinc-800 pl-4 py-1"
                  data-testid={`synthesis-rule-${r.rule}`}
                >
                  <div>
                    <p className="text-sm font-medium text-zinc-200 mb-1">{r.title}</p>
                    <p className="text-xs text-zinc-500 leading-relaxed">{r.body}</p>
                  </div>
                </div>
              ))}
            </div>

            <Prose>
              <p className="mt-4">
                The synthesis model was switched from Sonnet to Haiku after a 10-question comparison showed less
                than 2% quality delta across all RAGAS metrics, with Haiku being 3× cheaper and 2.3× faster per
                query. For synthesis, where the bottleneck is context quality rather than model capability, Haiku
                is the right choice at this corpus scale.
              </p>
            </Prose>
          </Section>

          {/* Evaluation */}
          <Section id="evaluation">
            <SectionTitle>Evaluation</SectionTitle>
            <Prose>
              <p>
                Every architectural decision in this system is backed by RAGAS evaluation over a fixed golden test
                set of 56 questions — 29 single-paper factoid questions and 27 cross-paper synthesis questions.
                The cross-paper questions are deliberately adversarial: taxonomy comparisons, causal-link questions,
                hypotheticals, and quantitative comparisons that require reasoning across multiple papers.
              </p>
              <p>
                Evaluation uses <Code>gpt-4o-mini</Code> as the RAGAS judge for cost efficiency, with spot-checks
                against Sonnet to validate judge calibration.
              </p>
            </Prose>

            <div className="mt-6 space-y-3">
              {[
                {
                  metric: 'Faithfulness',
                  target: '≥ 0.85',
                  desc: 'Fraction of claims in the answer that are directly supported by retrieved chunks. The primary metric — hallucination is never acceptable.',
                },
                {
                  metric: 'Answer Relevancy',
                  target: '≥ 0.80',
                  desc: 'How well the answer addresses the actual question asked. Can be low even with high faithfulness if the answer is technically accurate but off-topic.',
                },
                {
                  metric: 'Context Recall',
                  target: '≥ 0.80',
                  desc: 'Fraction of ground-truth answer claims covered by the retrieved chunks. Low recall means the retriever missed relevant evidence.',
                },
                {
                  metric: 'Context Precision',
                  target: '≥ 0.70',
                  desc: 'Fraction of retrieved chunks that are actually relevant to the answer. Low precision means the synthesizer is seeing noisy context.',
                },
                {
                  metric: 'Hit@5',
                  target: '≥ 90%',
                  desc: 'Whether the correct supporting chunk appears in the top-5 retrieved results. A retrieval-layer metric, evaluated before synthesis.',
                },
              ].map((m) => (
                <div
                  key={m.metric}
                  className="flex gap-4"
                  data-testid={`eval-metric-${m.metric.toLowerCase().replace(' ', '-')}`}
                >
                  <div className="flex-shrink-0 w-40">
                    <p className="font-mono text-xs text-zinc-300">{m.metric}</p>
                    <p className="font-mono text-xs text-zinc-600">target {m.target}</p>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">{m.desc}</p>
                </div>
              ))}
            </div>

            <div className="mt-8">
              <p className="font-mono text-xs text-zinc-600 mb-4">RAGAS progression across pipeline phases</p>
              <RagasTable />
            </div>

            <Prose>
              <p className="mt-6">
                The key finding from evaluation: the agentic pipeline (Phase 4a) dramatically improves faithfulness
                and context recall, but introduces a precision regression because the v1 planner over-decomposes
                simple questions into too many sub-queries, flooding the synthesizer with noisy context.
                Phase 4b targets a precision fix by capping sub-query count per query complexity class.
              </p>
            </Prose>
          </Section>

          {/* Design Decisions */}
          <Section id="decisions">
            <SectionTitle>Design Decisions</SectionTitle>

            <div className="space-y-4">
              <DecisionCard phase="2" title="Haiku over Sonnet for synthesis">
                <p>
                  Initial evaluation used Sonnet. After comparing on 10 questions, Haiku achieved nearly identical
                  results — less than 2% delta on faithfulness, relevancy, and recall. Haiku is 3× cheaper and
                  2.3× faster. Cost savings compound across every future evaluation iteration, which matters when
                  eval runs 56 questions per architectural change.
                </p>
              </DecisionCard>

              <DecisionCard phase="3" title="Cohere reranking disabled by default">
                <p>
                  Reranking over top-20 candidates had a neutral-to-negative effect on the current corpus.
                  The root cause: BM25 already achieves near-perfect recall on single-paper questions
                  (Hit@5 = 97%). When the initial pool is this clean, reranking has no headroom to improve —
                  and occasionally shuffles a well-placed chunk down.
                </p>
                <p>
                  Cohere reranking remains available and becomes valuable at Phase 4 where the pool size expands
                  to 40-60 candidates across multi-hop retrieval. With a larger, noisier pool, cross-encoder
                  scoring should show genuine improvement.
                </p>
              </DecisionCard>

              <DecisionCard phase="1.5" title="Graph retrieval demoted from default fusion">
                <p>
                  The initial trimodal hybrid (dense + BM25 + graph) with RRF showed lower precision than the
                  bimodal hybrid. Langfuse trace inspection revealed the cause: graph entities like
                  "evaluation metric", "generation", and "tolerance" appear across all 10 papers — Cohere's
                  semantic scorer distributed relevance too broadly.
                </p>
                <p>
                  Graph traversal is now conditionally enabled for questions that explicitly involve entity
                  relationships or citation chains. For all other queries, dense+BM25 is more precise.
                </p>
              </DecisionCard>

              <DecisionCard phase="4" title="LLM sufficiency gate over heuristic entity overlap">
                <p>
                  The v1 gate used a coverage heuristic — checking whether query entities appeared in the retrieved
                  chunk pool. This failed systematically on cross-paper questions because entities from different
                  papers only appeared in their respective sub-query windows and never reached the threshold together.
                </p>
                <p>
                  The v2 gate uses a Haiku LLM judge that evaluates sufficiency directly: "Given this question and
                  these retrieved chunks, do we have enough context to answer reliably?" This approach handles
                  cross-paper coverage correctly and reduces unnecessary replans.
                </p>
              </DecisionCard>
            </div>
          </Section>

          {/* Known Tradeoffs */}
          <Section id="tradeoffs">
            <SectionTitle>Known Tradeoffs</SectionTitle>

            <div className="space-y-4">
              {[
                {
                  name: 'Latency',
                  value: 'p50 ~23s on agent v1',
                  desc: 'The agentic pipeline is significantly slower than naive RAG (~4s p50). Each sub-query runs a full retrieval pass, and replans add another full cycle. Phase 4b targets sub-query count caps. Semantic caching (Phase 6) addresses repeat questions.',
                },
                {
                  name: 'GraphRAG ingestion cost',
                  value: '~Haiku, sampled',
                  desc: 'Entity extraction uses a sampled subset of chunks per paper to keep ingestion cost predictable. Full coverage would improve multi-hop retrieval quality but costs more per paper. This tradeoff is revisited at corpus scale.',
                },
                {
                  name: '10-paper corpus limit',
                  value: 'v1 constraint',
                  desc: 'The 10-paper limit is a deliberate v1 scope decision, not a technical ceiling. It keeps evaluation tractable and focuses the tool on curated reading sessions rather than broad literature search. Future versions can raise this with appropriate infrastructure scaling.',
                },
                {
                  name: 'Context precision regression in agent v1',
                  value: '0.857 → 0.614',
                  desc: 'The agentic pipeline hurt context precision because the v1 planner over-decomposes questions — generating 7-10 sub-queries for simple factoids. At 5 chunks per sub-query, the synthesizer receives 35-50 raw chunks, most irrelevant. The v2 fix adds a hard sub-query cap calibrated by complexity class.',
                },
              ].map((t) => (
                <div
                  key={t.name}
                  className="bg-zinc-900 border border-zinc-800 rounded-lg p-5"
                  data-testid={`tradeoff-${t.name.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <div className="flex items-center justify-between mb-2 gap-4">
                    <p className="font-heading text-sm font-medium text-zinc-200">{t.name}</p>
                    <span className="font-mono text-xs text-amber-600 flex-shrink-0">{t.value}</span>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">{t.desc}</p>
                </div>
              ))}
            </div>

            {/* Footer CTA */}
            <div className="mt-12 pt-8 border-t border-zinc-800/60 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-zinc-300">Ready to try it?</p>
                <p className="text-xs text-zinc-600 mt-0.5">Upload your papers and start asking questions.</p>
              </div>
              <Link
                href="/library"
                data-testid="about-cta-library"
                className="flex items-center gap-2 bg-white text-black text-sm font-medium rounded-md px-4 py-2 hover:bg-zinc-200 transition-colors"
              >
                Go to Library
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </Section>

        </main>
      </div>
    </div>
  )
}
