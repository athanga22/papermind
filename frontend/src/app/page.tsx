'use client'

import Link from 'next/link'
import { ArrowRight, BookOpen, BrainCircuit, FileText, GitBranch, Quote, Zap } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#09090B]">
      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-zinc-800/60 bg-[#09090B]/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-zinc-400" />
            <span className="font-heading text-sm font-medium text-zinc-100 tracking-wide">PaperMind</span>
          </div>
          <div className="flex items-center gap-6">
            <Link href="/library" className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
              Library
            </Link>
            <Link href="/chat" className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
              Chat
            </Link>
            <Link href="/about" className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors">
              About
            </Link>
            <Link
              href="/library"
              data-testid="nav-get-started"
              className="flex items-center gap-1.5 bg-white text-black text-xs font-medium rounded-md px-3.5 py-1.5 hover:bg-zinc-200 transition-colors"
            >
              Get started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden hero-grid noise-overlay">
        {/* Gradient blobs */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-zinc-800/30 rounded-full blur-3xl" />
          <div className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-zinc-700/20 rounded-full blur-3xl" />
        </div>

        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-zinc-800 bg-zinc-900/80 text-xs text-zinc-400 font-mono mb-8 animate-fade-in"
            data-testid="hero-badge"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Agentic RAG · Claude · Precise attribution
          </div>

          <h1
            className="font-heading text-5xl md:text-6xl lg:text-7xl font-semibold tracking-tight text-zinc-50 mb-6 animate-fade-up"
            data-testid="hero-headline"
          >
            Synthesize research,
            <br />
            <span className="text-zinc-400">not just search it.</span>
          </h1>

          <p
            className="text-zinc-400 text-lg md:text-xl leading-relaxed max-w-2xl mx-auto mb-10 animate-fade-up delay-100"
            data-testid="hero-subheadline"
          >
            Upload your papers. Ask complex questions. Get precise, attributed answers
            synthesized across your entire corpus — with citations to the exact section.
          </p>

          <div className="flex items-center justify-center gap-4 animate-fade-up delay-200">
            <Link
              href="/library"
              data-testid="hero-cta-primary"
              className="flex items-center gap-2 bg-white text-black font-medium rounded-md px-6 py-3 hover:bg-zinc-200 transition-colors text-sm"
            >
              Upload papers
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/chat"
              data-testid="hero-cta-secondary"
              className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 text-zinc-100 hover:bg-zinc-800 hover:border-zinc-700 rounded-md px-6 py-3 transition-colors text-sm"
            >
              Try a demo
            </Link>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-zinc-600 animate-fade-in delay-400">
          <span className="text-xs font-mono">scroll</span>
          <div className="w-px h-8 bg-gradient-to-b from-zinc-600 to-transparent" />
        </div>
      </section>

      {/* Demo answer strip */}
      <section className="border-y border-zinc-800/60 bg-zinc-900/30 py-8 overflow-hidden">
        <div className="max-w-6xl mx-auto px-6">
          <p className="font-mono text-xs text-zinc-600 mb-3">sample answer · confidence: high</p>
          <div className="flex items-start gap-3">
            <div className="mt-0.5 w-5 h-5 rounded border border-zinc-800 bg-zinc-900 flex items-center justify-center flex-shrink-0">
              <BrainCircuit className="w-3 h-3 text-zinc-500" />
            </div>
            <p className="font-serif text-zinc-300 text-sm leading-relaxed">
              BEST-Route and UniRoute use distinct primary generation metrics. BEST-Route employs a composite metric
              called the <em>Budget-Efficient Score with Tolerance</em>{' '}
              <span className="citation-pill">[BEST-Route, §Method]</span>, while UniRoute uses a trained{' '}
              <em>Unified Routing Score</em> <span className="citation-pill">[UniRoute, §Framework]</span>.{' '}
              <strong className="text-zinc-200">Note:</strong> BEST-Route is training-free; UniRoute requires routing
              supervision data <span className="citation-pill">[UniRoute, §Results]</span>.
            </p>
          </div>
        </div>
      </section>

      {/* Features bento */}
      <section className="py-24 px-6" data-testid="features-section">
        <div className="max-w-6xl mx-auto">
          <div className="mb-16 text-center">
            <h2 className="font-heading text-3xl md:text-4xl font-semibold text-zinc-50 mb-4">
              Built for depth, not breadth
            </h2>
            <p className="text-zinc-400 text-base max-w-xl mx-auto">
              Every component of the pipeline is optimized for precision. No hallucination, no lost attribution.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Feature 1 */}
            <div
              className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors group"
              data-testid="feature-card-rag"
            >
              <div className="w-8 h-8 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center mb-4 group-hover:border-zinc-700 transition-colors">
                <GitBranch className="w-4 h-4 text-zinc-400" />
              </div>
              <h3 className="font-heading text-base font-medium text-zinc-100 mb-2">Agentic RAG Pipeline</h3>
              <p className="text-sm text-zinc-500 leading-relaxed">
                LangGraph agent classifies query complexity, decomposes into sub-queries, retrieves in parallel,
                and re-plans when context is insufficient.
              </p>
              <div className="mt-4 pt-4 border-t border-zinc-800 font-mono text-xs text-zinc-600 flex gap-3">
                <span>classifier → planner → retrieve → synthesize</span>
              </div>
            </div>

            {/* Feature 2 */}
            <div
              className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors group"
              data-testid="feature-card-citations"
            >
              <div className="w-8 h-8 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center mb-4 group-hover:border-zinc-700 transition-colors">
                <Quote className="w-4 h-4 text-zinc-400" />
              </div>
              <h3 className="font-heading text-base font-medium text-zinc-100 mb-2">Inline Citations</h3>
              <p className="text-sm text-zinc-500 leading-relaxed">
                Every claim is attributed to its exact source: paper title and section. No claim goes unsupported.
                Contradictions across papers are flagged explicitly.
              </p>
              <div className="mt-4 pt-4 border-t border-zinc-800 font-mono text-xs text-zinc-600">
                <span>Format: [Short Title, §Section]</span>
              </div>
            </div>

            {/* Feature 3 */}
            <div
              className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors group"
              data-testid="feature-card-synthesis"
            >
              <div className="w-8 h-8 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center mb-4 group-hover:border-zinc-700 transition-colors">
                <Zap className="w-4 h-4 text-zinc-400" />
              </div>
              <h3 className="font-heading text-base font-medium text-zinc-100 mb-2">Cross-Paper Synthesis</h3>
              <p className="text-sm text-zinc-500 leading-relaxed">
                Ask questions that span multiple papers simultaneously. The agent retrieves from each paper
                independently and synthesizes a unified, comparative answer.
              </p>
              <div className="mt-4 pt-4 border-t border-zinc-800 font-mono text-xs text-zinc-600">
                <span>Up to 10 papers per corpus</span>
              </div>
            </div>
          </div>

          {/* Secondary features row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div
              className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors group"
              data-testid="feature-card-confidence"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center group-hover:border-zinc-700 transition-colors">
                  <BookOpen className="w-4 h-4 text-zinc-400" />
                </div>
                <h3 className="font-heading text-base font-medium text-zinc-100">Confidence Gating</h3>
              </div>
              <p className="text-sm text-zinc-500 leading-relaxed">
                Each answer includes a confidence level — high, medium, or low — based on how well the retrieved
                context supports the claims. Low-confidence answers are flagged, not fabricated.
              </p>
            </div>

            <div
              className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 hover:border-zinc-700 transition-colors group"
              data-testid="feature-card-graph"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center group-hover:border-zinc-700 transition-colors">
                  <FileText className="w-4 h-4 text-zinc-400" />
                </div>
                <h3 className="font-heading text-base font-medium text-zinc-100">Graph-Aware Retrieval</h3>
              </div>
              <p className="text-sm text-zinc-500 leading-relaxed">
                Entity relationships between papers are stored in Neo4j. Multi-hop questions traverse the citation
                graph to surface evidence that pure vector retrieval misses.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA footer */}
      <section className="py-24 px-6 border-t border-zinc-800/60" data-testid="cta-section">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="font-heading text-3xl font-semibold text-zinc-50 mb-4">
            Start synthesizing your corpus
          </h2>
          <p className="text-zinc-400 text-base mb-8">
            Upload up to 10 research papers and start asking questions that would take hours to answer manually.
          </p>
          <Link
            href="/library"
            data-testid="cta-section-button"
            className="inline-flex items-center gap-2 bg-white text-black font-medium rounded-md px-6 py-3 hover:bg-zinc-200 transition-colors text-sm"
          >
            Upload your papers
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-800/60 py-8 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-4 h-4 text-zinc-600" />
            <span className="font-mono text-xs text-zinc-600">PaperMind</span>
          </div>
          <p className="font-mono text-xs text-zinc-700">Agentic RAG for Research</p>
        </div>
      </footer>
    </div>
  )
}
