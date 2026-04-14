'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import {
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  AlertCircle,
  FileText,
  ChevronDown,
  RotateCcw,
  Layers,
  Clock,
  Zap,
} from 'lucide-react'
import { MOCK_CONVERSATIONS, MOCK_PAPERS, type RetrievedChunk } from '@/lib/mock-data'
import { formatMs } from '@/lib/utils'

// ── Parse inline citations from answer text ──────────────────────────────────

function AnswerWithCitations({ content }: { content: string }) {
  const parts = content.split(/(\[[^\]]+,\s*§[^\]]+\])/g)
  return (
    <>
      {parts.map((part, i) => {
        const match = part.match(/^\[(.+)\]$/)
        if (match) {
          return (
            <span
              key={i}
              data-testid="detail-citation-pill"
              className="citation-pill"
            >
              {match[1]}
            </span>
          )
        }
        const boldParts = part.split(/(\*\*[^*]+\*\*)/g)
        return (
          <span key={i}>
            {boldParts.map((bp, j) => {
              if (bp.startsWith('**') && bp.endsWith('**')) {
                return <strong key={j} className="text-zinc-100 font-medium">{bp.slice(2, -2)}</strong>
              }
              const italicParts = bp.split(/(\*[^*]+\*)/g)
              return (
                <span key={j}>
                  {italicParts.map((ip, k) => {
                    if (ip.startsWith('*') && ip.endsWith('*') && ip.length > 2) {
                      return <em key={k} className="not-italic text-zinc-300">{ip.slice(1, -1)}</em>
                    }
                    return <span key={k}>{ip}</span>
                  })}
                </span>
              )
            })}
          </span>
        )
      })}
    </>
  )
}

// ── Chunk card ───────────────────────────────────────────────────────────────

function ChunkCard({ chunk }: { chunk: RetrievedChunk }) {
  const paper = MOCK_PAPERS.find(p => p.id === chunk.paperId)

  return (
    <div
      data-testid={`chunk-card-${chunk.chunkId}`}
      className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden"
    >
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800 bg-zinc-900">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" />
          <span className="font-mono text-xs text-zinc-300 truncate">{chunk.shortTitle}</span>
          <span className="text-zinc-700">·</span>
          <span className="font-mono text-xs text-zinc-500 truncate">§{chunk.section}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          <span className="font-mono text-xs text-zinc-600">score</span>
          <span
            className={`font-mono text-xs font-medium ${
              chunk.score >= 0.9 ? 'text-green-400' : chunk.score >= 0.8 ? 'text-amber-400' : 'text-zinc-400'
            }`}
          >
            {chunk.score.toFixed(3)}
          </span>
        </div>
      </div>
      <div className="px-4 py-3">
        <p className="font-serif text-sm text-zinc-400 leading-relaxed">{chunk.text}</p>
        {chunk.subQuery && (
          <p className="mt-2 font-mono text-xs text-zinc-700">
            sub-query: "{chunk.subQuery}"
          </p>
        )}
      </div>
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function AnswerDetailPage({ params }: { params: { id: string } }) {
  const searchParams = useSearchParams()
  const msgId = searchParams.get('msg')

  const conversation = MOCK_CONVERSATIONS.find(c => c.id === params.id)
    ?? MOCK_CONVERSATIONS[0]

  // Find specific message or last assistant message
  const targetMsg = msgId
    ? conversation.messages.find(m => m.id === msgId)
    : conversation.messages.filter(m => m.role === 'assistant').at(-1)

  const targetIndex = targetMsg ? conversation.messages.indexOf(targetMsg) : -1
  const userMsg = targetMsg
    ? conversation.messages
        .filter((m, i) => m.role === 'user' && i < targetIndex)
        .at(-1) ?? null
    : null

  const [showAllChunks, setShowAllChunks] = useState(false)
  const [activeTab, setActiveTab] = useState<'answer' | 'chunks' | 'trace'>('answer')

  if (!targetMsg || targetMsg.role !== 'assistant') {
    return (
      <div className="min-h-screen bg-[#09090B] flex items-center justify-center">
        <div className="text-center text-zinc-600">
          <BrainCircuit className="w-8 h-8 mx-auto mb-3 opacity-40" />
          <p className="text-sm">Answer not found.</p>
          <Link href="/chat" className="text-xs text-zinc-500 hover:text-zinc-300 mt-2 block">
            ← Back to chat
          </Link>
        </div>
      </div>
    )
  }

  const chunks = targetMsg.retrievedChunks ?? []
  const displayedChunks = showAllChunks ? chunks : chunks.slice(0, 3)

  const confidenceConfig = {
    high: { icon: CheckCircle2, color: 'text-green-400', bg: 'bg-green-950/20 border-green-900/40', label: 'High confidence' },
    medium: { icon: AlertCircle, color: 'text-amber-400', bg: 'bg-amber-950/20 border-amber-900/40', label: 'Medium confidence' },
    low: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-950/20 border-red-900/40', label: 'Low confidence' },
  }
  const conf = targetMsg.confidence ? confidenceConfig[targetMsg.confidence] : null

  return (
    <div className="min-h-screen bg-[#09090B]">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-[#09090B]/90 backdrop-blur-md">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/chat"
              data-testid="detail-back-to-chat"
              className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-300 transition-colors text-xs"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Chat
            </Link>
            <span className="text-zinc-800">/</span>
            <span className="text-sm text-zinc-300 font-medium line-clamp-1 max-w-xs">
              {conversation.title}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {conf && (
              <span
                data-testid="detail-confidence"
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs font-mono ${conf.bg} ${conf.color}`}
              >
                <conf.icon className="w-3 h-3" />
                {conf.label}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10">
        {/* Question */}
        {userMsg && (
          <div className="mb-8" data-testid="detail-question">
            <p className="font-mono text-xs text-zinc-600 mb-2">question</p>
            <h1 className="font-heading text-xl font-medium text-zinc-100 leading-snug">
              {userMsg.content}
            </h1>
          </div>
        )}

        {/* Stats bar */}
        <div className="flex flex-wrap items-center gap-4 mb-8 font-mono text-xs text-zinc-600">
          {targetMsg.processingMs && (
            <span className="flex items-center gap-1.5">
              <Clock className="w-3 h-3" />
              {formatMs(targetMsg.processingMs)}
            </span>
          )}
          {targetMsg.subQueries && (
            <span className="flex items-center gap-1.5">
              <Zap className="w-3 h-3" />
              {targetMsg.subQueries.length} sub-queries
            </span>
          )}
          {targetMsg.replanCount !== undefined && targetMsg.replanCount > 0 && (
            <span className="flex items-center gap-1.5 text-amber-700">
              <RotateCcw className="w-3 h-3" />
              {targetMsg.replanCount} replan{targetMsg.replanCount !== 1 ? 's' : ''}
            </span>
          )}
          {chunks.length > 0 && (
            <span className="flex items-center gap-1.5">
              <Layers className="w-3 h-3" />
              {chunks.length} retrieved chunk{chunks.length !== 1 ? 's' : ''}
            </span>
          )}
          {targetMsg.citations && (
            <span className="flex items-center gap-1.5">
              <FileText className="w-3 h-3" />
              {targetMsg.citations.length} citation{targetMsg.citations.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Tab navigation */}
        <div className="flex border-b border-zinc-800 mb-6" data-testid="detail-tabs">
          {[
            { key: 'answer', label: 'Answer' },
            { key: 'chunks', label: `Retrieved Chunks (${chunks.length})` },
            { key: 'trace', label: 'Agent Trace' },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key as typeof activeTab)}
              data-testid={`tab-${key}`}
              className={`px-4 py-2.5 text-xs font-mono transition-colors border-b-2 -mb-px ${
                activeTab === key
                  ? 'border-zinc-400 text-zinc-200'
                  : 'border-transparent text-zinc-600 hover:text-zinc-400'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Tab: Answer */}
        {activeTab === 'answer' && (
          <div data-testid="tab-content-answer">
            <div className="max-w-3xl">
              <div className="answer-text leading-loose space-y-4">
                {targetMsg.content.split('\n\n').map((para, i) => {
                  if (para.startsWith('**') && para.endsWith('**') && !para.includes('\n')) {
                    return (
                      <h3 key={i} className="font-heading text-base font-medium text-zinc-200 mt-5 mb-2">
                        {para.slice(2, -2)}
                      </h3>
                    )
                  }
                  return (
                    <p key={i} className="text-[1.05rem] text-zinc-300 leading-[1.85]">
                      <AnswerWithCitations content={para} />
                    </p>
                  )
                })}
              </div>

              {/* Citation index */}
              {targetMsg.citations && targetMsg.citations.length > 0 && (
                <div className="mt-10 pt-6 border-t border-zinc-800" data-testid="citation-index">
                  <p className="font-mono text-xs text-zinc-600 mb-4">Citations</p>
                  <div className="space-y-2">
                    {targetMsg.citations.map((cit, i) => (
                      <div
                        key={cit.id}
                        data-testid={`citation-item-${i}`}
                        className="flex items-center gap-3 text-xs font-mono text-zinc-500"
                      >
                        <span className="text-zinc-700 w-5 text-right">[{i + 1}]</span>
                        <span className="text-zinc-400">{cit.shortTitle}</span>
                        <span className="text-zinc-700">§{cit.section}</span>
                        <span className="text-zinc-800 font-mono text-[10px]">{cit.chunkId}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab: Chunks */}
        {activeTab === 'chunks' && (
          <div data-testid="tab-content-chunks">
            {chunks.length === 0 ? (
              <div className="text-center py-16 text-zinc-700">
                <Layers className="w-7 h-7 mx-auto mb-3 opacity-40" />
                <p className="text-sm">No retrieved chunks for this mock answer.</p>
                <p className="text-xs mt-1">Live backend will populate this with actual retrieved chunks.</p>
              </div>
            ) : (
              <div className="space-y-3" data-testid="chunks-list">
                {displayedChunks.map((chunk) => (
                  <ChunkCard key={chunk.chunkId} chunk={chunk} />
                ))}
                {chunks.length > 3 && (
                  <button
                    onClick={() => setShowAllChunks(!showAllChunks)}
                    data-testid="toggle-all-chunks"
                    className="w-full flex items-center justify-center gap-2 py-3 text-xs font-mono text-zinc-600 hover:text-zinc-400 border border-zinc-800 rounded-lg transition-colors"
                  >
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showAllChunks ? 'rotate-180' : ''}`} />
                    {showAllChunks ? 'Show less' : `Show ${chunks.length - 3} more chunks`}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab: Agent Trace */}
        {activeTab === 'trace' && (
          <div data-testid="tab-content-trace" className="max-w-3xl space-y-6">
            {/* Sub-queries */}
            {targetMsg.subQueries && (
              <div>
                <p className="font-mono text-xs text-zinc-600 mb-3">Sub-query decomposition</p>
                <div className="space-y-2">
                  {targetMsg.subQueries.map((sq, i) => (
                    <div
                      key={i}
                      data-testid={`trace-sub-query-${i}`}
                      className="flex items-start gap-3 bg-zinc-900 border border-zinc-800 rounded-md px-4 py-3"
                    >
                      <span className="font-mono text-xs text-zinc-700 flex-shrink-0 mt-0.5 w-5">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span className="text-sm text-zinc-400">{sq}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pipeline stages */}
            <div>
              <p className="font-mono text-xs text-zinc-600 mb-3">Pipeline execution</p>
              <div className="space-y-1">
                {[
                  { stage: 'classifier', status: 'complete', note: 'Query complexity: moderate' },
                  { stage: 'planner', status: 'complete', note: `${targetMsg.subQueries?.length ?? 3} sub-queries generated` },
                  { stage: 'retrieve_one (×' + (targetMsg.subQueries?.length ?? 3) + ')', status: 'complete', note: 'Parallel fan-out complete' },
                  { stage: 'rerank', status: 'complete', note: 'Dedup + score sort (Cohere disabled)' },
                  { stage: 'gate', status: 'complete', note: targetMsg.replanCount ? 'Insufficient → replan' : 'Context sufficient → synthesize' },
                  ...(targetMsg.replanCount
                    ? [{ stage: 'replan', status: 'complete', note: 'Fresh sub-queries generated' }]
                    : []),
                  { stage: 'synthesize', status: 'complete', note: `Claude Haiku · confidence: ${targetMsg.confidence}` },
                ].map(({ stage, status, note }, i) => (
                  <div
                    key={i}
                    data-testid={`trace-stage-${i}`}
                    className="flex items-center gap-3 px-4 py-2.5 bg-zinc-950 border border-zinc-800 rounded-md"
                  >
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                      status === 'complete' ? 'bg-green-500' : 'bg-zinc-700'
                    }`} />
                    <span className="font-mono text-xs text-zinc-400 w-48 flex-shrink-0">{stage}</span>
                    <span className="text-xs text-zinc-600">{note}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* RAGAS context */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-5">
              <p className="font-mono text-xs text-zinc-600 mb-3">RAGAS context (agent v1 eval)</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Faithfulness', value: '0.956' },
                  { label: 'Ctx Recall', value: '0.958' },
                  { label: 'Relevancy', value: '0.760' },
                  { label: 'Precision', value: '0.614' },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center">
                    <div className="font-mono text-sm text-zinc-300">{value}</div>
                    <div className="font-mono text-xs text-zinc-600 mt-0.5">{label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
