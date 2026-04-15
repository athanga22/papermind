'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  ArrowLeft,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  FileText,
  Plus,
  Send,
  Sparkles,
  Clock,
  CheckCircle2,
  AlertCircle,
  Info,
} from 'lucide-react'
import { MOCK_PAPERS, MOCK_CONVERSATIONS, type Message, type Conversation } from '@/lib/mock-data'
import { formatRelativeTime, formatMs } from '@/lib/utils'

// ── Confidence badge ─────────────────────────────────────────────────────────

function ConfidenceBadge({ level }: { level: 'high' | 'medium' | 'low' }) {
  const styles = {
    high: 'text-green-400 border-green-900/60 bg-green-950/20',
    medium: 'text-amber-400 border-amber-900/60 bg-amber-950/20',
    low: 'text-red-400 border-red-900/60 bg-red-950/20',
  }
  const icons = {
    high: <CheckCircle2 className="w-3 h-3" />,
    medium: <AlertCircle className="w-3 h-3" />,
    low: <AlertCircle className="w-3 h-3" />,
  }
  return (
    <span
      data-testid={`confidence-badge-${level}`}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono border ${styles[level]}`}
    >
      {icons[level]}
      {level}
    </span>
  )
}

// ── Render answer with inline citations + streaming cursor ───────────────────

function AnswerContent({ content, isStreaming }: { content: string; isStreaming?: boolean }) {
  const parts = content.split(/(\[[^\]]+,\s*§[^\]]+\])/g)
  return (
    <span>
      {parts.map((part, i) => {
        const isLast = i === parts.length - 1
        const match = part.match(/^\[(.+)\]$/)
        if (match) {
          return (
            <span key={i} data-testid="citation-pill" className="citation-pill">
              {match[1]}
            </span>
          )
        }
        const boldParts = part.split(/(\*\*[^*]+\*\*)/g)
        return (
          <span key={i}>
            {boldParts.map((bp, j) => {
              const isBoldLast = j === boldParts.length - 1
              if (bp.startsWith('**') && bp.endsWith('**')) {
                return <strong key={j} className="text-zinc-100 font-medium">{bp.slice(2, -2)}</strong>
              }
              const italicParts = bp.split(/(\*[^*]+\*)/g)
              return (
                <span key={j}>
                  {italicParts.map((ip, k) => {
                    const isItalicLast = k === italicParts.length - 1
                    if (ip.startsWith('*') && ip.endsWith('*') && ip.length > 2) {
                      return <em key={k} className="text-zinc-300 not-italic">{ip.slice(1, -1)}</em>
                    }
                    return (
                      <span key={k}>
                        {ip}
                        {isStreaming && isLast && isBoldLast && isItalicLast && (
                          <span className="inline-block w-0.5 h-[1em] bg-zinc-400 ml-0.5 align-middle animate-pulse" />
                        )}
                      </span>
                    )
                  })}
                </span>
              )
            })}
          </span>
        )
      })}
    </span>
  )
}

// ── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({
  message,
  conversationId,
  isStreaming,
}: {
  message: Message
  conversationId: string
  isStreaming: boolean
}) {
  const [showTrace, setShowTrace] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end" data-testid="message-user">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 max-w-[75%]">
          <p className="text-sm text-zinc-100">{message.content}</p>
        </div>
      </div>
    )
  }

  const paragraphs = message.content.split('\n\n').filter(Boolean)

  return (
    <div className="flex gap-3" data-testid="message-assistant">
      <div className="w-6 h-6 rounded-md border border-zinc-800 bg-zinc-900 flex items-center justify-center flex-shrink-0 mt-1">
        <BrainCircuit className={`w-3.5 h-3.5 ${isStreaming ? 'text-zinc-300 animate-pulse' : 'text-zinc-500'}`} />
      </div>

      <div className="flex-1 min-w-0">
        {/* Answer body */}
        <div className="answer-text space-y-3 mb-3" data-testid="answer-content">
          {paragraphs.map((para, i) => {
            const isLastPara = i === paragraphs.length - 1
            if (para.startsWith('**') && para.endsWith('**') && !para.includes('\n')) {
              return (
                <h4 key={i} className="font-heading text-sm font-medium text-zinc-200 mt-4 mb-1">
                  {para.slice(2, -2)}
                </h4>
              )
            }
            return (
              <p key={i} className="text-sm leading-relaxed text-zinc-300">
                <AnswerContent content={para} isStreaming={isStreaming && isLastPara} />
              </p>
            )
          })}
        </div>

        {/* Meta row — only show when not streaming */}
        {!isStreaming && (
          <div className="flex flex-wrap items-center gap-3 mb-2">
            {message.confidence && <ConfidenceBadge level={message.confidence} />}
            {message.processingMs && (
              <span className="font-mono text-xs text-zinc-600 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatMs(message.processingMs)}
              </span>
            )}
            {message.cacheHit && (
              <span className="font-mono text-xs text-zinc-600">cache hit</span>
            )}
            {message.replanCount !== undefined && message.replanCount > 0 && (
              <span className="font-mono text-xs text-amber-700">
                {message.replanCount} replan{message.replanCount !== 1 ? 's' : ''}
              </span>
            )}
            {message.citations && message.citations.length > 0 && (
              <Link
                href={`/chat/${conversationId}?msg=${message.id}`}
                data-testid="view-detail-link"
                className="font-mono text-xs text-zinc-600 hover:text-zinc-400 flex items-center gap-1 transition-colors"
              >
                {message.citations.length} citations
                <ChevronRight className="w-3 h-3" />
              </Link>
            )}
          </div>
        )}

        {/* Agent trace accordion — only after streaming */}
        {!isStreaming && message.subQueries && message.subQueries.length > 0 && (
          <div className="border border-zinc-800 rounded-md overflow-hidden">
            <button
              onClick={() => setShowTrace(!showTrace)}
              data-testid="toggle-agent-trace"
              className="w-full flex items-center justify-between px-3 py-2 text-xs text-zinc-500 hover:text-zinc-400 hover:bg-zinc-900/50 transition-colors bg-zinc-950"
            >
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-3 h-3" />
                <span className="font-mono">Agent trace · {message.subQueries.length} sub-queries</span>
              </span>
              <ChevronDown
                className={`w-3.5 h-3.5 transition-transform ${showTrace ? 'rotate-180' : ''}`}
              />
            </button>

            {showTrace && (
              <div className="px-3 py-2 bg-zinc-950 border-t border-zinc-800 space-y-1.5" data-testid="agent-trace-panel">
                {message.subQueries.map((sq, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="font-mono text-xs text-zinc-700 flex-shrink-0 mt-0.5">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <span className="text-xs text-zinc-500">{sq}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Thinking indicator (before streaming starts) ─────────────────────────────

function ThinkingIndicator() {
  return (
    <div className="flex gap-3" data-testid="typing-indicator">
      <div className="w-6 h-6 rounded-md border border-zinc-800 bg-zinc-900 flex items-center justify-center flex-shrink-0 mt-1">
        <BrainCircuit className="w-3.5 h-3.5 text-zinc-500 animate-pulse" />
      </div>
      <div className="flex items-center gap-1.5 pt-1">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="font-mono text-xs text-zinc-700 ml-2">thinking…</span>
      </div>
    </div>
  )
}

// ── Sample questions ─────────────────────────────────────────────────────────

const SAMPLE_QUESTIONS = [
  'How do BEST-Route and UniRoute differ in their routing metric design?',
  'What failure modes does naive RAG have on cross-paper questions?',
  'Does GraphRAG improve multi-hop recall over BM25 hybrid retrieval?',
  'How does the conversational memory paper achieve 4× context compression?',
]

// ── Main page ────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>(MOCK_CONVERSATIONS)
  const [activeConvId, setActiveConvId] = useState<string>(MOCK_CONVERSATIONS[0].id)
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const activeConv = conversations.find((c) => c.id === activeConvId) ?? conversations[0]

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeConv?.messages, isThinking, streamingMsgId])

  // Cleanup interval on unmount
  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  const handleNewConversation = () => {
    const id = `conv_${Date.now()}`
    setConversations((prev) => [{
      id,
      title: 'New conversation',
      createdAt: new Date().toISOString(),
      messages: [],
    }, ...prev])
    setActiveConvId(id)
  }

  const handleSend = useCallback((text?: string) => {
    const query = (text ?? input).trim()
    if (!query || isThinking || streamingMsgId) return

    const userMsg: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    }

    const convIdSnapshot = activeConvId

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convIdSnapshot) return c
        return {
          ...c,
          title: c.messages.length === 0 ? query.slice(0, 50) : c.title,
          messages: [...c.messages, userMsg],
        }
      })
    )
    setInput('')
    setIsThinking(true)

    const fullContent = getMockResponse(query)
    const subQueries = getMockSubQueries(query)
    const confidence: 'high' | 'medium' | 'low' = Math.random() > 0.3 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low'
    const replanCount = Math.random() > 0.7 ? 1 : 0
    const startTime = Date.now()

    // Phase 1: thinking delay (1.2–1.8s)
    const thinkDelay = 1200 + Math.random() * 600
    setTimeout(() => {
      const msgId = `msg_${Date.now()}_ai`

      // Add empty streaming message
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convIdSnapshot) return c
          return {
            ...c,
            messages: [...c.messages, {
              id: msgId,
              role: 'assistant' as const,
              content: '',
              timestamp: new Date().toISOString(),
            }],
          }
        })
      )
      setIsThinking(false)
      setStreamingMsgId(msgId)

      // Phase 2: stream characters
      let charIndex = 0
      const CHUNK = 4       // chars per tick
      const TICK = 22       // ms per tick → ~180 chars/sec

      intervalRef.current = setInterval(() => {
        charIndex = Math.min(charIndex + CHUNK, fullContent.length)
        const partial = fullContent.slice(0, charIndex)

        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convIdSnapshot) return c
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === msgId ? { ...m, content: partial } : m
              ),
            }
          })
        )

        if (charIndex >= fullContent.length) {
          if (intervalRef.current) clearInterval(intervalRef.current)
          // Finalise with metadata
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convIdSnapshot) return c
              return {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === msgId
                    ? {
                        ...m,
                        content: fullContent,
                        confidence,
                        processingMs: Date.now() - startTime,
                        cacheHit: false,
                        subQueries,
                        replanCount,
                        citations: [],
                      }
                    : m
                ),
              }
            })
          )
          setStreamingMsgId(null)
        }
      }, TICK)
    }, thinkDelay)
  }, [input, isThinking, streamingMsgId, activeConvId])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isBusy = isThinking || !!streamingMsgId

  return (
    <div className="flex h-screen bg-[#09090B] overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 border-r border-zinc-800 flex flex-col bg-[#09090B]" data-testid="chat-sidebar">
        <div className="px-4 py-4 border-b border-zinc-800">
          <Link
            href="/"
            data-testid="sidebar-logo"
            className="flex items-center gap-2 mb-4"
          >
            <BrainCircuit className="w-4 h-4 text-zinc-400" />
            <span className="font-heading text-sm font-medium text-zinc-200">PaperMind</span>
          </Link>
          <button
            onClick={handleNewConversation}
            data-testid="new-conversation-btn"
            className="w-full flex items-center gap-2 bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-zinc-300 text-xs rounded-md px-3 py-2 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            New conversation
          </button>
        </div>

        {/* Conversations */}
        <div className="flex-1 overflow-y-auto py-2" data-testid="conversations-list">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => setActiveConvId(conv.id)}
              data-testid={`conversation-item-${conv.id}`}
              className={`w-full text-left px-4 py-2.5 transition-colors ${
                conv.id === activeConvId
                  ? 'bg-zinc-900 text-zinc-200'
                  : 'text-zinc-500 hover:bg-zinc-900/50 hover:text-zinc-300'
              }`}
            >
              <p className="text-xs font-medium line-clamp-1">{conv.title}</p>
              <p className="text-xs text-zinc-700 mt-0.5 font-mono">
                {formatRelativeTime(conv.createdAt)} · {conv.messages.filter((m) => m.role === 'user').length}q
              </p>
            </button>
          ))}
        </div>

        {/* Bottom nav + papers */}
        <div className="border-t border-zinc-800">
          <div className="px-4 pt-3 pb-1 flex items-center gap-3">
            <Link
              href="/library"
              className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
              data-testid="sidebar-library-link"
            >
              <FileText className="w-3 h-3" />
              Library
            </Link>
            <span className="text-zinc-800">·</span>
            <Link
              href="/about"
              className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
              data-testid="sidebar-about-link"
            >
              <Info className="w-3 h-3" />
              About
            </Link>
          </div>

          <div className="px-4 pb-3">
            <p className="font-mono text-xs text-zinc-700 mb-1.5">Corpus · {MOCK_PAPERS.length} papers</p>
            <div className="space-y-1">
              {MOCK_PAPERS.slice(0, 4).map((p) => (
                <div
                  key={p.id}
                  data-testid={`sidebar-paper-${p.id}`}
                  className="flex items-center gap-2 text-xs text-zinc-600"
                >
                  <FileText className="w-3 h-3 flex-shrink-0" />
                  <span className="line-clamp-1 font-mono">{p.shortTitle}</span>
                </div>
              ))}
              {MOCK_PAPERS.length > 4 && (
                <Link
                  href="/library"
                  className="text-xs text-zinc-700 hover:text-zinc-500 font-mono transition-colors"
                >
                  +{MOCK_PAPERS.length - 4} more
                </Link>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-zinc-800 flex-shrink-0">
          <div>
            <p
              className="text-sm font-medium text-zinc-200 line-clamp-1"
              data-testid="active-conversation-title"
            >
              {activeConv?.title ?? 'Conversation'}
            </p>
            <p className="text-xs text-zinc-600 font-mono">
              {activeConv?.messages.filter((m) => m.role === 'user').length ?? 0} questions
            </p>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/about"
              data-testid="header-about-link"
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              About
            </Link>
            <Link
              href="/library"
              data-testid="header-library-link"
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Library
            </Link>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6" data-testid="messages-container">
          <div className="max-w-3xl mx-auto space-y-6">
            {activeConv?.messages.length === 0 && !isBusy && (
              <EmptyState onSelect={(q) => handleSend(q)} />
            )}

            {activeConv?.messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                conversationId={activeConvId}
                isStreaming={msg.id === streamingMsgId}
              />
            ))}

            {isThinking && <ThinkingIndicator />}
            <div ref={endRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="flex-shrink-0 px-6 py-4 border-t border-zinc-800 bg-[#09090B]">
          <div className="max-w-3xl mx-auto">
            <div className={`flex items-end gap-3 bg-zinc-900 border rounded-xl px-4 py-3 transition-colors ${
              isBusy ? 'border-zinc-800 opacity-60' : 'border-zinc-800 focus-within:border-zinc-700'
            }`}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isBusy}
                placeholder={isBusy ? 'Generating…' : 'Ask a question about your papers…'}
                data-testid="chat-input"
                rows={1}
                className="flex-1 bg-transparent text-sm text-zinc-100 placeholder:text-zinc-600 resize-none outline-none leading-relaxed min-h-[24px] max-h-40 disabled:cursor-not-allowed"
                style={{ overflow: 'auto' }}
              />
              <button
                onClick={() => handleSend()}
                disabled={!input.trim() || isBusy}
                data-testid="chat-send-btn"
                className="flex-shrink-0 w-8 h-8 rounded-lg bg-white text-black flex items-center justify-center hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
            <p className="text-xs text-zinc-700 mt-2 text-center font-mono">
              Enter to send · Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="text-center py-16" data-testid="chat-empty-state">
      <BrainCircuit className="w-8 h-8 text-zinc-700 mx-auto mb-4" />
      <h2 className="font-heading text-lg font-medium text-zinc-300 mb-1">
        What do you want to know?
      </h2>
      <p className="text-sm text-zinc-600 mb-8">
        Ask questions about your {MOCK_PAPERS.length} indexed papers
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto text-left">
        {SAMPLE_QUESTIONS.map((q, i) => (
          <button
            key={i}
            onClick={() => onSelect(q)}
            data-testid={`sample-question-${i}`}
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 transition-colors text-left leading-relaxed"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Mock response generator ───────────────────────────────────────────────────

function getMockResponse(query: string): string {
  const q = query.toLowerCase()
  if (q.includes('best-route') || q.includes('uniroute') || q.includes('routing')) {
    return `BEST-Route and UniRoute represent two distinct philosophies in LLM routing.

**BEST-Route** uses a training-free composite score combining task accuracy and token budget adherence [BEST-Route, §Method]. The tolerance parameter ε allows configurable trade-offs at inference time without any routing-specific fine-tuning.

**UniRoute** instead learns a unified routing objective end-to-end [UniRoute, §Unified Framework]. Weights are optimized over routing supervision data, which yields better calibration when sufficient training examples exist [UniRoute, §Results].

**Note:** [BEST-Route] achieves significantly lower latency on simple queries by design, while [UniRoute] requires a training pipeline but generalizes better to novel query types [UniRoute, §Analysis].`
  }
  if (q.includes('graph') || q.includes('multi-hop')) {
    return `GraphRAG addresses multi-hop retrieval failures that arise from BM25's lexical matching limitation [From BM25 to Corrective RAG, §BM25 Era].

The system builds a knowledge graph of entity-relation triples from all ingested papers [GraphRAG, §Graph Construction]. At retrieval time, anchor entities from initial seed retrieval are expanded up to 2 hops through the graph, surfacing related chunks from other papers [GraphRAG, §Multi-Hop Retrieval].

GraphRAG achieves significantly higher recall on 2-hop benchmarks versus dense+BM25 hybrid [GraphRAG, §Experiments]. However, gains diminish at 3 hops, consistent with noise accumulation at deeper traversal depths — a known trade-off the authors acknowledge [GraphRAG, §Ablation Study].`
  }
  if (q.includes('memory') || q.includes('compression') || q.includes('context')) {
    return `The Conversational Memory paper uses a hierarchical compression architecture with two tiers [Conversational Memory, §Memory Architecture].

**Short-term** memory retains the most recent N turns verbatim — preserving immediate coherence without any compression loss [Conversational Memory, §Compression Strategies].

**Long-term** memory compresses older turns into structured *memory blocks* — summaries of entities, references, and key claims — retrieved by semantic similarity rather than prepended wholesale [Conversational Memory, §Retrieval-Based Memory].

The key insight is that verbatim recall is rarely required — semantic content preservation suffices for most conversational queries [Conversational Memory, §Experiments]. The paper does note one important caveat: compression quality degrades for conversations with dense numerical data where exact values must be preserved.`
  }
  if (q.includes('failure') || q.includes('naive rag') || q.includes('cross-paper')) {
    return `Naive RAG has a systematic failure on cross-paper questions — questions that require synthesizing information from two or more papers in a single retrieval pass.

The core issue is that a single vector retrieval query can only surface chunks from whichever paper scores highest semantically [From BM25 to Corrective RAG, §Dense Retrieval]. When a question asks "how does A compare to B?", the retriever picks a winner and returns only its context — leaving the answer one-sided and marked low-confidence.

This is why the agentic pipeline uses **sub-query decomposition** [GraphRAG, §Multi-Hop Retrieval]: the planner splits "compare A vs B" into two independent sub-queries, retrieves from each paper in parallel fan-out, then synthesizes across both result sets. The gate node validates that both papers are represented before synthesis proceeds [BEST-Route, §Method].`
  }
  return `Based on the indexed corpus, the retrieved context shows relevant information across multiple papers.

The question touches on aspects covered in several of the indexed papers [From BM25 to Corrective RAG, §Introduction]. The core methodology involves retrieval-augmented generation with attribution-preserving synthesis [BEST-Route, §Method], evaluated against standard benchmarks [UniRoute, §Results].

Cross-paper analysis reveals that the approaches share common assumptions about retrieval quality [GraphRAG, §Background], though they differ in how they handle edge cases where context is ambiguous or sparse.

For a more specific answer, consider refining your question to focus on a particular paper or methodology.`
}

function getMockSubQueries(query: string): string[] {
  const q = query.toLowerCase()
  if (q.includes('routing') || q.includes('best-route')) {
    return [
      'What is the primary routing metric in BEST-Route?',
      'How does UniRoute define its unified routing objective?',
      'Comparison of training-free vs. learned routing approaches',
    ]
  }
  if (q.includes('graph') || q.includes('multi-hop')) {
    return [
      'How does BM25 fail on multi-hop retrieval?',
      'GraphRAG knowledge graph construction methodology',
      'Multi-hop retrieval evaluation on benchmark datasets',
      'Limitations of graph traversal at depth ≥ 3',
    ]
  }
  if (q.includes('memory') || q.includes('compression')) {
    return [
      'Short-term memory window strategy in conversational LLMs',
      'Long-term memory compression using structured summaries',
      'Quality degradation trade-offs in memory compression',
    ]
  }
  if (q.includes('failure') || q.includes('naive')) {
    return [
      'Single-pass retrieval limitations for cross-paper questions',
      'Sub-query decomposition for multi-paper synthesis',
      'Gate node validation for coverage sufficiency',
    ]
  }
  return [
    query.slice(0, 60) + (query.length > 60 ? '…' : ''),
    'Related methodology across indexed papers',
    'Comparative analysis of approaches',
  ]
}
