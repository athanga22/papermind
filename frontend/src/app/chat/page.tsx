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
import { listPapers, streamQuery, type Paper, type ConfidenceLevel } from '@/lib/api'
import { formatRelativeTime, formatMs } from '@/lib/utils'

// ── Local types ───────────────────────────────────────────────────────────────

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  confidence?: ConfidenceLevel
  subQueries?: string[]
  replanCount?: number
  processingMs?: number
  cacheHit?: boolean
  isError?: boolean
}

interface Conversation {
  id: string
  title: string
  createdAt: string
  messages: Message[]
}

// ── Confidence badge ─────────────────────────────────────────────────────────

function ConfidenceBadge({ level }: { level: ConfidenceLevel }) {
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

  if (message.isError) {
    return (
      <div className="flex gap-3" data-testid="message-error">
        <div className="w-6 h-6 rounded-md border border-red-900 bg-red-950/20 flex items-center justify-center flex-shrink-0 mt-1">
          <AlertCircle className="w-3.5 h-3.5 text-red-500" />
        </div>
        <p className="text-sm text-red-400 pt-0.5">{message.content}</p>
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
          </div>
        )}

        {/* Agent trace accordion — only after streaming, when sub-queries present */}
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

// ── Thinking indicator ────────────────────────────────────────────────────────

function ThinkingIndicator({ status }: { status: string }) {
  return (
    <div className="flex gap-3" data-testid="typing-indicator">
      <div className="w-6 h-6 rounded-md border border-zinc-800 bg-zinc-900 flex items-center justify-center flex-shrink-0 mt-1">
        <BrainCircuit className="w-3.5 h-3.5 text-zinc-500 animate-pulse" />
      </div>
      <div className="flex items-center gap-1.5 pt-1">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="font-mono text-xs text-zinc-700 ml-2">{status}</span>
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

// ── Main page ─────────────────────────────────────────────────────────────────

const INITIAL_CONV: Conversation = {
  id: `conv_${Date.now()}`,
  title: 'New conversation',
  createdAt: new Date().toISOString(),
  messages: [],
}

export default function ChatPage() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([INITIAL_CONV])
  const [activeConvId, setActiveConvId] = useState<string>(INITIAL_CONV.id)
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [thinkingStatus, setThinkingStatus] = useState('thinking…')
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  const activeConv = conversations.find((c) => c.id === activeConvId) ?? conversations[0]

  // ── Load papers for sidebar ───────────────────────────────────────────────
  useEffect(() => {
    listPapers().then(setPapers).catch(() => {/* sidebar is decorative */})
  }, [])

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeConv?.messages, isThinking, streamingMsgId])

  // ── Cleanup SSE on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => { abortRef.current?.() }
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

    const convIdSnapshot = activeConvId
    const startTime = Date.now()

    // Add user message
    const userMsgId = `msg_${Date.now()}_user`
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convIdSnapshot) return c
        return {
          ...c,
          title: c.messages.length === 0 ? query.slice(0, 50) : c.title,
          messages: [...c.messages, {
            id: userMsgId,
            role: 'user' as const,
            content: query,
            timestamp: new Date().toISOString(),
          }],
        }
      })
    )
    setInput('')
    setIsThinking(true)
    setThinkingStatus('thinking…')

    let aiMsgId: string | null = null

    abortRef.current = streamQuery(query, (event) => {
      if (event.type === 'progress') {
        setThinkingStatus(event.message)
      } else if (event.type === 'token') {
        if (!aiMsgId) {
          // First token — switch from thinking indicator to streaming message
          aiMsgId = `msg_${Date.now()}_ai`
          const capturedId = aiMsgId
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convIdSnapshot) return c
              return {
                ...c,
                messages: [...c.messages, {
                  id: capturedId,
                  role: 'assistant' as const,
                  content: '',
                  timestamp: new Date().toISOString(),
                }],
              }
            })
          )
          setIsThinking(false)
          setStreamingMsgId(capturedId)
        }
        const capturedId = aiMsgId
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convIdSnapshot) return c
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === capturedId
                  ? { ...m, content: m.content + event.content }
                  : m
              ),
            }
          })
        )
      } else if (event.type === 'done') {
        const capturedId = aiMsgId
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convIdSnapshot) return c
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === capturedId
                  ? {
                      ...m,
                      // Use done.answer as ground truth (handles edge case where
                      // token stream and assembled answer diverge)
                      content: event.answer || m.content,
                      confidence: event.confidence,
                      processingMs: Date.now() - startTime,
                      cacheHit: false,
                    }
                  : m
              ),
            }
          })
        )
        setStreamingMsgId(null)
        setIsThinking(false)
        abortRef.current = null
      } else if (event.type === 'error') {
        const errId = `msg_${Date.now()}_err`
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convIdSnapshot) return c
            return {
              ...c,
              messages: [...c.messages, {
                id: errId,
                role: 'assistant' as const,
                content: `Error: ${event.detail}`,
                timestamp: new Date().toISOString(),
                isError: true,
              }],
            }
          })
        )
        setIsThinking(false)
        setStreamingMsgId(null)
        abortRef.current = null
      }
    })
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

        {/* Bottom nav + corpus */}
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
            <p className="font-mono text-xs text-zinc-700 mb-1.5">Corpus · {papers.length} papers</p>
            <div className="space-y-1">
              {papers.slice(0, 4).map((p) => (
                <div
                  key={p.id}
                  data-testid={`sidebar-paper-${p.id}`}
                  className="flex items-center gap-2 text-xs text-zinc-600"
                >
                  <FileText className="w-3 h-3 flex-shrink-0" />
                  <span className="line-clamp-1 font-mono">{p.title.split(' ').slice(0, 2).join(' ')}</span>
                </div>
              ))}
              {papers.length > 4 && (
                <Link
                  href="/library"
                  className="text-xs text-zinc-700 hover:text-zinc-500 font-mono transition-colors"
                >
                  +{papers.length - 4} more
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
              <EmptyState paperCount={papers.length} onSelect={(q) => handleSend(q)} />
            )}

            {activeConv?.messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                conversationId={activeConvId}
                isStreaming={msg.id === streamingMsgId}
              />
            ))}

            {isThinking && <ThinkingIndicator status={thinkingStatus} />}
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

function EmptyState({ paperCount, onSelect }: { paperCount: number; onSelect: (q: string) => void }) {
  return (
    <div className="text-center py-16" data-testid="chat-empty-state">
      <BrainCircuit className="w-8 h-8 text-zinc-700 mx-auto mb-4" />
      <h2 className="font-heading text-lg font-medium text-zinc-300 mb-1">
        What do you want to know?
      </h2>
      <p className="text-sm text-zinc-600 mb-8">
        {paperCount > 0
          ? `Ask questions about your ${paperCount} indexed papers`
          : 'Upload papers in the Library to get started'}
      </p>
      {paperCount > 0 && (
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
      )}
    </div>
  )
}
