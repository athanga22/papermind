/**
 * PaperMind API client.
 *
 * All requests go to NEXT_PUBLIC_API_URL (defaults to http://localhost:8000).
 * The /query/stream endpoint uses the browser's fetch + ReadableStream so
 * Server-Sent Events arrive as tokens in real time.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Paper {
  id: string
  title: string
  authors: string[]
  year: number | null
  sections: string[]
  chunkCount: number
}

export type ConfidenceLevel = 'high' | 'medium' | 'low'

export interface SSEProgress {
  type: 'progress'
  node: string
  message: string
}
export interface SSEToken {
  type: 'token'
  content: string
}
export interface SSEDone {
  type: 'done'
  answer: string
  confidence: ConfidenceLevel
}
export interface SSEError {
  type: 'error'
  detail: string
}

export type SSEEvent = SSEProgress | SSEToken | SSEDone | SSEError

// ── Paper management ──────────────────────────────────────────────────────────

export async function listPapers(): Promise<Paper[]> {
  const res = await fetch(`${BASE}/papers`)
  if (!res.ok) throw new Error(`Failed to list papers: ${res.statusText}`)
  return res.json()
}

export async function uploadPaper(file: File): Promise<Paper> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/papers/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

export async function deletePaper(paperId: string): Promise<void> {
  const res = await fetch(`${BASE}/papers/${paperId}`, { method: 'DELETE' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
}

// ── Query streaming ───────────────────────────────────────────────────────────

/**
 * Stream a query to /query/stream.
 *
 * Calls onEvent for each SSE event (progress, token, done, error).
 * Returns a cleanup function that aborts the request.
 */
export function streamQuery(
  question: string,
  onEvent: (event: SSEEvent) => void,
): () => void {
  const controller = new AbortController()

  ;(async () => {
    let res: Response
    try {
      res = await fetch(`${BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      })
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        onEvent({ type: 'error', detail: String(err) })
      }
      return
    }

    if (!res.ok || !res.body) {
      onEvent({ type: 'error', detail: `HTTP ${res.status}: ${res.statusText}` })
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames are separated by double newlines
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        const line = frame.trim()
        if (!line.startsWith('data:')) continue
        const json = line.slice('data:'.length).trim()
        if (!json) continue
        try {
          onEvent(JSON.parse(json) as SSEEvent)
        } catch {
          // malformed frame — skip
        }
      }
    }
  })()

  return () => controller.abort()
}
