'use client'

import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import Link from 'next/link'
import {
  ArrowLeft,
  BrainCircuit,
  FileText,
  MessageSquare,
  Plus,
  Trash2,
  Upload,
  Users,
  Calendar,
  Layers,
  ChevronRight,
  AlertCircle,
  Loader2,
} from 'lucide-react'
import { listPapers, uploadPaper, deletePaper, type Paper } from '@/lib/api'

export default function LibraryPage() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [removingId, setRemovingId] = useState<string | null>(null)

  const MAX_PAPERS = 10

  // ── Load papers on mount ───────────────────────────────────────────────────
  useEffect(() => {
    listPapers()
      .then(setPapers)
      .catch((err) => setUploadError(`Failed to load papers: ${err.message}`))
      .finally(() => setLoading(false))
  }, [])

  // ── Upload ─────────────────────────────────────────────────────────────────
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      setUploadError(null)
      const remaining = MAX_PAPERS - papers.length
      if (remaining <= 0) {
        setUploadError(`Maximum of ${MAX_PAPERS} papers reached. Remove a paper to upload more.`)
        return
      }
      const filesToAdd = acceptedFiles.slice(0, remaining)
      setUploading(true)

      // Upload sequentially — ingestion is blocking on the backend
      ;(async () => {
        const added: Paper[] = []
        for (const f of filesToAdd) {
          try {
            const paper = await uploadPaper(f)
            added.push(paper)
            setPapers((prev) => {
              // Avoid duplicates if paper was already indexed
              const exists = prev.some((p) => p.id === paper.id)
              return exists ? prev.map((p) => (p.id === paper.id ? paper : p)) : [...prev, paper]
            })
          } catch (err: unknown) {
            setUploadError((err as Error).message ?? 'Upload failed')
          }
        }
        if (acceptedFiles.length > remaining) {
          setUploadError(
            `Only ${remaining} paper${remaining !== 1 ? 's' : ''} were added (limit: ${MAX_PAPERS}).`
          )
        }
        setUploading(false)
      })()
    },
    [papers.length]
  )

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    disabled: uploading || loading || papers.length >= MAX_PAPERS,
  })

  // ── Remove ─────────────────────────────────────────────────────────────────
  const handleRemove = async (id: string) => {
    setRemovingId(id)
    try {
      await deletePaper(id)
      // Small delay so the fade-out animation plays
      setTimeout(() => {
        setPapers((prev) => prev.filter((p) => p.id !== id))
        setRemovingId(null)
      }, 300)
    } catch (err: unknown) {
      setUploadError((err as Error).message ?? 'Remove failed')
      setRemovingId(null)
    }
  }

  return (
    <div className="min-h-screen bg-[#09090B]">
      {/* Top bar */}
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-[#09090B]/90 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              data-testid="nav-back-home"
              className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-300 transition-colors text-xs"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Home
            </Link>
            <span className="text-zinc-800">/</span>
            <div className="flex items-center gap-1.5">
              <BrainCircuit className="w-4 h-4 text-zinc-500" />
              <span className="text-sm text-zinc-200 font-medium">Paper Library</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-zinc-600">
              {papers.length}/{MAX_PAPERS} papers
            </span>
            <Link
              href="/chat"
              data-testid="nav-go-to-chat"
              className="flex items-center gap-1.5 bg-white text-black text-xs font-medium rounded-md px-3.5 py-1.5 hover:bg-zinc-200 transition-colors"
            >
              <MessageSquare className="w-3.5 h-3.5" />
              Open chat
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="font-heading text-2xl font-semibold text-zinc-50 mb-1">Paper Library</h1>
          <p className="text-sm text-zinc-500">
            Upload and manage your research corpus. Uploaded papers are parsed, chunked, and indexed automatically.
          </p>
        </div>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          data-testid="upload-dropzone"
          className={`
            border-2 border-dashed rounded-xl p-12 mb-6 flex flex-col items-center justify-center
            transition-all cursor-pointer select-none
            ${isDragActive && !isDragReject
              ? 'border-zinc-500 bg-zinc-900/60'
              : isDragReject
              ? 'border-red-800 bg-red-950/20'
              : papers.length >= MAX_PAPERS || uploading || loading
              ? 'border-zinc-800 bg-transparent opacity-50 cursor-not-allowed'
              : 'border-zinc-800 hover:border-zinc-600 hover:bg-zinc-900/40'
            }
          `}
        >
          <input {...getInputProps()} data-testid="upload-file-input" />
          {uploading ? (
            <>
              <div className="w-8 h-8 rounded-full border-2 border-zinc-600 border-t-zinc-300 animate-spin mb-4" />
              <p className="text-sm text-zinc-400">Parsing and indexing paper…</p>
              <p className="text-xs text-zinc-600 mt-1 font-mono">chunking · embedding · BM25</p>
            </>
          ) : (
            <>
              <div className="w-10 h-10 rounded-xl border border-zinc-800 bg-zinc-900 flex items-center justify-center mb-4">
                <Upload className="w-5 h-5 text-zinc-400" />
              </div>
              <p className="text-sm text-zinc-300 font-medium mb-1">
                {isDragActive ? 'Drop PDFs here' : 'Drag & drop PDFs, or click to select'}
              </p>
              <p className="text-xs text-zinc-600">
                PDF files only · max {MAX_PAPERS} papers · no file size limit
              </p>
            </>
          )}
        </div>

        {/* Upload/API error */}
        {uploadError && (
          <div
            data-testid="upload-error"
            className="flex items-center gap-2 text-xs text-amber-400 bg-amber-950/20 border border-amber-900/40 rounded-lg px-4 py-2.5 mb-6"
          >
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            {uploadError}
          </div>
        )}

        {/* Loading skeleton */}
        {loading ? (
          <div className="flex items-center justify-center py-20 text-zinc-600">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            <span className="text-sm">Loading corpus…</span>
          </div>
        ) : papers.length === 0 ? (
          <div
            data-testid="empty-library"
            className="text-center py-20 text-zinc-600"
          >
            <FileText className="w-8 h-8 mx-auto mb-3 opacity-40" />
            <p className="text-sm">No papers yet. Upload PDFs above to get started.</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-mono text-zinc-500 uppercase tracking-widest">
                Corpus · {papers.length} paper{papers.length !== 1 ? 's' : ''}
              </h2>
              <span className="text-xs text-zinc-600 font-mono">
                {papers.reduce((a, p) => a + p.chunkCount, 0)} total chunks indexed
              </span>
            </div>

            <div
              className="grid grid-cols-1 md:grid-cols-2 gap-3"
              data-testid="papers-grid"
            >
              {papers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  paper={paper}
                  isRemoving={removingId === paper.id}
                  onRemove={() => handleRemove(paper.id)}
                />
              ))}

              {/* Add more slot */}
              {papers.length < MAX_PAPERS && (
                <div
                  {...getRootProps()}
                  data-testid="add-more-slot"
                  className="border border-dashed border-zinc-800 rounded-lg p-6 flex flex-col items-center justify-center gap-2 text-zinc-600 hover:border-zinc-700 hover:text-zinc-500 transition-colors cursor-pointer"
                >
                  <input {...getInputProps()} />
                  <Plus className="w-5 h-5" />
                  <span className="text-xs">Add more papers</span>
                  <span className="font-mono text-xs">{MAX_PAPERS - papers.length} slots remaining</span>
                </div>
              )}
            </div>
          </>
        )}

        {/* Go to chat CTA */}
        {papers.length > 0 && (
          <div className="mt-10 pt-8 border-t border-zinc-800/60 flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-300 font-medium">Ready to query your corpus?</p>
              <p className="text-xs text-zinc-600 mt-0.5">
                {papers.length} paper{papers.length !== 1 ? 's' : ''} indexed and ready for questions.
              </p>
            </div>
            <Link
              href="/chat"
              data-testid="library-go-to-chat"
              className="flex items-center gap-2 bg-white text-black text-sm font-medium rounded-md px-4 py-2 hover:bg-zinc-200 transition-colors"
            >
              Start querying
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        )}
      </main>
    </div>
  )
}

function PaperCard({
  paper,
  isRemoving,
  onRemove,
}: {
  paper: Paper
  isRemoving: boolean
  onRemove: () => void
}) {
  return (
    <div
      data-testid={`paper-card-${paper.id}`}
      className={`
        bg-zinc-900 border border-zinc-800 rounded-lg p-5 transition-all
        ${isRemoving ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}
        hover:border-zinc-700
      `}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-7 h-7 rounded-md border border-zinc-800 bg-zinc-950 flex items-center justify-center flex-shrink-0 mt-0.5">
            <FileText className="w-3.5 h-3.5 text-zinc-500" />
          </div>
          <div className="min-w-0">
            <h3
              className="text-sm font-medium text-zinc-100 leading-snug line-clamp-2"
              data-testid={`paper-title-${paper.id}`}
            >
              {paper.title}
            </h3>
          </div>
        </div>
        <button
          onClick={onRemove}
          data-testid={`paper-remove-${paper.id}`}
          className="flex-shrink-0 p-1.5 rounded text-zinc-600 hover:text-red-400 hover:bg-red-950/20 transition-colors"
          title="Remove paper"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap items-center gap-3 mb-3 font-mono text-xs text-zinc-600">
        {paper.authors.length > 0 && (
          <span className="flex items-center gap-1">
            <Users className="w-3 h-3" />
            {paper.authors.slice(0, 2).join(', ')}
            {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
          </span>
        )}
        {paper.year && (
          <span className="flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {paper.year}
          </span>
        )}
        <span className="flex items-center gap-1">
          <Layers className="w-3 h-3" />
          {paper.chunkCount} chunks
        </span>
      </div>

      {/* Sections */}
      {paper.sections.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {paper.sections.slice(0, 5).map((s) => (
            <span
              key={s}
              className="px-1.5 py-0.5 text-zinc-600 text-xs font-mono bg-zinc-950 border border-zinc-800/60 rounded"
            >
              {s}
            </span>
          ))}
          {paper.sections.length > 5 && (
            <span className="px-1.5 py-0.5 text-zinc-700 text-xs font-mono">
              +{paper.sections.length - 5}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
