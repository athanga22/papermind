import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'PaperMind — Agentic Research Synthesis',
  description: 'Ask complex questions across your research papers. Get synthesized, cited answers powered by agentic RAG.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#09090B] text-zinc-50 antialiased">
        {children}
      </body>
    </html>
  )
}
