# PaperMind Frontend — PRD

## Problem Statement
Build the frontend for PaperMind, an agentic RAG system for deep cross-document synthesis of academic research papers. Dark, clean, no-fuss research tool UI matching the application's vibe.

## Architecture

### Tech Stack
- **Framework**: Next.js 14 (App Router) running via `yarn dev` on port 3000
- **Styling**: Tailwind CSS + custom CSS (globals.css)
- **Fonts**: IBM Plex Sans (headings), Inter (body), JetBrains Mono (mono/citations), Newsreader (serif/answers)
- **Deps**: lucide-react (icons), react-dropzone (upload), framer-motion (installed), clsx/tailwind-merge (utils)
- **Mock Data**: `/app/frontend/src/lib/mock-data.ts` — 6 papers from DESIGN.md, 2 conversations with full mock answers

### Design System
- Background: #09090B (zinc-950), Surface: #18181B (zinc-900)
- Borders: #27272A (zinc-800) with hover: #3F3F46 (zinc-700)
- Text: #FAFAFA primary, #A1A1AA secondary, #71717A muted
- Confidence colors: green-400 (high), amber-400 (medium), red-400 (low)
- Citation pills: inline monospace badges with §Section formatting

## Pages Built

### 1. Landing Page (`/`)
- Hero with grid overlay, animated badge, H1, CTAs
- Demo answer strip showing inline citation rendering
- Features bento grid (5 cards): Agentic RAG, Citations, Cross-Paper Synthesis, Confidence Gating, Graph Retrieval
- RAGAS metrics strip (4 metrics from DESIGN.md Phase 4a)
- Footer CTA

### 2. Paper Library (`/library`)
- Drag & drop upload dropzone with simulated ingestion
- 6 mock papers from DESIGN.md routing papers
- Paper cards: title, authors, year, venue badge, chunk count, section pills
- Abstract toggle (expand/collapse)
- Remove paper (with animation)
- Slot counter (n/10 papers)
- Navigation to chat

### 3. Chat Interface (`/chat`)
- Sidebar: logo, new conversation button, conversation list, paper corpus panel
- Main area: message bubbles (user right-aligned, AI left-aligned with icon)
- AI answers: Newsreader serif font, inline citation pills `[Title, §Section]`, confidence badge, timing, replan indicator
- Agent trace accordion (expandable sub-query list)
- Empty state with 4 sample questions
- Mock AI response generator with 2s delay + typing indicator
- Input area: textarea (Enter to send, Shift+Enter for newline)

### 4. Answer Detail (`/chat/[id]`)
- Header: conversation title, confidence badge
- Stats bar: latency, sub-query count, replan count, chunk count, citation count
- 3-tab interface:
  - **Answer**: Full Newsreader-rendered answer with inline citations + citation index
  - **Retrieved Chunks**: Chunk cards (paper, section, score, text in serif, sub-query attribution)
  - **Agent Trace**: Sub-query decomposition list + pipeline execution stages + RAGAS context

## What's Been Implemented
- [2025-01-xx] All 4 pages built and tested
- [2025-01-xx] Mock data: 6 papers, 2 conversations, 9 messages, 11 retrieved chunks
- [2025-01-xx] Bug fix: Answer detail shows correct preceding question in multi-turn conversations

## Prioritized Backlog

### P0 (Wire backend)
- Connect `/api/papers` — list, upload, delete papers
- Connect `/api/query` — run agent, stream response
- Connect `/api/conversations` — persist chat history

### P1 (Nice to have)
- PDF upload progress indicator with real ingestion status
- Streaming AI response (token-by-token render)
- Session persistence (localStorage conversations)
- Paper detail page with full metadata

### P2 (Future)
- Dark mode toggle (light mode)
- Export conversation as PDF/markdown
- Share a conversation link
- RAGAS scores per question (live eval)
- Settings page (model selection, temperature)
- Multi-corpus support
