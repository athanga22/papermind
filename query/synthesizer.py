"""
Phase 2 — Answer synthesis with inline citations.

Takes a user query + a list of retrieved chunks and calls Claude Sonnet
to produce a cited, faithful answer.

Design decisions:
  - Chunks wrapped in XML tags so Claude can reference them precisely
  - Citations format: [Short Title, §Section]
  - "I don't know" fallback when retrieved context is insufficient
  - Confidence level returned alongside answer (high / medium / low)
  - System prompt enforces: attribute every claim, no extrapolation,
    flag contradictions between papers

No retrieval happens here — caller is responsible for passing chunks.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from query.retriever import RetrievedChunk

# ── Constants ─────────────────────────────────────────────────────────────────

SYNTHESIS_MODEL = "claude-sonnet-4-5"
MAX_TOKENS      = 1024

_SYSTEM_PROMPT = """\
You are PaperMind, a precise research synthesis assistant. You answer questions
about academic papers using ONLY the provided source chunks. Every factual claim
in your answer must be supported by a specific chunk.

Citation format: [Short Title, §Section Name]
  - Short Title: first 4-6 significant words of the paper title
  - §Section: exact section name from the chunk metadata

Rules:
1. Ground every claim in a chunk. If you make a claim, cite it immediately.
2. If no chunk supports a claim, do NOT make it. Say "the provided sources do
   not address this" for that point.
3. If chunks from multiple papers CONTRADICT each other on a point, flag it
   explicitly: "Note: [Paper A] states X while [Paper B] states Y."
4. If the retrieved context is clearly insufficient to answer the question,
   respond with: "I don't have enough information in the provided sources to
   answer this question reliably."
5. Be precise and academic in tone. Do not pad with generic statements.
6. At the end of your answer, output a JSON block on its own line:
   {"confidence": "high"|"medium"|"low"}
   - high: all key claims directly supported by chunks
   - medium: answer is mostly supported but some inference was needed
   - low: context is thin or question only partially answerable
"""


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    answer:     str
    confidence: str                        # "high" | "medium" | "low"
    citations:  list[str] = field(default_factory=list)   # cited chunk_ids
    raw:        str = ""                   # full model output before parsing


# ── Synthesizer ───────────────────────────────────────────────────────────────

class Synthesizer:
    """
    Wraps Claude Sonnet for grounded answer synthesis.
    Instantiate once, call synthesize() per query.
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Build context block ───────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as numbered XML blocks.
        Each block carries: chunk_id, paper title (short), section, text.
        """
        parts = []
        for i, c in enumerate(chunks, 1):
            # Short title: first 5 significant words
            short = _short_title(c.paper_title)
            parts.append(
                f'<chunk id="{i}" chunk_id="{c.chunk_id}" '
                f'paper="{short}" section="{c.section}">\n'
                f'{c.text}\n'
                f'</chunk>'
            )
        return "\n\n".join(parts)

    # ── Parse response ────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
        """Extract answer text, confidence, and cited chunk_ids from raw output."""
        # Pull confidence JSON from last line(s)
        confidence = "medium"
        conf_match = re.search(r'\{"confidence":\s*"(high|medium|low)"\}', raw)
        if conf_match:
            confidence = conf_match.group(1)

        # Answer is everything before the JSON line
        answer = re.sub(r'\s*\{"confidence":\s*"(high|medium|low)"\}\s*$', "", raw).strip()

        # Find which chunk_ids were cited (look for [id=N] patterns)
        cited_ids: list[str] = []
        cited_nums = set(re.findall(r'\[(?:chunk\s+)?(\d+)\]', raw, re.IGNORECASE))
        for num_str in cited_nums:
            idx = int(num_str) - 1
            if 0 <= idx < len(chunks):
                cited_ids.append(chunks[idx].chunk_id)

        return SynthesisResult(
            answer=answer,
            confidence=confidence,
            citations=cited_ids,
            raw=raw,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> SynthesisResult:
        """
        Generate a cited answer for the query given the retrieved chunks.

        Returns SynthesisResult with answer, confidence, and cited chunk_ids.
        If chunks is empty, returns a "no information" result immediately.
        """
        if not chunks:
            return SynthesisResult(
                answer="I don't have enough information in the provided sources to answer this question.",
                confidence="low",
            )

        context = self._build_context(chunks)

        user_msg = (
            f"<question>{query}</question>\n\n"
            f"<sources>\n{context}\n</sources>\n\n"
            "Answer the question using only the sources above. "
            "Cite each claim with [Paper Short Title, §Section]. "
            "End with the confidence JSON."
        )

        response = self._client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        return self._parse(raw, chunks)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_title(full_title: str) -> str:
    """
    Extract a short reference-friendly title.
    E.g. "BEST-Route: Adaptive LLM Routing..." → "BEST-Route"
         "Memory in the LLM Era: ..."         → "Memory in the LLM Era"
    Uses the part before the first colon if one exists, else first 5 words.
    """
    if ":" in full_title:
        return full_title.split(":")[0].strip()
    words = full_title.split()
    return " ".join(words[:5]) if len(words) > 5 else full_title
