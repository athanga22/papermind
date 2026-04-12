"""
Generates docs/review_round1.md — a human-readable review document.

For each question shows:
  - The question
  - The expected answer (from silver set)
  - The TARGET chunk text (what should have been retrieved)
  - The TOP 3 actually retrieved chunks (what the system found)
  - Hit/miss status

This is the only way to verify:
  1. Is the question answerable from the target chunk? (validates the question)
  2. Is the expected answer actually in the chunk? (validates the answer)
  3. Did the retriever find the right thing? (validates retrieval)
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

EVAL_PATH   = Path("docs/eval_round1.json")
SILVER_PATH = Path("docs/golden_set_silver.json")
OUT_PATH    = Path("docs/review_round1.md")


def get_chunk_text(qdrant: QdrantClient, chunk_id: str) -> dict:
    results, _ = qdrant.scroll(
        collection_name="papers",
        scroll_filter={"must": [{"key": "chunk_id", "match": {"value": chunk_id}}]},
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        p = results[0].payload
        return {
            "text":    p["text"],
            "section": p["section"],
            "paper":   p["paper_title"],
        }
    return {"text": "[chunk not found]", "section": "?", "paper": "?"}


def main():
    qdrant = QdrantClient(host="localhost", port=6333)

    with open(EVAL_PATH)   as f: eval_results = json.load(f)
    with open(SILVER_PATH) as f: silver = json.load(f)

    silver_by_id = {i+1: qa for i, qa in enumerate(silver)}

    lines = []
    lines.append("# Retrieval Review — Round 1\n")
    lines.append("**How to review:**")
    lines.append("- Read the question → read the TARGET CHUNK → does the chunk contain the answer?")
    lines.append("- If yes: question is valid. Check if TOP RETRIEVED also got it right.")
    lines.append("- If no: the question is bad (answer isn't in that chunk). Mark for deletion.")
    lines.append("- For cross-paper (✦): just check if the top retrieved chunks are from the right papers.\n")
    lines.append("---\n")

    hit5_pass = 0
    hit5_total = 0

    for r in eval_results:
        qa     = silver_by_id.get(r["id"], {})
        is_cross = r["source"] == "cross_paper"
        target_id = r["chunk_id"]

        tri_hit5 = r["modes"]["trimodal"].get("hits", {}).get("hit@5", None)
        if tri_hit5 is not None:
            hit5_total += 1
            if tri_hit5:
                hit5_pass += 1

        # Header
        status = ""
        if is_cross:
            status = "✦ CROSS-PAPER"
        elif tri_hit5 is True:
            status = "✅ HIT@5"
        elif tri_hit5 is False:
            status = "❌ MISS"

        lines.append(f"## [{r['id']:02d}] {r['type'].upper()} — {status}")
        lines.append(f"**Paper:** {r['paper_title'][:80]}\n")
        lines.append(f"**Q:** {r['question']}\n")
        lines.append(f"**Expected answer:**")
        lines.append(f"> {r['answer']}\n")

        # Target chunk (single-paper only)
        if target_id and not is_cross:
            chunk = get_chunk_text(qdrant, target_id)
            lines.append(f"**TARGET CHUNK** `{target_id}`")
            lines.append(f"*Paper: {chunk['paper'][:60]} | Section: {chunk['section']}*")
            lines.append(f"```")
            lines.append(chunk["text"])
            lines.append(f"```\n")

        # Top 3 retrieved chunks
        top3 = r["modes"]["trimodal"]["top3"]
        lines.append(f"**TOP 3 RETRIEVED (trimodal):**\n")
        for i, h in enumerate(top3, 1):
            marker = "🎯" if h["chunk_id"] == target_id else "  "
            lines.append(f"{marker} **#{i}** `{h['chunk_id']}` score={h['score']} [{', '.join(h['sources'])}]")
            lines.append(f"*{h['paper'][:60]} | {h['section']}*")
            lines.append(f"```")
            lines.append(h["text"])
            lines.append(f"```\n")

        # Review action
        lines.append("**Your verdict:** [ ] Keep as-is  [ ] Fix answer  [ ] Delete  [ ] Fix question\n")
        lines.append("---\n")

    # Summary at top
    summary = [
        f"**Hit@5 (trimodal): {hit5_pass}/{hit5_total} = {hit5_pass/hit5_total:.0%}**  \n",
        f"Total questions: {len(eval_results)} ({hit5_total} single-paper, {len(eval_results)-hit5_total} cross-paper)\n\n",
    ]
    lines = lines[:5] + summary + lines[5:]

    OUT_PATH.write_text("\n".join(lines))
    print(f"Saved → {OUT_PATH}")
    print(f"Hit@5: {hit5_pass}/{hit5_total} = {hit5_pass/hit5_total:.0%}")
    print(f"\nOpen docs/review_round1.md and go through each question.")
    print("The TARGET CHUNK shows exactly what text the answer should come from.")
    print("The TOP 3 shows what the system actually found.")


if __name__ == "__main__":
    main()
