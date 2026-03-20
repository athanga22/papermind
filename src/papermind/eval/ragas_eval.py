"""
RAGAS evaluation harness.

Runs the four core RAGAS metrics at each build stage:
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy

Usage:
    from papermind.eval.ragas_eval import run_evaluation
    results = run_evaluation(test_set, stage_name="baseline")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from ..agent.graph import run as agent_run
from .test_set import TestSet

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    stage: str
    timestamp: str
    n_questions: int
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def as_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "n_questions": self.n_questions,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
        }

    def pretty_print(self) -> None:
        print(f"\n{'='*60}")
        print(f"Stage: {self.stage}  ({self.timestamp})")
        print(f"{'='*60}")
        print(f"  Context Precision:  {self.context_precision:.4f}")
        print(f"  Context Recall:     {self.context_recall:.4f}")
        print(f"  Faithfulness:       {self.faithfulness:.4f}")
        print(f"  Answer Relevancy:   {self.answer_relevancy:.4f}")
        print(f"{'='*60}\n")


def run_evaluation(
    test_set: TestSet,
    stage_name: str,
    output_dir: str | Path = "eval_results",
) -> EvalResult:
    """
    Run RAGAS metrics over the test set.

    For each question, calls the agent and collects:
      - question
      - answer (generated)
      - contexts (retrieved chunk texts)
      - ground_truth

    Then passes the assembled dataset to RAGAS.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []  # type: ignore[type-arg]

    for i, q in enumerate(test_set.questions, start=1):
        logger.info("[%d/%d] %s", i, len(test_set), q.question)
        try:
            state = agent_run(q.question)
            contexts = [rc.chunk.text for rc in state.get("retrieved_chunks", [])]
            answer = state.get("answer", "")
        except Exception as e:
            logger.error("Agent failed on question %s: %s", q.id, e)
            contexts = []
            answer = ""

        rows.append(
            {
                "question": q.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": q.ground_truth,
            }
        )

    dataset = Dataset.from_list(rows)

    logger.info("Running RAGAS metrics for stage '%s'", stage_name)
    ragas_result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )

    scores = ragas_result.to_pandas()

    result = EvalResult(
        stage=stage_name,
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_questions=len(test_set),
        context_precision=float(scores["context_precision"].mean()),
        context_recall=float(scores["context_recall"].mean()),
        faithfulness=float(scores["faithfulness"].mean()),
        answer_relevancy=float(scores["answer_relevancy"].mean()),
    )

    # Persist result
    out_path = output_dir / f"{stage_name}_{result.timestamp.replace(':', '-')}.json"
    out_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    result.pretty_print()
    return result


def load_results(output_dir: str | Path = "eval_results") -> list[EvalResult]:
    """Load all saved eval results, sorted by timestamp."""
    results: list[EvalResult] = []
    for f in sorted(Path(output_dir).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        results.append(
            EvalResult(
                stage=data["stage"],
                timestamp=data["timestamp"],
                n_questions=data["n_questions"],
                context_precision=data["context_precision"],
                context_recall=data["context_recall"],
                faithfulness=data["faithfulness"],
                answer_relevancy=data["answer_relevancy"],
            )
        )
    return results
