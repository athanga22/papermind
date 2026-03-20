"""Test set loader and validator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


QuestionType = Literal["factual", "cross_paper", "vague"]


@dataclass
class TestQuestion:
    id: str
    question: str
    question_type: QuestionType
    ground_truth: str
    papers: list[str] = field(default_factory=list)  # Paper titles relevant to question


@dataclass
class TestSet:
    version: str
    questions: list[TestQuestion]

    def filter_by_type(self, *types: QuestionType) -> "TestSet":
        filtered = [q for q in self.questions if q.question_type in types]
        return TestSet(version=self.version, questions=filtered)

    def __len__(self) -> int:
        return len(self.questions)


def load_test_set(path: str | Path) -> TestSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        TestQuestion(
            id=q["id"],
            question=q["question"],
            question_type=q["question_type"],
            ground_truth=q["ground_truth"],
            papers=q.get("papers", []),
        )
        for q in data["questions"]
    ]
    return TestSet(version=data.get("version", "1.0"), questions=questions)
