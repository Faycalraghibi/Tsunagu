"""The owner's question set: model, loader, validator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

QuestionType = Literal["single_fact", "multi_hop", "relational", "exact_reference", "unanswerable"]
EXPECTED_COUNTS: dict[str, int] = {
    "single_fact": 7,
    "multi_hop": 8,
    "relational": 8,
    "exact_reference": 4,
    "unanswerable": 3,
}


class Question(BaseModel):
    id: str
    type: QuestionType
    question: str
    gold_answer: str
    gold_pages: list[str]
    notes: str = ""

    @model_validator(mode="after")
    def _pages_match_type(self) -> Question:
        if self.type == "unanswerable" and self.gold_pages:
            raise ValueError("unanswerable questions must have gold_pages = []")
        if self.type != "unanswerable" and not 1 <= len(self.gold_pages) <= 4:
            raise ValueError("answerable questions need 1 to 4 gold_pages")
        return self


def load_questions(path: Path) -> list[Question]:
    """Load questions from JSONL (one object per line)."""
    with path.open(encoding="utf-8") as f:
        return [Question.model_validate(json.loads(line)) for line in f if line.strip()]


def validate_questions(questions: list[Question], known_pages: set[str]) -> list[str]:
    """Return human-readable problems; an empty list means the set is ready."""
    problems: list[str] = []
    for qid, n in Counter(q.id for q in questions).items():
        if n > 1:
            problems.append(f"duplicate id {qid}")
    for q in questions:
        problems.extend(f"{q.id}: unknown page {p}" for p in q.gold_pages if p not in known_pages)
    counts: Counter[str] = Counter(q.type for q in questions)
    for qtype, expected in EXPECTED_COUNTS.items():
        if counts.get(qtype, 0) != expected:
            problems.append(f"{qtype}: {counts.get(qtype, 0)} (expected {expected})")
    return problems
