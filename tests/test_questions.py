from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tsunagu.questions import EXPECTED_COUNTS, Question, load_questions, validate_questions


def _q(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q01",
        "type": "single_fact",
        "question": "What does a reducer do?",
        "gold_answer": "It merges updates into a state key.",
        "gold_pages": ["langgraph/graph-api"],
    }
    base.update(kw)
    return base


def test_unanswerable_must_have_no_gold_pages() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_q(type="unanswerable"))
    Question.model_validate(_q(type="unanswerable", gold_pages=[]))


def test_answerable_needs_one_to_four_pages() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_q(gold_pages=[]))
    with pytest.raises(ValidationError):
        Question.model_validate(_q(gold_pages=["a", "b", "c", "d", "e"]))


def test_validate_reports_unknown_pages_duplicates_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [_q(), _q(gold_pages=["langgraph/nope"])]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    problems = validate_questions(load_questions(path), {"langgraph/graph-api"})
    assert any("duplicate id q01" in p for p in problems)
    assert any("unknown page langgraph/nope" in p for p in problems)
    assert any("multi_hop: 0 (expected 8)" in p for p in problems)
    assert sum(EXPECTED_COUNTS.values()) == 30
