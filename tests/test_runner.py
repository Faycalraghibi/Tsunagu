from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tsunagu.questions import Question
from tsunagu.retrievers import RetrievalResult, RetrievedChunk
from tsunagu.runner import merge_rows, run_eval, write_rows

QUESTIONS = [
    Question(
        id="q1",
        type="single_fact",
        question="What saves state?",
        gold_answer="Checkpointers.",
        gold_pages=["langgraph/persistence"],
    ),
    Question(id="q2", type="unanswerable", question="Bedrock?", gold_answer="n/a", gold_pages=[]),
]


class FixedRetriever:
    name = "fixed"

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        chunk = RetrievedChunk(
            "p::000", "langgraph/persistence", ["Persistence"], "Checkpointers save state.", 1.0
        )
        return RetrievalResult(self.name, [chunk], 0.01, {"pool_size": 5, "fallback": 0})


def _responder(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    if "Rubric" in content:
        return json.dumps({"score": 2, "reason": "ok"})
    if "atomic factual claims" in content:
        return json.dumps({"claims": [{"claim": "c", "supported": True, "support_labels": []}]})
    if "Bedrock?" in content:
        return json.dumps({"answer": "INSUFFICIENT_CONTEXT", "citations": [], "abstained": True})
    return json.dumps(
        {
            "answer": "Checkpointers.",
            "citations": ["langgraph/persistence#Persistence"],
            "abstained": False,
        }
    )


def test_run_eval_rows(make_llm: Any) -> None:
    rows = run_eval(QUESTIONS, {"fixed": FixedRetriever()}, make_llm(_responder), repeats=2, k=8)
    assert len(rows) == 4
    q1 = [r for r in rows if r["qid"] == "q1"]
    assert [r["repeat"] for r in q1] == [0, 1]
    assert q1[0]["hit"] == 1.0 and q1[0]["mrr"] == 1.0 and q1[0]["correctness"] == 2
    assert q1[0]["graph_pool_size"] == 5 and q1[0]["question"] == "What saves state?"
    q2 = [r for r in rows if r["qid"] == "q2"]
    assert q2[0]["hit"] is None and q2[0]["abstained"] and q2[0]["correctness"] == 2


def test_retrieval_only_rows_have_no_answers() -> None:
    rows = run_eval(
        QUESTIONS, {"fixed": FixedRetriever()}, None, repeats=3, k=8, retrieval_only=True
    )
    assert len(rows) == 2 and rows[0]["answer"] is None and rows[0]["repeat"] == 0


def test_merge_rows_replaces_only_rerun_systems(tmp_path: Path) -> None:
    old = [{"system": "dense", "qid": "q1"}, {"system": "graph", "qid": "q1"}]
    new = [{"system": "graph", "qid": "q1", "v": 2}]
    merged = merge_rows(old, new)
    assert merged == [{"system": "dense", "qid": "q1"}, {"system": "graph", "qid": "q1", "v": 2}]
    write_rows(merged, tmp_path)
    assert (tmp_path / "eval_rows.jsonl").exists()
    assert (tmp_path / "per_question.csv").read_text(encoding="utf-8").startswith("qid,")
