from __future__ import annotations

import json
from typing import Any

from tsunagu.answer import Answer
from tsunagu.judge import citation_precision, judge_answer
from tsunagu.questions import Question
from tsunagu.retrievers import RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        "p::000", "langgraph/persistence", ["Persistence"], "Checkpointers save state.", 1.0
    )
]
Q = Question(
    id="q1",
    type="single_fact",
    question="What saves state?",
    gold_answer="Checkpointers.",
    gold_pages=["langgraph/persistence"],
)
UNANSWERABLE = Question(
    id="q9", type="unanswerable", question="Bedrock?", gold_answer="n/a", gold_pages=[]
)


def _responder(messages: list[dict[str, Any]]) -> str:
    content = messages[-1]["content"]
    if "Rubric" in content:
        return json.dumps({"score": 2, "reason": "matches"})
    return json.dumps(
        {
            "claims": [
                {
                    "claim": "Checkpointers save state",
                    "supported": True,
                    "support_labels": ["langgraph/persistence#Persistence"],
                },
                {"claim": "They use Redis", "supported": False, "support_labels": []},
            ]
        }
    )


def test_answerable_judgement(make_llm: Any) -> None:
    ans = Answer(
        "Checkpointers save state, using Redis.",
        ["langgraph/persistence#Persistence", "langgraph/other#X"],
        False,
        True,
    )
    j = judge_answer(make_llm(_responder), Q, CHUNKS, ans)
    assert j.correctness == 2 and j.correctness_reason == "matches"
    assert (j.claims_total, j.claims_supported) == (2, 1)
    assert (j.cited_pages, j.precise_citations) == (2, 1)


def test_abstentions_skip_the_judge(make_llm: Any) -> None:
    def boom(messages: list[dict[str, Any]]) -> str:
        raise AssertionError("judge must not be called")

    llm = make_llm(boom)
    abstained = Answer("INSUFFICIENT_CONTEXT", [], True, True)
    assert judge_answer(llm, UNANSWERABLE, CHUNKS, abstained).correctness == 2
    assert judge_answer(llm, Q, CHUNKS, abstained).correctness == 0
    answered = Answer("Use Bedrock.", [], False, True)
    assert judge_answer(llm, UNANSWERABLE, CHUNKS, answered).correctness == 0


def test_unparseable_judge_gives_none(make_llm: Any) -> None:
    ans = Answer("x", [], False, True)
    j = judge_answer(make_llm(lambda m: "garbage"), Q, CHUNKS, ans)
    assert j.correctness is None and j.claims_total == 0


def test_citation_precision_counts_gold_or_supporting() -> None:
    claims = [{"supported": True, "support_labels": ["[langgraph/b#H]"]}]
    assert citation_precision(
        ["langgraph/a#X", "langgraph/b#H", "langgraph/c#Y"], ["langgraph/a"], claims
    ) == (3, 2)
