from __future__ import annotations

import json
from typing import Any

from tsunagu.answer import INSUFFICIENT, answer_question, build_messages, cited_pages
from tsunagu.retrievers import RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        "p1::000",
        "langgraph/persistence",
        ["Persistence", "Checkpointers"],
        "Checkpointers save state.",
        1.0,
    ),
    RetrievedChunk(
        "p2::000",
        "langgraph/interrupts",
        ["Interrupts"],
        "interrupt() pauses the graph.",
        0.9,
    ),
]


def test_prompt_contains_labels_and_rules() -> None:
    msgs = build_messages("What saves state?", CHUNKS)
    assert msgs[0]["role"] == "system" and INSUFFICIENT in msgs[0]["content"]
    assert "[langgraph/persistence#Checkpointers]" in msgs[1]["content"]
    assert "What saves state?" in msgs[1]["content"]


def test_valid_json_answer(make_llm: Any) -> None:
    reply = json.dumps(
        {
            "answer": "Checkpointers.",
            "citations": ["langgraph/persistence#Checkpointers"],
            "abstained": False,
        }
    )
    ans = answer_question(make_llm(lambda m: reply), "What saves state?", CHUNKS, tag="t")
    assert ans.parse_ok and not ans.abstained
    assert ans.citations == ["langgraph/persistence#Checkpointers"]
    assert cited_pages(ans.citations) == ["langgraph/persistence"]


def test_retry_once_on_bad_json(make_llm: Any) -> None:
    replies = iter(["not json", json.dumps({"answer": "ok", "citations": [], "abstained": False})])
    seen: list[int] = []

    def responder(messages: list[dict[str, Any]]) -> str:
        seen.append(len(messages))
        return next(replies)

    ans = answer_question(make_llm(responder), "q", CHUNKS, tag="t")
    assert ans.parse_ok and ans.answer == "ok"
    assert seen == [2, 4]


def test_two_bad_replies_give_parse_failure(make_llm: Any) -> None:
    ans = answer_question(make_llm(lambda m: "nope"), "q", CHUNKS, tag="t")
    assert not ans.parse_ok and not ans.abstained and ans.answer == "nope"


def test_insufficient_context_sets_abstained(make_llm: Any) -> None:
    reply = json.dumps({"answer": INSUFFICIENT, "citations": [], "abstained": False})
    ans = answer_question(make_llm(lambda m: reply), "q", CHUNKS, tag="t")
    assert ans.abstained
