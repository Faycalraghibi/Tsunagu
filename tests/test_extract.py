from __future__ import annotations

import json
from typing import Any

import pytest

from tsunagu.chunking import Chunk
from tsunagu.extract import JsonExtractor, filter_strict

CHUNK = Chunk(
    "p::000",
    "p",
    ["Persistence"],
    "InMemorySaver is a checkpointer.",
    8,
    ["InMemorySaver"],
    [],
)


def test_filter_strict_normalises_and_drops() -> None:
    items = [
        {
            "subject": "InMemorySaver",
            "subject_type": "CLASS",
            "relation": "part of",
            "object": "checkpointer",
            "object_type": "concept",
        },
        {
            "subject": "A",
            "subject_type": "Widget",
            "relation": "USES",
            "object": "B",
            "object_type": "Concept",
        },
        {
            "subject": "A",
            "subject_type": "Concept",
            "relation": "LIKES",
            "object": "B",
            "object_type": "Concept",
        },
        {
            "subject": "Same",
            "subject_type": "Concept",
            "relation": "USES",
            "object": "same",
            "object_type": "Concept",
        },
        {
            "subject": "",
            "subject_type": "Concept",
            "relation": "USES",
            "object": "B",
            "object_type": "Concept",
        },
    ]
    [t] = filter_strict(items, "p::000")
    assert (t.subject, t.subject_type, t.relation, t.object, t.object_type) == (
        "InMemorySaver",
        "Class",
        "PART_OF",
        "checkpointer",
        "Concept",
    )
    assert t.chunk_id == "p::000"


def test_json_extractor(make_llm: Any) -> None:
    reply = json.dumps(
        {
            "triplets": [
                {
                    "subject": "InMemorySaver",
                    "subject_type": "Class",
                    "relation": "PART_OF",
                    "object": "checkpointer",
                    "object_type": "Concept",
                }
            ]
        }
    )
    triplets = JsonExtractor(make_llm(lambda m: reply)).extract(CHUNK)
    assert len(triplets) == 1 and triplets[0].relation == "PART_OF"


def test_json_extractor_bad_reply_gives_nothing(make_llm: Any) -> None:
    assert JsonExtractor(make_llm(lambda m: "sorry")).extract(CHUNK) == []


def test_llamaindex_extractor_with_fake_llm(make_llm: Any) -> None:
    pytest.importorskip("llama_index.core")
    from tsunagu.extract import LlamaIndexExtractor

    reply = json.dumps(
        {
            "triplets": [
                {
                    "subject": {"type": "CLASS", "name": "InMemorySaver"},
                    "relation": {"type": "PART_OF"},
                    "object": {"type": "CONCEPT", "name": "checkpointer"},
                }
            ]
        }
    )
    triplets = LlamaIndexExtractor(make_llm(lambda m: reply)).extract(CHUNK)
    assert [(t.subject, t.subject_type, t.relation, t.object, t.object_type) for t in triplets] == [
        ("InMemorySaver", "Class", "PART_OF", "checkpointer", "Concept")
    ]
