from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tsunagu.llm import DiskCache, LLMClient

Responder = Callable[[list[dict[str, Any]]], str]


@pytest.fixture
def make_llm(tmp_path: Path) -> Callable[[Responder], LLMClient]:
    """Build an LLMClient whose transport answers with `responder(messages)`."""

    def factory(responder: Responder) -> LLMClient:
        def transport(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "text": responder(payload["messages"]),
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

        return LLMClient(model="fake", transport=transport, cache=DiskCache(tmp_path / "cache"))

    return factory
