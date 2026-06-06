from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tsunagu.llm import (
    DiskCache,
    LLMClient,
    RateLimiter,
    RetryableLLMError,
    extract_json,
    strip_thinking,
    usage_by_stage,
)


def test_cache_hit_skips_transport_and_keeps_latency(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {"text": "hi", "prompt_tokens": 7, "completion_tokens": 2}

    llm = LLMClient(model="m", transport=transport, cache=DiskCache(tmp_path))
    msgs = [{"role": "user", "content": "hello"}]
    first = llm.complete(msgs, stage="answer", tag="a")
    second = llm.complete(msgs, stage="answer", tag="a")
    third = llm.complete(msgs, stage="answer", tag="b")
    assert len(calls) == 2
    assert not first.cached and second.cached and not third.cached
    assert second.text == "hi" and second.latency_s == first.latency_s
    assert "tag" not in calls[0]
    usage = usage_by_stage(tmp_path)
    assert usage["answer"]["calls"] == 2 and usage["answer"]["prompt_tokens"] == 14


def test_retry_on_retryable_error(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableLLMError("429")
        return {"text": "ok", "prompt_tokens": 1, "completion_tokens": 1}

    slept: list[float] = []
    llm = LLMClient(model="m", transport=transport, cache=DiskCache(tmp_path), sleep=slept.append)
    assert llm.complete([{"role": "user", "content": "x"}], stage="answer").text == "ok"
    assert attempts["n"] == 3 and len(slept) == 2


def test_retry_gives_up(tmp_path: Path) -> None:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise RetryableLLMError("down")

    llm = LLMClient(
        model="m",
        transport=transport,
        cache=DiskCache(tmp_path),
        max_retries=1,
        sleep=lambda s: None,
    )
    with pytest.raises(RetryableLLMError):
        llm.complete([{"role": "user", "content": "x"}], stage="answer")


def test_rate_limiter_spaces_calls() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now["t"] += s

    limiter = RateLimiter(rpm=60, clock=lambda: now["t"], sleep=sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert slept == [1.0, 1.0]


def test_extract_json_and_strip_thinking() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert strip_thinking("<think>hmm</think>\nAnswer") == "Answer"


def test_make_llm_fixture(make_llm: Any) -> None:
    llm = make_llm(lambda msgs: f"echo: {msgs[0]['content']}")
    resp = llm.complete([{"role": "user", "content": "test"}], stage="test")
    assert resp.text == "echo: test"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
