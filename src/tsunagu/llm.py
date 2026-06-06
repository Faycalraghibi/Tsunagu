"""OpenAI-compatible LLM client with a disk cache, a rate limiter and retries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsunagu.config import Settings

Transport = Callable[[dict[str, Any]], dict[str, Any]]
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class RetryableLLMError(RuntimeError):
    """Rate limit, timeout, connection or 5xx error: worth retrying."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class DiskCache:
    """One JSON file per call under `root/<2 hex>/<sha256>.json`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return record

    def put(self, key: str, record: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


class RateLimiter:
    """Space calls at least 60/rpm seconds apart."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0

    def wait(self) -> None:
        now = self.clock()
        if now < self._next:
            self.sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


def strip_thinking(text: str) -> str:
    """Remove `<think>...</think>` blocks some reasoning models emit."""
    return THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in `text` (tolerates code fences and chatter).

    Raises:
        ValueError: if no JSON object can be parsed.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON value is not an object")
    return obj


class LLMClient:
    """The single entry point for every LLM call in the project."""

    def __init__(
        self,
        model: str,
        transport: Transport,
        cache: DiskCache,
        *,
        limiter: RateLimiter | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 5,
        extra_body: dict[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.transport = transport
        self.cache = cache
        self.limiter = limiter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.sleep = sleep

    def complete(self, messages: list[dict[str, Any]], *, stage: str, tag: str = "") -> LLMResponse:
        """Return the model reply, from cache when this exact call was made before.

        Args:
            messages: Chat messages.
            stage: Accounting bucket: "extraction", "answer" or "judge".
            tag: Distinguishes otherwise identical calls (for example repeat index).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "tag": tag,
        }
        key = DiskCache.key(payload)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(
                hit["text"], hit["prompt_tokens"], hit["completion_tokens"], hit["latency_s"], True
            )

        request = {k: v for k, v in payload.items() if k != "tag"}
        for attempt in range(self.max_retries + 1):
            if self.limiter:
                self.limiter.wait()
            started = time.perf_counter()
            try:
                out = self.transport(request)
            except RetryableLLMError:
                if attempt == self.max_retries:
                    raise
                self.sleep(min(60.0, 5.0 * 2**attempt))
                continue
            latency = time.perf_counter() - started
            record: dict[str, Any] = {
                "text": strip_thinking(out["text"]),
                "prompt_tokens": int(out.get("prompt_tokens", 0)),
                "completion_tokens": int(out.get("completion_tokens", 0)),
                "latency_s": latency,
                "stage": stage,
                "model": self.model,
                "tag": tag,
                "created": time.time(),
            }
            self.cache.put(key, record)
            return LLMResponse(
                record["text"], record["prompt_tokens"], record["completion_tokens"], latency, False
            )
        raise RuntimeError("unreachable")


def openai_transport(base_url: str, api_key: str, timeout_s: float) -> Transport:
    """Transport that calls an OpenAI-compatible /chat/completions endpoint."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    def call(request: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = client.chat.completions.create(
                model=request["model"],
                messages=request["messages"],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                extra_body=request["extra_body"] or None,
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            raise RetryableLLMError(str(exc)) from exc
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

    return call


def build_llm(settings: Settings) -> LLMClient:
    """Build the real client from settings and the API key in `.env`."""
    from dotenv import load_dotenv

    load_dotenv()
    cfg = settings.llm
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key or api_key == "__PLACEHOLDER__":
        raise RuntimeError(f"{cfg.api_key_env} is not set. Add it to .env (see .env.example).")
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if cfg.disable_thinking else {}
    return LLMClient(
        model=cfg.model,
        transport=openai_transport(cfg.base_url, api_key, cfg.timeout_s),
        cache=DiskCache(settings.paths.cache_dir),
        limiter=RateLimiter(cfg.rpm),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        extra_body=extra,
    )


def usage_by_stage(cache_root: Path) -> dict[str, dict[str, float]]:
    """Sum calls, tokens and original latency per stage over every cached call."""
    totals: dict[str, dict[str, float]] = {}
    for path in cache_root.glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        bucket = totals.setdefault(
            record.get("stage", "unknown"),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += record.get("prompt_tokens", 0)
        bucket["completion_tokens"] += record.get("completion_tokens", 0)
        bucket["latency_s"] += record.get("latency_s", 0.0)
    return totals
