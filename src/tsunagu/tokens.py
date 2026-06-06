"""Token counting with tiktoken cl100k_base (a stable, model-agnostic yardstick)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens in `text`."""
    return len(_encoding().encode(text, disallowed_special=()))
