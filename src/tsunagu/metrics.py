"""Retrieval metrics, Wilson intervals, Cohen's kappa, percentiles."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def page_ranking(page_paths: list[str]) -> list[str]:
    """Pages in order of first appearance among retrieved chunks."""
    return list(dict.fromkeys(page_paths))


def hit_at_k(pages: list[str], gold: list[str]) -> float:
    return 1.0 if set(pages) & set(gold) else 0.0


def recall_at_k(pages: list[str], gold: list[str]) -> float:
    return len(set(pages) & set(gold)) / len(set(gold)) if gold else 0.0


def mrr(pages: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, page in enumerate(pages, start=1):
        if page in gold_set:
            return 1.0 / rank
    return 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Unweighted Cohen's kappa between two raters."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile; NaN for an empty list."""
    return float(np.percentile(values, q)) if values else float("nan")
