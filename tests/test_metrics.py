from __future__ import annotations

import pytest

from tsunagu.metrics import (
    cohen_kappa,
    hit_at_k,
    mrr,
    page_ranking,
    percentile,
    recall_at_k,
    wilson_interval,
)


def test_page_ranking_dedupes_in_order() -> None:
    assert page_ranking(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_retrieval_metrics() -> None:
    pages = ["x", "g1", "y", "g2"]
    assert hit_at_k(pages, ["g1", "g3"]) == 1.0
    assert hit_at_k(pages, ["z"]) == 0.0
    assert recall_at_k(pages, ["g1", "g2", "g3", "g4"]) == 0.5
    assert mrr(pages, ["g2", "g1"]) == 0.5
    assert mrr(pages, ["z"]) == 0.0


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_cohen_kappa() -> None:
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == 1.0
    assert cohen_kappa([0, 1, 2, 0], [0, 1, 2, 1]) == pytest.approx(0.6364, abs=1e-3)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([], 95) != percentile([], 95)  # NaN for empty input
