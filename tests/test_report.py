from __future__ import annotations

from pathlib import Path
from typing import Any

from tsunagu.report import (
    AUTO_END,
    AUTO_START,
    correctness_section,
    observations,
    question_outcomes,
    retrieval_section,
    write_results,
)


def _row(
    qid: str, qtype: str, system: str, repeat: int, score: int, hit: float | None = 1.0
) -> dict[str, Any]:
    return {
        "qid": qid,
        "type": qtype,
        "system": system,
        "repeat": repeat,
        "correctness": score,
        "hit": hit,
        "recall": hit,
        "mrr": hit,
        "abstained": qtype == "unanswerable" and score == 2,
        "parse_ok": True,
        "claims_total": 2,
        "claims_supported": 1,
        "cited_pages": 1,
        "precise_citations": 1,
        "retrieval_latency_s": 0.01,
        "answer_latency_s": 1.0,
        "answer_prompt_tokens": 100,
        "answer_completion_tokens": 20,
        "graph_pool_size": None,
        "graph_fallback": None,
    }


ROWS = (
    [_row("q1", "multi_hop", "dense", r, s) for r, s in enumerate([2, 2, 0])]
    + [_row("q1", "multi_hop", "graph", r, s) for r, s in enumerate([2, 2, 2])]
    + [_row("q2", "multi_hop", "dense", r, 1, hit=0.0) for r in range(3)]
    + [_row("q2", "multi_hop", "graph", r, 2) for r in range(3)]
    + [_row("q3", "unanswerable", s, r, 2, hit=None) for s in ("dense", "graph") for r in range(3)]
)


def test_question_outcomes_majority_vote() -> None:
    out = question_outcomes(ROWS, "dense")
    assert out["q1"]["passed"] is True and out["q2"]["passed"] is False
    assert out["q1"]["stable"] is False and out["q2"]["stable"] is True


def test_retrieval_section_excludes_unanswerable() -> None:
    md = retrieval_section(ROWS, ["dense", "graph"])
    assert "| multi_hop (n=2) | 0.50 / 0.50 / 0.50 | 1.00 / 1.00 / 1.00 |" in md


def test_correctness_section_has_counts_and_ci() -> None:
    md = correctness_section(ROWS, ["dense", "graph"])
    assert "| multi_hop | 1/2 = 50% [" in md
    assert "| all | 2/3 = 67% [" in md


def test_observations_flag_overlap() -> None:
    text = observations(ROWS, ["dense", "graph"])
    assert "multi_hop" in text and "overlap" in text


def test_write_results_preserves_manual_sections(tmp_path: Path) -> None:
    path = tmp_path / "results.md"
    write_results(path, "first auto")
    text = path.read_text(encoding="utf-8").replace("_To be written by the owner", "My finding")
    path.write_text(text, encoding="utf-8")
    write_results(path, "second auto")
    final = path.read_text(encoding="utf-8")
    assert "second auto" in final and "first auto" not in final
    assert "My finding" in final
    assert final.count(AUTO_START) == 1 and final.count(AUTO_END) == 1


def test_cli_report(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tsunagu.cli import app
    from tsunagu.util import write_json, write_jsonl

    runner = CliRunner()
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"qid": "q1"}\n', encoding="utf-8")

    write_jsonl(results_dir / "eval_rows.jsonl", ROWS)
    write_json(
        processed_dir / "corpus_stats.json",
        {"commit": "1234567890abcdef", "n_pages": 10, "n_tokens": 1000, "n_chunks": 20},
    )
    write_json(
        graph_dir / "graph_stats.json",
        {
            "before_resolution": {"entities": 15, "llm_entities": 10},
            "after_resolution": {
                "entities": 12,
                "llm_entities": 8,
                "llm_relations": 14,
                "explicit_links": 6,
            },
            "merges_by_reason": {"stemming": 3},
        },
    )
    write_json(
        results_dir / "indexing.json",
        {
            "extraction_model": "test-model",
            "extraction_backend": "test-backend",
            "dense_index_seconds": 1.2,
            "embed_model": "test-embed",
            "chunks_without_triplets": 1,
            "chunks_extracted": 20,
        },
    )

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
corpus:
  repo: "langchain-ai/langgraph"
  commit: "abc"
  pages: []
llm:
  base_url: "https://example.com"
  model: "fake"
paths:
  results_dir: "{results_dir.as_posix()}"
  processed_dir: "{processed_dir.as_posix()}"
  graph_dir: "{graph_dir.as_posix()}"
  cache_dir: "{cache_dir.as_posix()}"
  questions: "{questions_path.as_posix()}"
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    res = runner.invoke(app, ["report", "--config", str(cfg_path)])
    assert res.exit_code == 0
    results_md = results_dir / "results.md"
    assert results_md.exists()
    assert AUTO_START in results_md.read_text(encoding="utf-8")
    obs_md = results_dir / "observations.md"
    assert obs_md.exists()
    assert "# Observations" in obs_md.read_text(encoding="utf-8")
