"""Run retrieval, answering and judging for every question and system."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tsunagu.answer import answer_question
from tsunagu.judge import judge_answer
from tsunagu.llm import LLMClient
from tsunagu.metrics import hit_at_k, mrr, page_ranking, recall_at_k
from tsunagu.questions import Question
from tsunagu.retrievers import Retriever
from tsunagu.util import write_jsonl

CSV_COLUMNS = [
    "qid",
    "type",
    "system",
    "repeat",
    "hit",
    "recall",
    "mrr",
    "correctness",
    "abstained",
    "parse_ok",
    "claims_total",
    "claims_supported",
    "cited_pages",
    "precise_citations",
    "retrieval_latency_s",
    "answer_latency_s",
    "answer_prompt_tokens",
    "answer_completion_tokens",
    "graph_pool_size",
    "graph_fallback",
    "retrieved_pages",
    "citations",
    "judge_reason",
    "answer",
]
EMPTY_ANSWER: dict[str, Any] = {
    "answer": None,
    "citations": [],
    "abstained": None,
    "parse_ok": None,
    "answer_latency_s": None,
    "answer_prompt_tokens": None,
    "answer_completion_tokens": None,
    "correctness": None,
    "judge_reason": "",
    "claims_total": None,
    "claims_supported": None,
    "cited_pages": None,
    "precise_citations": None,
}


def run_eval(
    questions: list[Question],
    retrievers: dict[str, Retriever],
    llm: LLMClient | None,
    *,
    repeats: int,
    k: int,
    retrieval_only: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> list[dict[str, Any]]:
    """Evaluate every (question, system); answer and judge `repeats` times unless retrieval_only."""
    rows: list[dict[str, Any]] = []
    for qi, q in enumerate(questions, start=1):
        for name, retriever in retrievers.items():
            result = retriever.retrieve(q.question, k)
            pages = page_ranking([c.page_path for c in result.chunks])
            answerable = q.type != "unanswerable"
            base: dict[str, Any] = {
                "qid": q.id,
                "type": q.type,
                "system": name,
                "question": q.question,
                "gold_answer": q.gold_answer,
                "gold_pages": q.gold_pages,
                "retrieved_pages": pages,
                "retrieved_chunks": [c.chunk_id for c in result.chunks],
                "hit": hit_at_k(pages, q.gold_pages) if answerable else None,
                "recall": recall_at_k(pages, q.gold_pages) if answerable else None,
                "mrr": mrr(pages, q.gold_pages) if answerable else None,
                "retrieval_latency_s": result.latency_s,
                "graph_pool_size": result.trace.get("pool_size"),
                "graph_fallback": result.trace.get("fallback"),
            }
            if retrieval_only or llm is None:
                rows.append({**base, "repeat": 0, **EMPTY_ANSWER})
                continue
            for r in range(repeats):
                ans = answer_question(llm, q.question, result.chunks, tag=f"{name}:{q.id}:r{r}")
                j = judge_answer(llm, q, result.chunks, ans)
                rows.append(
                    {
                        **base,
                        "repeat": r,
                        "answer": ans.answer,
                        "citations": ans.citations,
                        "abstained": ans.abstained,
                        "parse_ok": ans.parse_ok,
                        "answer_latency_s": ans.latency_s,
                        "answer_prompt_tokens": ans.prompt_tokens,
                        "answer_completion_tokens": ans.completion_tokens,
                        "correctness": j.correctness,
                        "judge_reason": j.correctness_reason,
                        "claims_total": j.claims_total,
                        "claims_supported": j.claims_supported,
                        "cited_pages": j.cited_pages,
                        "precise_citations": j.precise_citations,
                    }
                )
        log(f"{qi}/{len(questions)} questions done")
    return rows


def merge_rows(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace rows of the systems present in `new`, keep the others."""
    rerun = {r["system"] for r in new}
    return [r for r in old if r["system"] not in rerun] + new


def write_rows(rows: list[dict[str, Any]], results_dir: Path, stem: str = "eval_rows") -> None:
    """Write `<stem>.jsonl` (full) and `per_question.csv` (flat, lists joined with ';')."""
    write_jsonl(results_dir / f"{stem}.jsonl", rows)
    if stem != "eval_rows":
        return
    with (results_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {k: (";".join(v) if isinstance(v, list) else v) for k, v in row.items()}
            writer.writerow(flat)
