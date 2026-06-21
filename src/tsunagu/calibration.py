"""Blind manual labels vs the LLM judge: agreement and Cohen's kappa."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from tsunagu.metrics import cohen_kappa


def make_blind_sample(
    rows: list[dict[str, Any]], n: int = 20, seed: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick n judged answers balanced across systems; return (label sheet, hidden key)."""
    rng = random.Random(seed)
    pool = [
        r
        for r in rows
        if r["repeat"] == 0
        and r["type"] != "unanswerable"
        and not r["abstained"]
        and r["correctness"] is not None
    ]
    by_system: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        by_system.setdefault(str(r["system"]), []).append(r)
    for items in by_system.values():
        rng.shuffle(items)
    picked: list[dict[str, Any]] = []
    order = sorted(by_system)
    while len(picked) < n and any(by_system[s] for s in order):
        for s in order:
            if by_system[s] and len(picked) < n:
                picked.append(by_system[s].pop())
    rng.shuffle(picked)
    labels: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for i, r in enumerate(picked, start=1):
        item = f"L{i:02d}"
        labels.append(
            {
                "item": item,
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "answer": r["answer"],
                "your_score": "",
            }
        )
        key.append({"item": item, "qid": r["qid"], "system": r["system"], "repeat": r["repeat"]})
    return labels, key


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_calibration(
    labels_path: Path, key_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the owner's scores with the judge's on the labelled items."""
    with labels_path.open(encoding="utf-8") as f:
        labels = {r["item"]: r["your_score"].strip() for r in csv.DictReader(f)}
    with key_path.open(encoding="utf-8") as f:
        key = {r["item"]: r for r in csv.DictReader(f)}
    judge = {(str(r["qid"]), str(r["system"]), int(r["repeat"])): r["correctness"] for r in rows}  # type: ignore[arg-type]
    human: list[int] = []
    model: list[int] = []
    disagreements: list[dict[str, Any]] = []
    for item, score in sorted(labels.items()):
        if score == "":
            continue
        k = key[item]
        j = judge[(str(k["qid"]), str(k["system"]), int(k["repeat"]))]
        human.append(int(score))
        model.append(int(j))
        if int(score) != int(j):
            disagreements.append(
                {"item": item, "qid": k["qid"], "human": int(score), "judge": int(j)}
            )
    if not human:
        return {"n": 0}
    pass_h = [int(h == 2) for h in human]
    pass_m = [int(m == 2) for m in model]
    return {
        "n": len(human),
        "agreement": sum(h == m for h, m in zip(human, model, strict=True)) / len(human),
        "kappa": cohen_kappa(human, model),
        "agreement_pass": sum(h == m for h, m in zip(pass_h, pass_m, strict=True)) / len(human),
        "kappa_pass": cohen_kappa(pass_h, pass_m),
        "disagreements": disagreements,
    }
