from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tsunagu.calibration import compute_calibration, make_blind_sample, write_csv


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in ("dense", "hybrid", "graph"):
        for i in range(10):
            rows.append(
                {
                    "qid": f"q{i}",
                    "type": "multi_hop",
                    "system": system,
                    "repeat": 0,
                    "question": f"Q{i}",
                    "gold_answer": "G",
                    "answer": f"{system} answer {i}",
                    "abstained": False,
                    "correctness": i % 3,
                }
            )
    rows.append({**rows[0], "repeat": 1})
    rows.append({**rows[1], "type": "unanswerable"})
    return rows


def test_blind_sample_is_balanced_and_hides_systems() -> None:
    labels, key = make_blind_sample(_rows(), n=20, seed=7)
    assert len(labels) == 20 and len(key) == 20
    assert [r["item"] for r in labels] == [f"L{i:02d}" for i in range(1, 21)]
    assert all("system" not in r for r in labels)
    counts = {s: sum(1 for k in key if k["system"] == s) for s in ("dense", "hybrid", "graph")}
    assert sorted(counts.values()) == [6, 7, 7]
    assert all(k["repeat"] == 0 for k in key)


def test_compute_calibration(tmp_path: Path) -> None:
    rows = _rows()
    labels, key = make_blind_sample(rows, n=20, seed=7)
    judge = {(r["qid"], r["system"]): r["correctness"] for r in rows if r["repeat"] == 0}
    for i, (label, k) in enumerate(zip(labels, key, strict=True)):
        label["your_score"] = judge[(k["qid"], k["system"])] if i < 18 else 0
    write_csv(labels, tmp_path / "labels.csv")
    write_csv(key, tmp_path / "key.csv")
    result = compute_calibration(tmp_path / "labels.csv", tmp_path / "key.csv", rows)
    assert result["n"] == 20
    assert 0.85 <= result["agreement"] <= 1.0
    assert result["kappa"] <= 1.0 and "kappa_pass" in result


def test_unlabelled_rows_are_ignored(tmp_path: Path) -> None:
    rows = _rows()
    labels, key = make_blind_sample(rows, n=20, seed=7)
    write_csv(labels, tmp_path / "labels.csv")
    write_csv(key, tmp_path / "key.csv")
    with (tmp_path / "labels.csv").open(encoding="utf-8") as f:
        assert all(r["your_score"] == "" for r in csv.DictReader(f))
    assert compute_calibration(tmp_path / "labels.csv", tmp_path / "key.csv", rows)["n"] == 0


def test_cli_calib_sample_and_calib(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tsunagu.cli import app
    from tsunagu.util import read_json, write_jsonl

    runner = CliRunner()
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    labels_path = tmp_path / "labels.csv"
    cfg_path = tmp_path / "config.yaml"
    rows = _rows()
    write_jsonl(results_dir / "eval_rows.jsonl", rows)

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
  labels: "{labels_path.as_posix()}"
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")

    # 1. run calib-sample
    res = runner.invoke(app, ["calib-sample", "--config", str(cfg_path)])
    assert res.exit_code == 0
    assert labels_path.exists()
    assert (results_dir / "calibration_key.csv").exists()

    # 2. calib-sample when file exists should exit with code 1
    res_overwrite = runner.invoke(app, ["calib-sample", "--config", str(cfg_path)])
    assert res_overwrite.exit_code == 1
    assert "already exists; not overwriting" in res_overwrite.output

    # 3. calib before any labels are filled (all unlabelled)
    res_calib = runner.invoke(app, ["calib", "--config", str(cfg_path)])
    assert res_calib.exit_code == 0
    assert (results_dir / "calibration.json").exists()

    # 4. fill in scores and test calib
    with (results_dir / "calibration_key.csv").open(encoding="utf-8") as f:
        key_rows = list(csv.DictReader(f))
    with labels_path.open(encoding="utf-8") as f:
        lbl_rows = list(csv.DictReader(f))
    judge = {(r["qid"], r["system"]): r["correctness"] for r in rows if r["repeat"] == 0}
    for label_item, k in zip(lbl_rows, key_rows, strict=True):
        label_item["your_score"] = str(judge[(k["qid"], k["system"])])
    write_csv(lbl_rows, labels_path)

    res_calib2 = runner.invoke(app, ["calib", "--config", str(cfg_path)])
    assert res_calib2.exit_code == 0
    calib_json = read_json(results_dir / "calibration.json")
    assert calib_json["n"] == 20
    assert calib_json["agreement"] == 1.0
    assert calib_json["kappa"] == 1.0
