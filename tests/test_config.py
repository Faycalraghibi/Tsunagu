from __future__ import annotations

from pathlib import Path
from typing import Any

from tsunagu.config import Settings, load_settings
from tsunagu.util import read_jsonl, update_json, write_jsonl


def test_repo_config_loads() -> None:
    settings = load_settings(Path("config.yaml"))
    assert isinstance(settings, Settings)
    assert len(settings.corpus.commit) == 40
    assert 40 <= len(settings.corpus.pages) <= 60
    assert settings.retrieval.top_k == 8
    assert settings.evaluation.systems == ["dense", "hybrid", "graph"]
    assert settings.paths.questions == Path("data/questions.jsonl")


def test_jsonl_roundtrip_and_update_json(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [{"a": 1}, {"a": 2, "b": "é"}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows
    update_json(tmp_path / "s.json", {"x": 1})
    update_json(tmp_path / "s.json", {"y": 2})
    assert (tmp_path / "s.json").read_text(encoding="utf-8").count('"') == 4
