"""Small JSON and JSONL helpers shared by every stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    """Write `data` as pretty UTF-8 JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def update_json(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Merge `values` into the JSON object stored at `path` (created if missing)."""
    data: dict[str, Any] = read_json(path) if path.exists() else {}
    data.update(values)
    write_json(path, data)
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
