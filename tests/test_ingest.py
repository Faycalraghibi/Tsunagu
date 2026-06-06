from __future__ import annotations

from pathlib import Path

from tsunagu.config import CorpusConfig, LLMConfig, PathsConfig, Settings
from tsunagu.ingest import run_ingest
from tsunagu.util import read_json, read_jsonl


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    (raw / "src/oss/langgraph").mkdir(parents=True)
    (raw / "src/oss/langgraph/a.mdx").write_text(
        "---\ntitle: Page A\n---\n\n## One\n\nUse `StateGraph`, see [B](/oss/langgraph/b) "
        "and [web](https://example.com).\n",
        encoding="utf-8",
    )
    (raw / "src/oss/langgraph/b.mdx").write_text(
        "---\ntitle: Page B\n---\n\n## Two\n\nBack to [A](/oss/langgraph/a).\n", encoding="utf-8"
    )
    return Settings(
        corpus=CorpusConfig(
            repo="r",
            commit="c" * 40,
            pages=["src/oss/langgraph/a.mdx", "src/oss/langgraph/b.mdx"],
        ),
        llm=LLMConfig(base_url="http://x", model="m"),
        paths=PathsConfig(raw_dir=raw, processed_dir=tmp_path / "processed"),
    )


def test_run_ingest_writes_all_outputs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stats = run_ingest(settings)
    out = tmp_path / "processed"
    pages = read_jsonl(out / "pages.jsonl")
    chunks = read_jsonl(out / "chunks.jsonl")
    assert [p["page_path"] for p in pages] == ["langgraph/a", "langgraph/b"]
    assert pages[0]["title"] == "Page A"
    assert chunks[0]["symbols"] == ["StateGraph"]
    edges = (out / "link_edges.csv").read_text(encoding="utf-8").splitlines()
    assert edges == ["source,target", "langgraph/a,langgraph/b", "langgraph/b,langgraph/a"]
    assert stats["n_pages"] == 2 and stats["links_external"] == 1
    assert read_json(out / "corpus_stats.json")["n_link_edges"] == 2
    assert "## langgraph/a: Page A" in (out / "page_index.md").read_text(encoding="utf-8")
