"""Typer CLI: one command per pipeline stage."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from tsunagu.config import Settings, load_settings

app = typer.Typer(add_completion=False, help="GraphRAG vs vector RAG on the LangGraph docs.")
CONFIG_OPTION = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml.")


@app.callback()
def main() -> None:
    """GraphRAG vs vector RAG on the LangGraph docs."""


def _settings(config: Path) -> Settings:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return load_settings(config)


@app.command()
def fetch(config: Path = CONFIG_OPTION, force: bool = False) -> None:
    """Download the pinned corpus pages."""
    from tsunagu.corpus import fetch_pages

    s = _settings(config)
    paths = fetch_pages(
        s.corpus.repo, s.corpus.commit, s.corpus.pages, s.paths.raw_dir, force=force
    )
    total = sum(p.stat().st_size for p in paths)
    typer.echo(f"{len(paths)} pages, {total / 1024:.0f} KiB in {s.paths.raw_dir}")


@app.command()
def ingest(config: Path = CONFIG_OPTION) -> None:
    """Clean, chunk and link the corpus."""
    from tsunagu.ingest import run_ingest

    stats = run_ingest(_settings(config))
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")


if __name__ == "__main__":
    app()
