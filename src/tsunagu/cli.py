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


@app.command("check-questions")
def check_questions(config: Path = CONFIG_OPTION) -> None:
    """Validate data/questions.jsonl against the corpus (never edits it)."""
    from tsunagu.questions import load_questions, validate_questions
    from tsunagu.util import read_jsonl

    s = _settings(config)
    known = {p["page_path"] for p in read_jsonl(s.paths.processed_dir / "pages.jsonl")}
    questions = load_questions(s.paths.questions)
    problems = validate_questions(questions, known)
    for problem in problems:
        typer.echo(f"PROBLEM {problem}")
    typer.echo("OK" if not problems else f"{len(problems)} problem(s)")
    raise typer.Exit(code=1 if problems else 0)


@app.command()
def usage(config: Path = CONFIG_OPTION) -> None:
    """Print calls, tokens and LLM time per stage, from the cache."""
    from tsunagu.llm import usage_by_stage

    s = _settings(config)
    for stage, values in sorted(usage_by_stage(s.paths.cache_dir).items()):
        typer.echo(
            f"{stage}: {values['calls']:.0f} calls, {values['prompt_tokens']:.0f} prompt + "
            f"{values['completion_tokens']:.0f} completion tokens, {values['latency_s'] / 60:.1f} min"
        )


@app.command()
def smoke(config: Path = CONFIG_OPTION) -> None:
    """One real LLM call to check the key, the model and the cache."""
    from tsunagu.llm import build_llm

    reply = build_llm(_settings(config)).complete(
        [{"role": "user", "content": 'Reply with the JSON {"ok": true} and nothing else.'}],
        stage="smoke",
    )
    typer.echo(
        f"{reply.text!r} cached={reply.cached} tokens={reply.prompt_tokens}+{reply.completion_tokens}"
    )


if __name__ == "__main__":
    app()
