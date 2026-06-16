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


@app.command()
def index(config: Path = CONFIG_OPTION) -> None:
    """Embed chunks and build the Chroma index."""
    from tsunagu.embeddings import SentenceTransformerEmbedder
    from tsunagu.index import build_dense_index
    from tsunagu.store import ChunkStore

    s = _settings(config)
    store = ChunkStore.load(s.paths.processed_dir / "chunks.jsonl")
    facts = build_dense_index(s, store, SentenceTransformerEmbedder(s.retrieval.embed_model))
    typer.echo(facts)


@app.command("extract-spike")
def extract_spike(config: Path = CONFIG_OPTION, n: int = 10) -> None:
    """Run both extraction backends on n chunks and recommend one (objective rule)."""
    from typing import Any

    from tsunagu.chunking import load_chunks
    from tsunagu.extract import build_extractor
    from tsunagu.llm import build_llm
    from tsunagu.util import write_json

    s = _settings(config)
    chunks = [
        c
        for c in load_chunks(s.paths.processed_dir / "chunks.jsonl")
        if c.symbols and c.n_tokens > 150
    ]
    picked = chunks[:: max(1, len(chunks) // n)][:n]
    llm = build_llm(s)
    rows: list[dict[str, Any]] = []
    for backend in ("json", "llamaindex"):
        try:
            extractor = build_extractor(
                backend, llm, s.graph.max_triplets_per_chunk, tag=f"spike:{backend}"
            )
            counts = [len(extractor.extract(c)) for c in picked]
            error = ""
        except Exception as exc:  # spike: report any failure instead of crashing
            counts, error = [0] * len(picked), repr(exc)
        rows.append({"backend": backend, "counts": counts, "error": error})
        typer.echo(f"{backend}: {counts} {error}")
    json_counts, li_counts = rows[0]["counts"], rows[1]["counts"]
    json_nonempty = sum(1 for c in json_counts if c > 0)
    li_misses = sum(1 for j, li in zip(json_counts, li_counts, strict=True) if j > 0 and li == 0)
    recommend = (
        "llamaindex"
        if not rows[1]["error"] and li_misses <= 0.3 * max(1, json_nonempty)
        else "json"
    )
    write_json(
        s.paths.results_dir / "extraction_spike.json",
        {"chunks": [c.chunk_id for c in picked], "rows": rows, "recommend": recommend},
    )
    typer.echo(f"recommend: {recommend}")


@app.command("build-graph")
def build_graph(config: Path = CONFIG_OPTION, limit: int = 0) -> None:
    """Deterministic skeleton plus LLM triplets for every chunk (cached, resumable)."""
    import time

    from tsunagu.chunking import load_chunks
    from tsunagu.extract import build_extractor
    from tsunagu.graph import add_triplets, build_deterministic_graph, load_link_edges
    from tsunagu.llm import build_llm
    from tsunagu.util import read_jsonl, update_json

    s = _settings(config)
    chunks = load_chunks(s.paths.processed_dir / "chunks.jsonl")
    pages = read_jsonl(s.paths.processed_dir / "pages.jsonl")
    g = build_deterministic_graph(
        pages, chunks, load_link_edges(s.paths.processed_dir / "link_edges.csv")
    )
    deterministic = g.stats()
    extractor = build_extractor(
        s.graph.extraction_backend, build_llm(s), s.graph.max_triplets_per_chunk
    )
    page_of = {c.chunk_id: c.page_path for c in chunks}
    targets = chunks[:limit] if limit else chunks
    started = time.perf_counter()
    kept = empty = 0
    for i, chunk in enumerate(targets, start=1):
        triplets = extractor.extract(chunk)
        kept += len(triplets)
        empty += not triplets
        add_triplets(g, triplets, page_of)
        if i % 25 == 0 or i == len(targets):
            typer.echo(
                f"{i}/{len(targets)} chunks, {kept} triplets, {time.perf_counter() - started:.0f}s"
            )
    g.save(s.paths.graph_dir / "graph_raw.json")
    update_json(
        s.paths.graph_dir / "graph_stats.json",
        {"deterministic_only": deterministic, "before_resolution": g.stats()},
    )
    update_json(
        s.paths.results_dir / "indexing.json",
        {
            "extraction_backend": s.graph.extraction_backend,
            "extraction_model": s.llm.model,
            "chunks_extracted": len(targets),
            "triplets_kept": kept,
            "chunks_without_triplets": empty,
            "extraction_wall_seconds_last_run": round(time.perf_counter() - started, 1),
        },
    )
    typer.echo(g.stats())


@app.command()
def resolve(config: Path = CONFIG_OPTION) -> None:
    """Resolve entities in graph_raw.json; write graph_resolved.json and results/merges.csv."""
    from collections import Counter

    from tsunagu.embeddings import SentenceTransformerEmbedder
    from tsunagu.graph import Graph
    from tsunagu.resolve import load_aliases, resolve_graph, write_merges_csv
    from tsunagu.util import update_json

    s = _settings(config)
    raw = Graph.load(s.paths.graph_dir / "graph_raw.json")
    embedder = SentenceTransformerEmbedder(s.retrieval.embed_model)
    resolved, merges = resolve_graph(
        raw, load_aliases(s.paths.aliases), embedder, s.graph.merge_threshold
    )
    resolved.save(s.paths.graph_dir / "graph_resolved.json")
    write_merges_csv(merges, s.paths.results_dir / "merges.csv")
    by_reason = dict(Counter(m.reason for m in merges))
    update_json(
        s.paths.graph_dir / "graph_stats.json",
        {"after_resolution": resolved.stats(), "merges_by_reason": by_reason},
    )
    typer.echo(
        f"entities {raw.stats()['entities']} -> {resolved.stats()['entities']}; merges {by_reason}"
    )


@app.command("merge-sample")
def merge_sample(config: Path = CONFIG_OPTION) -> None:
    """Write the 30-merge review sheet once (never overwrites the owner's answers)."""
    import csv

    from tsunagu.resolve import Merge, sample_for_review, write_review_csv

    s = _settings(config)
    target = s.paths.results_dir / "merges_review.csv"
    if target.exists():
        typer.echo(f"{target} already exists; not overwriting the owner's review.")
        raise typer.Exit(code=1)
    with (s.paths.results_dir / "merges.csv").open(encoding="utf-8") as f:
        merges = [
            Merge(r["reason"], r["type"], r["a"], r["b"], float(r["score"]))
            for r in csv.DictReader(f)
        ]
    write_review_csv(sample_for_review(merges), target)
    typer.echo(f"wrote {target}")


if __name__ == "__main__":
    app()
