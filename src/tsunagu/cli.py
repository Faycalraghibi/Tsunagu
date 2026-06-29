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


@app.command()
def ask(
    question: str,
    system: str = "graph",
    show_path: bool = typer.Option(False, "--show-path", help="Print graph seeds and edges."),
    config: Path = CONFIG_OPTION,
) -> None:
    """Answer one question with one system."""
    from tsunagu.answer import answer_question
    from tsunagu.llm import build_llm
    from tsunagu.systems import Resources, build_retriever

    s = _settings(config)
    result = build_retriever(system, s, Resources(s)).retrieve(question, s.retrieval.top_k)
    if show_path and result.trace:
        typer.echo("Seeds:")
        for nid, how, score in result.trace["seeds"]:
            typer.echo(f"  {nid}  ({how}, {score:.2f})")
        typer.echo(f"Edges traversed (first 30 of {len(result.trace['edges'])}):")
        for src, rel, dst, hop in result.trace["edges"][:30]:
            typer.echo(f"  hop {hop}: {src} -{rel}-> {dst}")
        typer.echo(
            f"Candidate pool: {result.trace['pool_size']} chunks, dense fallback: {result.trace['fallback']}"
        )
    typer.echo("Retrieved:")
    for c in result.chunks:
        typer.echo(f"  {c.score:.3f}  {c.label}")
    ans = answer_question(build_llm(s), question, result.chunks, tag=f"ask:{system}")
    typer.echo(f"\nAnswer: {ans.answer}\nCitations: {ans.citations}\nAbstained: {ans.abstained}")


@app.command("eval")
def eval_cmd(
    config: Path = CONFIG_OPTION,
    systems: str = typer.Option("", help="Comma-separated; default from config.yaml."),
    repeats: int = typer.Option(0, help="Default from config.yaml."),
    retrieval_only: bool = typer.Option(False, "--retrieval-only"),
    limit: int = typer.Option(0, help="Only the first N questions (debugging)."),
) -> None:
    """Evaluate systems on the owner's questions."""
    import hashlib

    from tsunagu.llm import build_llm
    from tsunagu.questions import load_questions
    from tsunagu.runner import merge_rows, run_eval, write_rows
    from tsunagu.systems import Resources, build_retriever
    from tsunagu.util import read_json, read_jsonl, write_json

    s = _settings(config)
    names = [n.strip() for n in systems.split(",") if n.strip()] or s.evaluation.systems
    reps = repeats or s.evaluation.repeats
    digest = hashlib.sha256(s.paths.questions.read_bytes()).hexdigest()
    lock = s.paths.results_dir / "questions_lock.json"
    if not lock.exists():
        write_json(lock, {"sha256": digest})
    elif read_json(lock)["sha256"] != digest:
        typer.echo(
            "WARNING: data/questions.jsonl changed since the first eval run. Tell the owner; results.md will show it."
        )
    questions = load_questions(s.paths.questions)
    if limit:
        questions = questions[:limit]
    res = Resources(s)
    retrievers = {n: build_retriever(n, s, res) for n in names}
    llm = None if retrieval_only else build_llm(s)
    rows = run_eval(
        questions,
        retrievers,
        llm,
        repeats=reps,
        k=s.retrieval.top_k,
        retrieval_only=retrieval_only,
        log=typer.echo,
    )
    stem = "retrieval_rows" if retrieval_only else "eval_rows"
    target = s.paths.results_dir / f"{stem}.jsonl"
    old = read_jsonl(target) if target.exists() else []
    write_rows(merge_rows(old, rows), s.paths.results_dir, stem=stem)
    typer.echo(f"wrote {len(rows)} rows to {target}")


@app.command("calib-sample")
def calib_sample(config: Path = CONFIG_OPTION) -> None:
    """Write the blind 20-answer label sheet (never overwrites existing labels)."""
    from tsunagu.calibration import make_blind_sample, write_csv
    from tsunagu.util import read_jsonl

    s = _settings(config)
    if s.paths.labels.exists():
        typer.echo(f"{s.paths.labels} already exists; not overwriting the owner's labels.")
        raise typer.Exit(code=1)
    labels, key = make_blind_sample(read_jsonl(s.paths.results_dir / "eval_rows.jsonl"))
    write_csv(labels, s.paths.labels)
    write_csv(key, s.paths.results_dir / "calibration_key.csv")
    typer.echo(f"wrote {len(labels)} items to {s.paths.labels}")


@app.command()
def calib(config: Path = CONFIG_OPTION) -> None:
    """Agreement and Cohen's kappa between the owner's labels and the judge."""
    from tsunagu.calibration import compute_calibration
    from tsunagu.util import read_jsonl, write_json

    s = _settings(config)
    rows = read_jsonl(s.paths.results_dir / "eval_rows.jsonl")
    result = compute_calibration(s.paths.labels, s.paths.results_dir / "calibration_key.csv", rows)
    write_json(s.paths.results_dir / "calibration.json", result)
    typer.echo({k: v for k, v in result.items() if k != "disagreements"})
    if result.get("n", 0) and result["kappa"] < 0.6:
        typer.echo(
            "kappa < 0.6: do NOT trust the judge yet. Stop and report to the owner (Task 27 Step 4)."
        )


@app.command()
def report(config: Path = CONFIG_OPTION) -> None:
    """Regenerate results.md (AUTO block), observations.md and their LaTeX copies."""
    import hashlib

    from tsunagu.llm import usage_by_stage
    from tsunagu.report import (
        auto_to_latex,
        header_section,
        observations,
        render_auto,
        write_results,
    )
    from tsunagu.resolve import read_review
    from tsunagu.util import read_json, read_jsonl

    s = _settings(config)
    r = s.paths.results_dir
    rows = read_jsonl(r / "eval_rows.jsonl")
    review_path = r / "merges_review.csv"
    lock = r / "questions_lock.json"
    changed = (
        lock.exists()
        and read_json(lock)["sha256"] != hashlib.sha256(s.paths.questions.read_bytes()).hexdigest()
    )
    header = header_section(
        read_json(s.paths.processed_dir / "corpus_stats.json"),
        read_json(s.paths.graph_dir / "graph_stats.json"),
        read_json(r / "indexing.json"),
        usage_by_stage(s.paths.cache_dir),
        read_review(review_path) if review_path.exists() else None,
        changed,
    )
    calib = read_json(r / "calibration.json") if (r / "calibration.json").exists() else None
    repeats = max((row["repeat"] for row in rows), default=0) + 1
    auto_md = render_auto(rows, header, calib, repeats)
    write_results(r / "results.md", auto_md)
    systems = list(dict.fromkeys(row["system"] for row in rows))
    obs_md = observations(rows, systems)
    (r / "observations.md").write_text(obs_md, encoding="utf-8")
    # Same numbers for docs/report/tsunagu.tex, converted, never typed by hand.
    (r / "results_auto.tex").write_text(auto_to_latex(auto_md), encoding="utf-8")
    (r / "observations.tex").write_text(auto_to_latex(obs_md), encoding="utf-8")
    typer.echo(f"updated {r / 'results.md'}, {r / 'observations.md'} and the .tex copies")


if __name__ == "__main__":
    app()
