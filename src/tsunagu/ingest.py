"""Ingestion: cleaned pages, chunks, explicit link edges, corpus stats."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tsunagu.chunking import Chunk, chunk_page, save_chunks
from tsunagu.config import Settings
from tsunagu.links import extract_links, page_id_from_repo_path
from tsunagu.mdx import clean_mdx
from tsunagu.tokens import count_tokens
from tsunagu.util import write_json, write_jsonl


def _page_index(pages: list[dict[str, Any]], chunks: list[Chunk]) -> str:
    lines = [
        "# Page index",
        "",
        "Use these page ids in `gold_pages`. Headings are listed to help write questions.",
        "",
    ]
    for page in pages:
        lines.append(f"## {page['page_path']}: {page['title']}")
        lines.append(
            f"{page['n_tokens']} tokens, links to: {', '.join(page['out_links']) or 'none'}"
        )
        seen: list[str] = []
        for chunk in chunks:
            if chunk.page_path == page["page_path"]:
                heading = " > ".join(chunk.heading_path[1:])
                if heading and heading not in seen:
                    seen.append(heading)
        lines.extend(f"- {h}" for h in seen)
        lines.append("")
    return "\n".join(lines)


def run_ingest(settings: Settings) -> dict[str, Any]:
    """Clean, chunk and link every configured page; write outputs to `processed_dir`.

    Returns:
        The corpus statistics also written to `corpus_stats.json`.
    """
    known = {page_id_from_repo_path(p) for p in settings.corpus.pages}
    pages: list[dict[str, Any]] = []
    chunks: list[Chunk] = []
    edges: list[tuple[str, str]] = []
    external = out_of_corpus = 0
    for repo_path in settings.corpus.pages:
        page_id = page_id_from_repo_path(repo_path)
        raw = (settings.paths.raw_dir / repo_path).read_text(encoding="utf-8")
        clean = clean_mdx(raw)
        title = clean.title or page_id
        targets, stats = extract_links(clean.text, page_id, known)
        external += stats.external
        out_of_corpus += stats.out_of_corpus
        edges.extend((page_id, t) for t in targets)
        page_chunks = chunk_page(page_id, title, clean.text, settings.chunking, known)
        chunks.extend(page_chunks)
        pages.append(
            {
                "page_path": page_id,
                "repo_path": repo_path,
                "title": title,
                "n_tokens": count_tokens(clean.text),
                "n_chunks": len(page_chunks),
                "out_links": targets,
                "api_refs": clean.api_refs,
                "link_stats": asdict(stats),
            }
        )

    out = settings.paths.processed_dir
    write_jsonl(out / "pages.jsonl", pages)
    save_chunks(chunks, out / "chunks.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    (out / "link_edges.csv").write_text(
        "source,target\n" + "".join(f"{s},{t}\n" for s, t in edges), encoding="utf-8"
    )
    (out / "page_index.md").write_text(_page_index(pages, chunks), encoding="utf-8")
    sizes = [c.n_tokens for c in chunks]
    stats_out: dict[str, Any] = {
        "repo": settings.corpus.repo,
        "commit": settings.corpus.commit,
        "n_pages": len(pages),
        "n_tokens": sum(p["n_tokens"] for p in pages),
        "n_chunks": len(chunks),
        "mean_chunk_tokens": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "max_chunk_tokens": max(sizes, default=0),
        "n_link_edges": len(edges),
        "links_external": external,
        "links_out_of_corpus": out_of_corpus,
        "n_chunks_with_symbols": sum(1 for c in chunks if c.symbols),
    }
    write_json(out / "corpus_stats.json", stats_out)
    return stats_out
