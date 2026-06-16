from __future__ import annotations

from pathlib import Path

import numpy as np

from tsunagu.graph import Graph
from tsunagu.resolve import (
    Merge,
    load_aliases,
    normalize_name,
    read_review,
    resolve_graph,
    sample_for_review,
    write_review_csv,
)


class TableEmbedder:
    """Fixed vectors for known names; orthogonal one-hot vectors otherwise."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.next = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t in self.table:
                v = np.array(self.table[t] + [0.0] * 8, dtype=np.float32)
            else:
                v = np.zeros(10, dtype=np.float32)
                v[2 + self.next % 8] = 1.0
                self.next += 1
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def test_normalize_name() -> None:
    assert normalize_name("`InMemorySaver()`") == "inmemorysaver"
    assert normalize_name("Checkpointers") == "checkpointer"
    assert normalize_name("human-in-the-loop") == "human in the loop"
    assert normalize_name("thread_id") == "thread id"
    assert normalize_name("libraries") == "library"
    assert normalize_name("class") == "class"
    assert normalize_name("status") == "status"


def test_load_aliases(tmp_path: Path) -> None:
    path = tmp_path / "a.yaml"
    path.write_text("checkpointer: [Checkpoint Saver, saver]\n", encoding="utf-8")
    assert load_aliases(path) == {"checkpoint saver": "checkpointer", "saver": "checkpointer"}


def _graph() -> Graph:
    g = Graph()
    g.add_node("p", "Page", title="P")
    a = g.add_node("Checkpointers", "Concept", chunk_id="c1")
    b = g.add_node("checkpointer", "Concept", chunk_id="c2")
    c = g.add_node("checkpoint saver", "Concept", chunk_id="c3")
    sym = g.add_node("InMemorySaver", "Symbol", chunk_id="c4")
    cls = g.add_node("InMemorySaver", "Class", chunk_id="c5")
    d1 = g.add_node("durable execution", "Feature", chunk_id="c6")
    d2 = g.add_node("durability", "Feature", chunk_id="c7")
    g.add_node("streaming", "Feature", chunk_id="c8")
    g.add_edge(cls, "PART_OF", a, "llm")
    g.add_edge(c, "USES", b, "llm")
    g.add_edge("Page:p", "MENTIONS", sym, "symbols")
    g.add_edge(d1, "ALTERNATIVE_TO", d2, "llm")
    return g


def test_resolve_graph_merges_and_logs() -> None:
    emb = TableEmbedder({"durable execution": [1.0, 0.0], "durability": [0.95, 0.31]})
    resolved, merges = resolve_graph(
        _graph(), {"checkpoint saver": "checkpointer"}, emb, threshold=0.92
    )
    reasons = sorted(m.reason for m in merges)
    assert reasons == ["alias", "embedding", "normalize", "symbol_link"]
    types = {n.type for n in resolved.nodes.values()}
    assert "Symbol" not in types
    concept = [n for n in resolved.nodes.values() if n.type == "Concept"]
    assert len(concept) == 1 and concept[0].chunk_ids == {"c1", "c2", "c3"}
    cls = resolved.nodes["Class:InMemorySaver"]
    assert cls.chunk_ids == {"c4", "c5"}
    rels = {(e.src, e.rel, e.dst) for e in resolved.edges}
    assert ("Page:p", "MENTIONS", "Class:InMemorySaver") in rels
    assert not any(e.rel == "USES" for e in resolved.edges)  # became a self-loop, dropped
    assert not any(e.rel == "ALTERNATIVE_TO" for e in resolved.edges)
    assert "Feature:streaming" in resolved.nodes


def test_review_sampling_and_reading(tmp_path: Path) -> None:
    merges = [Merge("embedding", "Concept", f"a{i}", f"b{i}", 0.95) for i in range(40)]
    merges += [Merge("symbol_link", "Symbol", "X", "X", 1.0)]
    sample = sample_for_review(merges, n=30, seed=42)
    assert len(sample) == 30 and all(m.reason == "embedding" for m in sample)
    assert sample == sample_for_review(merges, n=30, seed=42)
    path = tmp_path / "review.csv"
    write_review_csv(sample, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] += "n"
    lines[2] += "y"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert read_review(path) == (2, 1)
