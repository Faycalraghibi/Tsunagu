from __future__ import annotations

from pathlib import Path

from tsunagu.chunking import Chunk
from tsunagu.graph import (
    ExtractedTriplet,
    Graph,
    add_triplets,
    build_deterministic_graph,
    load_link_edges,
    node_id,
)

PAGES = [{"page_path": "a", "title": "Page A"}, {"page_path": "b", "title": "Page B"}]
CHUNKS = [
    Chunk("a::000", "a", ["Page A"], "Use InMemorySaver", 3, ["InMemorySaver"], ["b"]),
    Chunk("b::000", "b", ["Page B"], "Other", 1, [], []),
]


def _graph() -> Graph:
    return build_deterministic_graph(PAGES, CHUNKS, [("a", "b")])


def test_deterministic_graph() -> None:
    g = _graph()
    assert g.nodes["Page:a"].title == "Page A"
    assert g.nodes["Page:a"].chunk_ids == {"a::000"}
    assert g.nodes["Symbol:InMemorySaver"].chunk_ids == {"a::000"}
    rels = {(e.src, e.rel, e.dst) for e in g.edges}
    assert ("Page:a", "LINKS_TO", "Page:b") in rels
    assert ("Page:a", "MENTIONS", "Symbol:InMemorySaver") in rels


def test_neighbors_go_both_ways_and_filter_relations() -> None:
    g = _graph()
    assert {o for _, o in g.neighbors("Symbol:InMemorySaver", {"MENTIONS"})} == {"Page:a"}
    assert {o for _, o in g.neighbors("Page:b", {"LINKS_TO"})} == {"Page:a"}
    assert g.neighbors("Page:b", {"MENTIONS"}) == []
    assert g.degree("Page:a") == 2


def test_add_triplets_adds_llm_edges_and_page_mentions() -> None:
    g = _graph()
    t = ExtractedTriplet("InMemorySaver", "Class", "PART_OF", "persistence", "Concept", "a::000")
    add_triplets(g, [t], {"a::000": "a"})
    cls, concept = node_id("Class", "InMemorySaver"), node_id("Concept", "persistence")
    rels = {(e.src, e.rel, e.dst, e.source) for e in g.edges}
    assert (cls, "PART_OF", concept, "llm") in rels
    assert ("Page:a", "MENTIONS", concept, "llm") in rels
    assert g.nodes[concept].chunk_ids == {"a::000"}
    assert g.stats()["llm_relations"] == 1


def test_self_loops_ignored_and_roundtrip(tmp_path: Path) -> None:
    g = _graph()
    g.add_edge("Page:a", "LINKS_TO", "Page:a", "links")
    g.save(tmp_path / "g.json")
    loaded = Graph.load(tmp_path / "g.json")
    assert loaded.stats() == g.stats()
    assert loaded.nodes["Symbol:InMemorySaver"].chunk_ids == {"a::000"}


def test_load_link_edges(tmp_path: Path) -> None:
    csv_file = tmp_path / "links.csv"
    csv_file.write_text("source,target\npage_a,page_b\npage_c,page_d\n", encoding="utf-8")
    edges = load_link_edges(csv_file)
    assert edges == [("page_a", "page_b"), ("page_c", "page_d")]


def test_node_embed_text_and_stats() -> None:
    g = _graph()
    assert g.nodes["Page:a"].embed_text == "Page A"
    assert g.nodes["Symbol:InMemorySaver"].embed_text == "InMemorySaver"
    stats = g.stats()
    assert stats["nodes"] == 3
    assert stats["entities"] == 1
    assert stats["llm_entities"] == 0
    assert stats["nodes_by_type"] == {"Page": 2, "Symbol": 1}
    assert stats["edges"] == 2
    assert stats["llm_relations"] == 0
    assert stats["explicit_links"] == 1
    assert stats["edges_by_relation"] == {"LINKS_TO": 1, "MENTIONS": 1}
