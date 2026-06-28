from __future__ import annotations

import numpy as np

from tsunagu.chunking import Chunk
from tsunagu.embeddings import HashEmbedder
from tsunagu.graph import ALL_RELATIONS, build_deterministic_graph
from tsunagu.graph_retriever import GraphRetriever, query_terms, rank_pool
from tsunagu.store import ChunkStore

PAGES = [
    {"page_path": p, "title": t}
    for p, t in (("a", "Persistence"), ("b", "Durable execution"), ("c", "Streaming"))
]
CHUNKS = [
    Chunk(
        "a::000",
        "a",
        ["Persistence"],
        "Use InMemorySaver to save checkpoints in tests.",
        9,
        ["InMemorySaver"],
        ["b"],
    ),
    Chunk("a::001", "a", ["Persistence", "Threads"], "Threads group checkpoints.", 4, [], []),
    Chunk(
        "b::000",
        "b",
        ["Durable execution"],
        "Durable execution resumes from saved checkpoints.",
        7,
        [],
        [],
    ),
    Chunk("c::000", "c", ["Streaming"], "Streaming emits tokens.", 4, [], []),
]


def _retriever(**kw: object) -> GraphRetriever:
    graph = build_deterministic_graph(PAGES, CHUNKS, [("a", "b")])
    store = ChunkStore(CHUNKS)
    emb = HashEmbedder()
    node_ids = list(graph.nodes)
    params: dict[str, object] = {"hops": 2, "seed_min_sim": 0.99}
    params.update(kw)
    return GraphRetriever(
        graph,
        store,
        emb,
        chunk_matrix=emb.embed_documents([c.text for c in CHUNKS]),
        node_ids=node_ids,
        node_matrix=emb.embed_documents([graph.nodes[n].embed_text for n in node_ids]),
        **params,  # type: ignore[arg-type]
    )


def test_query_terms() -> None:
    terms = query_terms("How do I use `InMemorySaver` with durable execution?")
    assert "inmemorysaver" in terms and "durable execution" in terms
    assert "how" not in terms


def test_two_hops_reach_linked_page_not_unrelated_one() -> None:
    result = _retriever().retrieve("How do I use `InMemorySaver`?", k=3)
    ids = {c.chunk_id for c in result.chunks}
    assert ids == {"a::000", "a::001", "b::000"}
    assert result.system == "graph"
    assert result.trace["seeds"][0][0] == "Symbol:InMemorySaver"
    assert result.trace["fallback"] == 0 and result.trace["pool_size"] == 3
    assert any(edge[1] == "LINKS_TO" for edge in result.trace["edges"])


def test_one_hop_needs_fallback() -> None:
    result = _retriever(hops=1).retrieve("How do I use `InMemorySaver`?", k=3)
    assert result.trace["pool_size"] == 2 and result.trace["fallback"] == 1


def test_without_links_linked_page_is_unreachable() -> None:
    relations = set(ALL_RELATIONS) - {"LINKS_TO"}
    result = _retriever(relations=relations).retrieve("How do I use `InMemorySaver`?", k=2)
    assert {c.chunk_id for c in result.chunks} == {"a::000", "a::001"}


def test_rank_pool_hop_penalty_prefers_chunks_near_seeds() -> None:
    pool = {"near": 0, "far": 1}
    scores = {"near": 0.50, "far": 0.52}
    assert rank_pool(pool, scores, k=2, hop_penalty=0.0) == ["far", "near"]
    assert rank_pool(pool, scores, k=2, hop_penalty=0.05) == ["near", "far"]


def test_rank_pool_breaks_ties_by_chunk_id() -> None:
    assert rank_pool({"b": 0, "a": 0}, {"a": 0.5, "b": 0.5}, k=2, hop_penalty=0.0) == ["a", "b"]


def test_hub_entities_are_not_seeded_or_expanded() -> None:
    # InMemorySaver appears in 1 chunk; with hub_max_chunks=0 it counts as a hub.
    result = _retriever(hub_max_chunks=0).retrieve("How do I use `InMemorySaver`?", k=3)
    assert result.trace["seeds"] == []
    assert result.trace["pool_size"] == 0 and result.trace["fallback"] == 3


def test_pages_are_never_hubs() -> None:
    retriever = _retriever(hub_max_chunks=0)
    assert not any(nid.startswith("Page:") for nid in retriever.hubs)


def test_plain_single_word_only_seeds_code_entities() -> None:
    retriever = _retriever(code_only_single_words=True)
    retriever.graph.add_node("checkpoint", "Concept", chunk_id="a::001")
    retriever.name_index["checkpoint"].append("Concept:checkpoint")
    seeds = {nid for nid, _, _ in retriever.find_seeds("Where are checkpoints?", _zero())}
    assert "Concept:checkpoint" not in seeds
    seeds = {nid for nid, _, _ in retriever.find_seeds("Where are `checkpoints`?", _zero())}
    assert "Concept:checkpoint" in seeds


def test_plain_single_word_filter_is_off_by_default() -> None:
    retriever = _retriever()
    retriever.graph.add_node("checkpoint", "Concept", chunk_id="a::001")
    retriever.name_index["checkpoint"].append("Concept:checkpoint")
    seeds = {nid for nid, _, _ in retriever.find_seeds("Where are checkpoints?", _zero())}
    assert "Concept:checkpoint" in seeds


def _zero() -> np.ndarray:
    return np.zeros(HashEmbedder().embed_query("x").shape, dtype=np.float32)
