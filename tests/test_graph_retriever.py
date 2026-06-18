from __future__ import annotations

from tsunagu.chunking import Chunk
from tsunagu.embeddings import HashEmbedder
from tsunagu.graph import ALL_RELATIONS, build_deterministic_graph
from tsunagu.graph_retriever import GraphRetriever, query_terms
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
