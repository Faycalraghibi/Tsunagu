from __future__ import annotations

from pathlib import Path

import numpy as np

from tsunagu.chunking import Chunk, save_chunks
from tsunagu.embeddings import HashEmbedder
from tsunagu.store import ChunkStore


def make_chunks() -> list[Chunk]:
    texts = [
        ("p1", "Checkpointers save graph state; use InMemorySaver for tests."),
        ("p1", "Threads group checkpoints with a thread_id."),
        ("p2", "Interrupts pause the graph for human review and resume with Command."),
        ("p3", "Streaming emits tokens and updates from each node."),
    ]
    return [
        Chunk(f"{page}::{i:03d}", page, [page.upper()], text, 12, [], [])
        for i, (page, text) in enumerate(texts)
    ]


def test_hash_embedder_is_normalised_and_deterministic() -> None:
    emb = HashEmbedder()
    a = emb.embed_documents(["checkpoint state", "checkpoint state"])
    assert a.shape == (2, 256)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert emb.embed_query("checkpoint state") @ a[0] > 0.99


def test_hash_embedder_handles_empty() -> None:
    emb = HashEmbedder(dim=128)
    empty_docs = emb.embed_documents([])
    assert empty_docs.shape == (0, 128)
    q = emb.embed_query("")
    assert q.shape == (128,)
    assert np.all(q == 0.0)


def test_chunk_store_indexes() -> None:
    store = ChunkStore(make_chunks())
    assert store.ids[0] == "p1::000"
    assert store.position["p2::002"] == 2
    assert store.page_chunks["p1"] == ["p1::000", "p1::001"]
    assert store.get("p3::003").page_path == "p3"
    assert len(store) == 4
    assert "p1::000" in store
    assert "nonexistent" not in store


def test_chunk_store_load(tmp_path: Path) -> None:
    chunks = make_chunks()
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    store = ChunkStore.load(path)
    assert len(store) == 4
    assert store.get("p2::002").text == chunks[2].text
