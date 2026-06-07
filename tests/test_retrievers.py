from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from tsunagu.chunking import Chunk, save_chunks
from tsunagu.embeddings import HashEmbedder
from tsunagu.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    bm25_tokenize,
    rrf_fuse,
)
from tsunagu.store import ChunkStore
from tsunagu.vector_index import add_chunks, create_collection


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


def test_rrf_fuse_rewards_agreement() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [cid for cid, _ in fused][:2] == ["a", "b"] or [cid for cid, _ in fused][:2] == [
        "b",
        "a",
    ]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [cid for cid, _ in fused][-2:] == ["c", "d"]


def test_rrf_ties_keep_first_seen_order() -> None:
    assert [cid for cid, _ in rrf_fuse([["x"], ["y"]], k=60)] == ["x", "y"]


def test_bm25_tokenize_splits_code_identifiers() -> None:
    tokens = bm25_tokenize("Use InMemorySaver and thread_id")
    assert {"inmemorysaver", "in", "memory", "saver", "thread_id", "thread", "id"} <= set(tokens)


def test_bm25_finds_exact_symbol() -> None:
    store = ChunkStore(make_chunks())
    result = BM25Retriever(store).retrieve("InMemorySaver", k=2)
    assert result.chunks[0].chunk_id == "p1::000"
    assert result.system == "bm25"


def _dense(store: ChunkStore) -> DenseRetriever:
    chromadb = pytest.importorskip("chromadb")
    emb = HashEmbedder()
    collection = create_collection(chromadb.EphemeralClient(), name=f"t{uuid.uuid4().hex}")
    add_chunks(collection, store.chunks, emb.embed_documents([c.text for c in store.chunks]))
    return DenseRetriever(collection, store, emb)


def test_dense_and_hybrid_return_k_chunks() -> None:
    store = ChunkStore(make_chunks())
    dense = _dense(store)
    d = dense.retrieve("interrupts pause the graph for human review", k=2)
    assert d.chunks[0].chunk_id == "p2::002" and len(d.chunks) == 2
    hybrid = HybridRetriever(dense, BM25Retriever(store), candidate_k=4, rrf_k=60)
    h = hybrid.retrieve("InMemorySaver checkpointers", k=3)
    assert h.system == "hybrid" and len(h.chunks) == 3
    assert h.chunks[0].chunk_id == "p1::000"
    assert h.latency_s >= 0
