"""Retrievers A (dense) and B (hybrid BM25 + dense with RRF)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from rank_bm25 import BM25Okapi

from tsunagu.embeddings import Embedder
from tsunagu.store import ChunkStore

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


@dataclass
class RetrievedChunk:
    chunk_id: str
    page_path: str
    heading_path: list[str]
    text: str
    score: float

    @property
    def label(self) -> str:
        """Citation label shown to the answering model."""
        heading = self.heading_path[-1] if self.heading_path else ""
        return f"{self.page_path}#{heading}"


@dataclass
class RetrievalResult:
    system: str
    chunks: list[RetrievedChunk]
    latency_s: float
    trace: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> RetrievalResult: ...


def to_retrieved(store: ChunkStore, chunk_id: str, score: float) -> RetrievedChunk:
    chunk = store.get(chunk_id)
    return RetrievedChunk(chunk.chunk_id, chunk.page_path, chunk.heading_path, chunk.text, score)


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus the parts of camelCase and snake_case identifiers."""
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        tokens.append(low)
        parts = [p.lower() for piece in tok.split("_") for p in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


class DenseRetriever:
    name = "dense"

    def __init__(self, collection: Any, store: ChunkStore, embedder: Embedder) -> None:
        self.collection = collection
        self.store = store
        self.embedder = embedder

    def ranked_ids(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = self.embedder.embed_query(query)
        res = self.collection.query(query_embeddings=[vector.tolist()], n_results=k)
        return [
            (cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0], strict=True)
        ]

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        started = time.perf_counter()
        ranked = self.ranked_ids(query, k)
        chunks = [to_retrieved(self.store, cid, s) for cid, s in ranked]
        return RetrievalResult(self.name, chunks, time.perf_counter() - started)


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self.bm25 = BM25Okapi([bm25_tokenize(c.text) for c in store.chunks])

    def ranked_ids(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.store.ids[i], float(scores[i])) for i in top]

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        started = time.perf_counter()
        chunks = [to_retrieved(self.store, cid, s) for cid, s in self.ranked_ids(query, k)]
        return RetrievalResult(self.name, chunks, time.perf_counter() - started)


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self, dense: DenseRetriever, bm25: BM25Retriever, candidate_k: int = 20, rrf_k: int = 60
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        started = time.perf_counter()
        dense_ids = [cid for cid, _ in self.dense.ranked_ids(query, self.candidate_k)]
        bm25_ids = [cid for cid, _ in self.bm25.ranked_ids(query, self.candidate_k)]
        fused = rrf_fuse([dense_ids, bm25_ids], k=self.rrf_k)[:k]
        chunks = [to_retrieved(self.dense.store, cid, s) for cid, s in fused]
        return RetrievalResult(self.name, chunks, time.perf_counter() - started)
