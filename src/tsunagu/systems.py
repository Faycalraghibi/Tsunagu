"""Load shared resources once and build retrievers by name."""

from __future__ import annotations

import json
from functools import cached_property
from typing import Any

import numpy as np

from tsunagu.config import Settings
from tsunagu.embeddings import SentenceTransformerEmbedder
from tsunagu.graph import Graph
from tsunagu.graph_retriever import GraphRetriever
from tsunagu.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from tsunagu.store import ChunkStore
from tsunagu.vector_index import get_collection, persistent_client

SYSTEMS = ("dense", "hybrid", "graph")


class Resources:
    """Shared lazy-loaded resources for retrieval pipelines."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    @cached_property
    def store(self) -> ChunkStore:
        """Loaded chunk store."""
        return ChunkStore.load(self.s.paths.processed_dir / "chunks.jsonl")

    @cached_property
    def embedder(self) -> SentenceTransformerEmbedder:
        """Sentence transformer embedding model."""
        return SentenceTransformerEmbedder(self.s.retrieval.embed_model)

    @cached_property
    def collection(self) -> Any:
        """Chroma collection for vector retrieval."""
        return get_collection(persistent_client(self.s.paths.index_dir))

    @cached_property
    def chunk_matrix(self) -> np.ndarray:
        """Numpy embedding matrix for all chunks."""
        ids = json.loads((self.s.paths.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        if ids != self.store.ids:
            raise RuntimeError(
                "index/chunk_ids.json does not match chunks.jsonl: run `tsunagu index`"
            )
        return np.load(self.s.paths.index_dir / "chunk_embeddings.npy")

    @cached_property
    def graph(self) -> Graph:
        """Resolved property graph."""
        return Graph.load(self.s.paths.graph_dir / "graph_resolved.json")

    @cached_property
    def node_index(self) -> tuple[list[str], np.ndarray]:
        """Node IDs and embedding matrix, cached to disk."""
        ids_path = self.s.paths.index_dir / "node_ids.json"
        mat_path = self.s.paths.index_dir / "node_embeddings.npy"
        node_ids = list(self.graph.nodes)
        if (
            ids_path.exists()
            and mat_path.exists()
            and json.loads(ids_path.read_text(encoding="utf-8")) == node_ids
        ):
            return node_ids, np.load(mat_path)
        matrix = self.embedder.embed_documents([self.graph.nodes[n].embed_text for n in node_ids])
        np.save(mat_path, matrix)
        ids_path.write_text(json.dumps(node_ids), encoding="utf-8")
        return node_ids, matrix


def build_retriever(name: str, settings: Settings, res: Resources) -> Retriever:
    """Build retriever `name` ("dense", "hybrid" or "graph")."""
    if name == "dense":
        return DenseRetriever(res.collection, res.store, res.embedder)
    if name == "hybrid":
        dense = DenseRetriever(res.collection, res.store, res.embedder)
        return HybridRetriever(
            dense,
            BM25Retriever(res.store),
            settings.retrieval.candidate_k,
            settings.retrieval.rrf_k,
        )
    if name == "graph":
        node_ids, node_matrix = res.node_index
        g = settings.graph
        return GraphRetriever(
            res.graph,
            res.store,
            res.embedder,
            chunk_matrix=res.chunk_matrix,
            node_ids=node_ids,
            node_matrix=node_matrix,
            hops=g.hops,
            seed_vector_k=g.seed_vector_k,
            seed_min_sim=g.seed_min_sim,
            max_expand_degree=g.max_expand_degree,
        )
    raise ValueError(f"unknown system {name}; known: {', '.join(SYSTEMS)}")
