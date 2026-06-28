"""Retriever C: seed entities, expand along the graph, rank reached chunks by similarity and hop."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

import numpy as np

from tsunagu.embeddings import Embedder
from tsunagu.graph import ALL_RELATIONS, Graph
from tsunagu.resolve import normalize_name
from tsunagu.retrievers import RetrievalResult, to_retrieved
from tsunagu.store import ChunkStore

CODE_SPAN_RE = re.compile(r"`([^`]+)`")
WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")
PLAIN_WORD_RE = re.compile(r"[a-z]+")
CODE_TYPES = frozenset({"Symbol", "Class", "Function", "Parameter"})
STOPWORDS = {
    "how",
    "what",
    "which",
    "when",
    "where",
    "why",
    "who",
    "does",
    "do",
    "the",
    "and",
    "for",
    "with",
    "can",
    "use",
    "using",
    "are",
    "is",
    "you",
    "your",
    "that",
    "this",
    "from",
    "into",
    "should",
    "must",
    "need",
    "there",
    "their",
    "they",
    "about",
    "between",
    "after",
    "before",
}


def query_terms(query: str) -> set[str]:
    """Normalised candidate entity names: code spans and word 1- to 3-grams."""
    terms = {normalize_name(m.group(1)) for m in CODE_SPAN_RE.finditer(query)}
    words = WORD_RE.findall(query)
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            if n == 1 and (gram.lower() in STOPWORDS or len(gram) < 3):
                continue
            terms.add(normalize_name(gram))
    terms.discard("")
    return terms


def plain_words(query: str) -> set[str]:
    """Normalised lowercase dictionary-like words written outside code spans."""
    prose = CODE_SPAN_RE.sub(" ", query)
    return {normalize_name(w) for w in WORD_RE.findall(prose) if PLAIN_WORD_RE.fullmatch(w)}


def rank_pool(
    pool: dict[str, int], scores: dict[str, float], k: int, hop_penalty: float
) -> list[str]:
    """Top `k` chunk ids by similarity minus `hop_penalty` per hop; ties go to the smaller id."""
    return sorted(pool, key=lambda cid: (-(scores[cid] - hop_penalty * pool[cid]), cid))[:k]


class GraphRetriever:
    def __init__(
        self,
        graph: Graph,
        store: ChunkStore,
        embedder: Embedder,
        *,
        chunk_matrix: np.ndarray,
        node_ids: list[str],
        node_matrix: np.ndarray,
        hops: int = 2,
        relations: set[str] | frozenset[str] | None = None,
        seed_vector_k: int = 5,
        seed_min_sim: float = 0.6,
        max_expand_degree: int = 40,
        hub_max_chunks: int | None = None,
        hop_penalty: float = 0.0,
        code_only_single_words: bool = False,
        name: str = "graph",
    ) -> None:
        self.graph = graph
        self.store = store
        self.embedder = embedder
        self.chunk_matrix = chunk_matrix
        self.node_ids = node_ids
        self.node_pos = {nid: i for i, nid in enumerate(node_ids)}
        self.node_matrix = node_matrix
        self.hops = hops
        self.relations = frozenset(relations) if relations is not None else ALL_RELATIONS
        self.seed_vector_k = seed_vector_k
        self.seed_min_sim = seed_min_sim
        self.max_expand_degree = max_expand_degree
        self.hop_penalty = hop_penalty
        self.code_only_single_words = code_only_single_words
        self.name = name
        # Entities mentioned in more than `hub_max_chunks` chunks carry little signal and
        # connect most of the corpus, so they are neither seeds nor traversed. Pages are
        # routing nodes for LINKS_TO and are never hubs.
        self.hubs: set[str] = (
            {
                n.id
                for n in graph.nodes.values()
                if n.type != "Page" and len(n.chunk_ids) > hub_max_chunks
            }
            if hub_max_chunks is not None
            else set()
        )
        self.name_index: dict[str, list[str]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.type == "Page":
                continue
            for label in {node.name, *node.aliases}:
                self.name_index[normalize_name(label)].append(node.id)

    def find_seeds(self, query: str, q_vec: np.ndarray) -> list[tuple[str, str, float]]:
        """Seeds as (node_id, how, score): exact name matches first, then vector matches."""
        seeds: dict[str, tuple[str, float]] = {}
        code_terms = {normalize_name(m.group(1)) for m in CODE_SPAN_RE.finditer(query)}
        plain = plain_words(query) - code_terms if self.code_only_single_words else set()
        for term in sorted(query_terms(query)):
            for nid in self.name_index.get(term, []):
                if nid in self.hubs:
                    continue
                if term in plain and self.graph.nodes[nid].type not in CODE_TYPES:
                    continue
                seeds.setdefault(nid, ("name", 1.0))
        sims = self.node_matrix @ q_vec
        for i in np.argsort(-sims)[: self.seed_vector_k]:
            if sims[i] >= self.seed_min_sim and self.node_ids[i] not in self.hubs:
                seeds.setdefault(self.node_ids[i], ("vector", float(sims[i])))
        return [(nid, how, score) for nid, (how, score) in seeds.items()]

    def expand(
        self, seeds: list[str], q_vec: np.ndarray
    ) -> tuple[dict[str, int], list[tuple[str, str, str, int]]]:
        """Breadth-first expansion, at most `max_expand_degree` neighbours per node."""
        hop_of = {s: 0 for s in seeds}
        frontier = list(seeds)
        edges: list[tuple[str, str, str, int]] = []
        for hop in range(1, self.hops + 1):
            next_frontier: list[str] = []
            for nid in frontier:
                neighbours = [
                    (e, o)
                    for e, o in self.graph.neighbors(nid, self.relations)
                    if o not in hop_of and o not in self.hubs
                ]
                neighbours.sort(key=lambda eo: -self._node_sim(eo[1], q_vec))
                for edge, other in neighbours[: self.max_expand_degree]:
                    if other in hop_of:
                        continue
                    hop_of[other] = hop
                    next_frontier.append(other)
                    edges.append((edge.src, edge.rel, edge.dst, hop))
            frontier = next_frontier
        return hop_of, edges

    def _node_sim(self, nid: str, q_vec: np.ndarray) -> float:
        pos = self.node_pos.get(nid)
        return float(self.node_matrix[pos] @ q_vec) if pos is not None else 0.0

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        started = time.perf_counter()
        q_vec = self.embedder.embed_query(query)
        seeds = self.find_seeds(query, q_vec)
        hop_of, edges = self.expand([s for s, _, _ in seeds], q_vec)
        pool: dict[str, int] = {}
        for nid, hop in hop_of.items():
            for cid in self.graph.nodes[nid].chunk_ids:
                if cid in self.store.position:
                    pool[cid] = min(pool.get(cid, hop), hop)
        scores = self.chunk_matrix @ q_vec
        pool_scores = {cid: float(scores[self.store.position[cid]]) for cid in pool}
        ranked = rank_pool(pool, pool_scores, k, self.hop_penalty)
        fallback = 0
        if len(ranked) < k:
            for i in np.argsort(-scores):
                cid = self.store.ids[i]
                if cid not in ranked:
                    ranked.append(cid)
                    fallback += 1
                if len(ranked) == k:
                    break
        chunks = [
            to_retrieved(self.store, cid, float(scores[self.store.position[cid]])) for cid in ranked
        ]
        trace: dict[str, Any] = {
            "seeds": seeds,
            "edges": edges[:60],
            "pool_size": len(pool),
            "fallback": fallback,
            "hop_of_result": {cid: pool.get(cid) for cid in ranked},
        }
        return RetrievalResult(self.name, chunks, time.perf_counter() - started, trace)
