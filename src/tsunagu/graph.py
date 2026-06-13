"""A small JSON property graph: deterministic skeleton plus LLM triplets."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tsunagu.chunking import Chunk
from tsunagu.util import read_json, write_json

ENTITY_TYPES = ("Concept", "Class", "Function", "Parameter", "Feature")
LLM_RELATIONS = (
    "USES",
    "REQUIRES",
    "CONFIGURES",
    "PART_OF",
    "ALTERNATIVE_TO",
    "DEPRECATED_BY",
    "RETURNS",
)
STRUCTURAL_RELATIONS = ("LINKS_TO", "MENTIONS")
ALL_RELATIONS = frozenset(LLM_RELATIONS + STRUCTURAL_RELATIONS)


def node_id(type_: str, name: str) -> str:
    return f"{type_}:{name}"


@dataclass
class ExtractedTriplet:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    chunk_id: str


@dataclass
class Node:
    id: str
    name: str
    type: str
    title: str = ""
    chunk_ids: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)

    @property
    def embed_text(self) -> str:
        """Text embedded for seed search: the page title for pages, else the name."""
        return self.title or self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "title": self.title,
            "chunk_ids": sorted(self.chunk_ids),
            "aliases": sorted(self.aliases),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        return cls(
            d["id"],
            d["name"],
            d["type"],
            d.get("title", ""),
            set(d.get("chunk_ids", [])),
            set(d.get("aliases", [])),
        )


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str
    dst: str
    source: str  # "links", "symbols" or "llm"
    chunk_id: str = ""


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._adj: dict[str, list[tuple[Edge, str]]] | None = None

    def add_node(
        self, name: str, type_: str, *, chunk_id: str | None = None, title: str = ""
    ) -> str:
        nid = node_id(type_, name)
        node = self.nodes.get(nid)
        if node is None:
            node = Node(nid, name, type_, title)
            self.nodes[nid] = node
            self._adj = None
        if chunk_id:
            node.chunk_ids.add(chunk_id)
        return nid

    def add_edge(self, src: str, rel: str, dst: str, source: str, chunk_id: str = "") -> None:
        if src == dst:
            return
        key = (src, rel, dst)
        if key not in self._edges:
            self._edges[key] = Edge(src, rel, dst, source, chunk_id)
            self._adj = None

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    def _adjacency(self) -> dict[str, list[tuple[Edge, str]]]:
        if self._adj is None:
            adj: dict[str, list[tuple[Edge, str]]] = {}
            for e in self._edges.values():
                adj.setdefault(e.src, []).append((e, e.dst))
                adj.setdefault(e.dst, []).append((e, e.src))
            self._adj = adj
        return self._adj

    def neighbors(self, nid: str, relations: set[str] | frozenset[str]) -> list[tuple[Edge, str]]:
        """Edges touching `nid` in either direction whose relation is in `relations`."""
        return [(e, other) for e, other in self._adjacency().get(nid, []) if e.rel in relations]

    def degree(self, nid: str) -> int:
        return len(self._adjacency().get(nid, []))

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": len(self.nodes),
            "entities": sum(1 for n in self.nodes.values() if n.type != "Page"),
            "llm_entities": sum(1 for n in self.nodes.values() if n.type in ENTITY_TYPES),
            "nodes_by_type": dict(Counter(n.type for n in self.nodes.values())),
            "edges": len(self._edges),
            "llm_relations": sum(1 for e in self._edges.values() if e.rel in LLM_RELATIONS),
            "explicit_links": sum(1 for e in self._edges.values() if e.rel == "LINKS_TO"),
            "edges_by_relation": dict(Counter(e.rel for e in self._edges.values())),
        }

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [asdict(e) for e in self.edges],
            },
        )

    @classmethod
    def load(cls, path: Path) -> Graph:
        data = read_json(path)
        g = cls()
        for d in data["nodes"]:
            node = Node.from_dict(d)
            g.nodes[node.id] = node
        for e in data["edges"]:
            g.add_edge(e["src"], e["rel"], e["dst"], e["source"], e.get("chunk_id", ""))
        return g


def load_link_edges(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [(row["source"], row["target"]) for row in csv.DictReader(f)]


def build_deterministic_graph(
    pages: list[dict[str, Any]], chunks: list[Chunk], link_edges: list[tuple[str, str]]
) -> Graph:
    """Pages, explicit page links and code-symbol mentions: no LLM involved."""
    g = Graph()
    for page in pages:
        g.add_node(page["page_path"], "Page", title=page["title"])
    for chunk in chunks:
        pid = node_id("Page", chunk.page_path)
        g.nodes[pid].chunk_ids.add(chunk.chunk_id)
        for symbol in chunk.symbols:
            sid = g.add_node(symbol, "Symbol", chunk_id=chunk.chunk_id)
            g.add_edge(pid, "MENTIONS", sid, "symbols", chunk.chunk_id)
    for source, target in link_edges:
        g.add_edge(node_id("Page", source), "LINKS_TO", node_id("Page", target), "links")
    return g


def add_triplets(g: Graph, triplets: list[ExtractedTriplet], page_of: dict[str, str]) -> None:
    """Add LLM triplets, plus `Page -MENTIONS-> entity` edges for their source chunk."""
    for t in triplets:
        s = g.add_node(t.subject, t.subject_type, chunk_id=t.chunk_id)
        o = g.add_node(t.object, t.object_type, chunk_id=t.chunk_id)
        g.add_edge(s, t.relation, o, "llm", t.chunk_id)
        page = node_id("Page", page_of[t.chunk_id])
        g.add_edge(page, "MENTIONS", s, "llm", t.chunk_id)
        g.add_edge(page, "MENTIONS", o, "llm", t.chunk_id)
