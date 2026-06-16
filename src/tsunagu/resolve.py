"""Entity resolution: normalise, alias map, symbol links, embedding merge, review sample."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from tsunagu.embeddings import Embedder
from tsunagu.graph import ENTITY_TYPES, Graph, Node

CODE_TYPES = ("Class", "Function", "Parameter")


@dataclass
class Merge:
    reason: str  # normalize | alias | symbol_link | embedding
    type: str
    a: str
    b: str
    score: float


def singularize(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize_name(name: str) -> str:
    """Lowercase, strip backticks and `()`, collapse separators, singularise the last word."""
    s = name.strip().strip("`").strip().lower().removesuffix("()")
    s = re.sub(r"[\s_\-]+", " ", s).strip()
    if not s:
        return s
    words = s.split(" ")
    words[-1] = singularize(words[-1])
    return " ".join(words)


def load_aliases(path: Path) -> dict[str, str]:
    """`canonical: [aliases]` YAML to {normalised alias: normalised canonical}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for canonical, aliases in data.items():
        for alias in aliases or []:
            mapping[normalize_name(str(alias))] = normalize_name(str(canonical))
    return mapping


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _apply(graph: Graph, uf: UnionFind) -> Graph:
    groups: dict[str, list[Node]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        groups[uf.find(nid)].append(node)
    out = Graph()
    mapping: dict[str, str] = {}
    for members in groups.values():
        rep = max(members, key=lambda n: (n.type != "Symbol", len(n.chunk_ids), -len(n.name)))
        merged = Node(
            id=rep.id,
            name=rep.name,
            type=rep.type,
            title=rep.title,
            chunk_ids=set().union(*(m.chunk_ids for m in members)),
            aliases={m.name for m in members if m.name != rep.name}.union(
                *(m.aliases for m in members)
            ),
        )
        out.nodes[merged.id] = merged
        for m in members:
            mapping[m.id] = merged.id
    for e in graph.edges:
        out.add_edge(mapping[e.src], e.rel, mapping[e.dst], e.source, e.chunk_id)
    return out


def resolve_graph(
    graph: Graph, aliases: dict[str, str], embedder: Embedder, threshold: float
) -> tuple[Graph, list[Merge]]:
    """Merge duplicate entities; `Page` nodes are never touched.

    Returns:
        The resolved graph and the log of every union performed.
    """
    uf = UnionFind()
    merges: list[Merge] = []
    nodes = [n for n in graph.nodes.values() if n.type != "Page"]

    def canon(n: Node) -> tuple[str, bool]:
        norm = normalize_name(n.name)
        return aliases.get(norm, norm), norm in aliases

    groups: dict[tuple[str, str], list[tuple[Node, bool]]] = defaultdict(list)
    for n in nodes:
        name, via_alias = canon(n)
        groups[(n.type, name)].append((n, via_alias))
    for (type_, _), members in groups.items():
        first, first_alias = members[0]
        for other, other_alias in members[1:]:
            if uf.union(first.id, other.id):
                reason = "alias" if first_alias or other_alias else "normalize"
                merges.append(Merge(reason, type_, first.name, other.name, 1.0))

    code_index: dict[str, str] = {}
    for type_ in CODE_TYPES:
        for n in nodes:
            if n.type == type_:
                code_index.setdefault(canon(n)[0], n.id)
    for n in nodes:
        if n.type == "Symbol":
            target = code_index.get(canon(n)[0])
            if target and uf.union(target, n.id):
                merges.append(Merge("symbol_link", "Symbol", graph.nodes[target].name, n.name, 1.0))

    for type_ in ENTITY_TYPES:
        reps = sorted({uf.find(n.id) for n in nodes if n.type == type_})
        if len(reps) < 2:
            continue
        vectors = embedder.embed_documents([graph.nodes[r].name for r in reps])
        sims = vectors @ vectors.T
        pairs = [
            (float(sims[i, j]), i, j)
            for i in range(len(reps))
            for j in range(i + 1, len(reps))
            if sims[i, j] > threshold
        ]
        for score, i, j in sorted(pairs, reverse=True):
            if uf.union(reps[i], reps[j]):
                merges.append(
                    Merge(
                        "embedding",
                        type_,
                        graph.nodes[reps[i]].name,
                        graph.nodes[reps[j]].name,
                        score,
                    )
                )
    return _apply(graph, uf), merges


def write_merges_csv(merges: list[Merge], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(Merge)])
        writer.writeheader()
        for m in sorted(merges, key=lambda m: (m.reason, -m.score, m.type, m.a)):
            writer.writerow(asdict(m))


def sample_for_review(merges: list[Merge], n: int = 30, seed: int = 42) -> list[Merge]:
    """Random sample of embedding merges (topped up with other non-trivial merges if too few)."""
    rng = random.Random(seed)
    embedding = [m for m in merges if m.reason == "embedding"]
    others = [m for m in merges if m.reason in ("alias", "symbol_link")]
    if len(embedding) >= n:
        return rng.sample(embedding, n)
    return embedding + rng.sample(others, min(n - len(embedding), len(others)))


def write_review_csv(sample: list[Merge], path: Path) -> None:
    """Review sheet: the owner writes y (correct merge) or n (false merge) in `correct`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["reason", "type", "a", "b", "score", "correct"])
        for m in sample:
            writer.writerow([m.reason, m.type, m.a, m.b, f"{m.score:.3f}", ""])


def read_review(path: Path) -> tuple[int, int]:
    """Return (reviewed rows, false merges) from the review sheet."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    marks = [r["correct"].strip().lower() for r in rows if r.get("correct", "").strip()]
    return len(marks), sum(1 for m in marks if m.startswith("n"))
