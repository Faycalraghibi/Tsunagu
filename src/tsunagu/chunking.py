"""Heading-aware chunking that never splits a code block."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tsunagu.config import ChunkingConfig
from tsunagu.links import extract_links
from tsunagu.tokens import count_tokens

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
IMPORT_RE = re.compile(r"^\s*from\s+[\w.]+\s+import\s+\(?([\w,\s]+?)\)?\s*$", re.M)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")
STOP_SYMBOLS = {
    "true",
    "false",
    "none",
    "null",
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "self",
    "json",
    "python",
    "typescript",
    "javascript",
    "yes",
    "the",
}


@dataclass
class Chunk:
    chunk_id: str
    page_path: str
    heading_path: list[str]
    text: str
    n_tokens: int
    symbols: list[str] = field(default_factory=list)
    out_links: list[str] = field(default_factory=list)

    @property
    def anchor(self) -> str:
        """Citation label `page_path#last heading`."""
        heading = self.heading_path[-1] if self.heading_path else ""
        return f"{self.page_path}#{heading}"


@dataclass
class Block:
    text: str
    is_code: bool
    n_tokens: int


def normalize_symbol(raw: str) -> str | None:
    """Turn a code span into a symbol name, or None if it is not an identifier."""
    s = raw.strip().lstrip("@").removesuffix("()")
    if not IDENT_RE.match(s):
        return None
    s = s.split(".")[-1]
    if len(s) < 3 or s.lower() in STOP_SYMBOLS:
        return None
    return s


def _split_prose_code(text: str) -> tuple[str, str]:
    prose: list[str] = []
    code: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        (code if in_fence else prose).append(line)
    return "\n".join(prose), "\n".join(code)


def extract_symbols(text: str) -> list[str]:
    """Symbols from prose code spans and from `from x import A, B` lines in code."""
    prose, code = _split_prose_code(text)
    found: set[str] = set()
    for match in CODE_SPAN_RE.finditer(prose):
        sym = normalize_symbol(match.group(1))
        if sym:
            found.add(sym)
    for match in IMPORT_RE.finditer(code):
        for name in match.group(1).split(","):
            sym = normalize_symbol(name.strip().split(" as ")[0])
            if sym:
                found.add(sym)
    return sorted(found, key=str.lower)


def split_blocks(body: str) -> list[Block]:
    """Split a section body into paragraphs and whole code blocks."""
    blocks: list[Block] = []
    buf: list[str] = []
    in_fence = False

    def emit(is_code: bool) -> None:
        text = "\n".join(buf).strip("\n")
        if text.strip():
            blocks.append(Block(text, is_code, count_tokens(text)))
        buf.clear()

    for line in body.split("\n"):
        if FENCE_RE.match(line):
            if not in_fence:
                emit(False)
                buf.append(line)
                in_fence = True
            else:
                buf.append(line)
                emit(True)
                in_fence = False
            continue
        if in_fence:
            buf.append(line)
        elif not line.strip():
            emit(False)
        else:
            buf.append(line)
    emit(in_fence)
    return blocks


def _split_oversized(block: Block, max_tokens: int) -> list[Block]:
    if block.is_code or block.n_tokens <= max_tokens:
        return [block]
    pieces: list[Block] = []
    current: list[str] = []
    for sentence in (s for s in SENTENCE_RE.split(block.text) if s.strip()):
        candidate = " ".join([*current, sentence])
        if current and count_tokens(candidate) > max_tokens:
            text = " ".join(current)
            pieces.append(Block(text, False, count_tokens(text)))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        text = " ".join(current)
        pieces.append(Block(text, False, count_tokens(text)))
    return pieces


def pack_blocks(blocks: list[Block], cfg: ChunkingConfig) -> list[list[Block]]:
    """Greedy packing up to `target_tokens`, carrying trailing prose as overlap."""
    flat = [piece for b in blocks for piece in _split_oversized(b, cfg.max_tokens)]
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in flat:
        if current and size + block.n_tokens > cfg.target_tokens:
            groups.append(current)
            tail: list[Block] = []
            tail_size = 0
            for prev in reversed(current):
                if prev.is_code or tail_size + prev.n_tokens > cfg.overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_size += prev.n_tokens
            if len(tail) == len(current):
                tail, tail_size = [], 0
            current, size = tail, tail_size
        current.append(block)
        size += block.n_tokens
    if current:
        groups.append(current)
    return groups


def split_sections(text: str, title: str) -> list[tuple[list[str], str]]:
    """Split Markdown on headings (outside code) into (heading_path, body) pairs."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    path = [title]
    body: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        sections.append((path, "\n".join(body)))
        body = []
        level, heading = len(match.group(1)), match.group(2).strip()
        stack = [(lvl, h) for lvl, h in stack if lvl < level]
        stack.append((level, heading))
        path = [title] + [h for _, h in stack if h != title]
    sections.append((path, "\n".join(body)))
    return [(p, b) for p, b in sections if b.strip()]


def chunk_page(
    page_path: str, title: str, text: str, cfg: ChunkingConfig, known_pages: set[str]
) -> list[Chunk]:
    """Chunk one cleaned page.

    Args:
        page_path: Page id such as `langgraph/persistence`.
        title: Page title, used as the root of every heading path.
        text: Cleaned Markdown.
        cfg: Chunk sizes.
        known_pages: Page ids in the corpus, for link normalisation.

    Returns:
        Chunks in reading order, ids `page_path::000`, `page_path::001`, ...
    """
    chunks: list[Chunk] = []
    for heading_path, body in split_sections(text, title):
        for group in pack_blocks(split_blocks(body), cfg):
            header = " > ".join(heading_path)
            chunk_text = header + "\n\n" + "\n\n".join(b.text for b in group)
            links, _ = extract_links(chunk_text, page_path, known_pages)
            chunks.append(
                Chunk(
                    chunk_id=f"{page_path}::{len(chunks):03d}",
                    page_path=page_path,
                    heading_path=list(heading_path),
                    text=chunk_text,
                    n_tokens=count_tokens(chunk_text),
                    symbols=extract_symbols(chunk_text),
                    out_links=links,
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks from JSONL."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
