"""In-memory view of all chunks, shared by every retriever."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tsunagu.chunking import Chunk, load_chunks


class ChunkStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.position = {cid: i for i, cid in enumerate(self.ids)}
        self._by_id = {c.chunk_id: c for c in chunks}
        pages: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            pages[c.page_path].append(c.chunk_id)
        self.page_chunks = dict(pages)

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        return cls(load_chunks(path))

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id
