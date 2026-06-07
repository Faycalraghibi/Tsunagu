"""ChromaDB helpers (embeddings are always computed by us, never by Chroma)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np

from tsunagu.chunking import Chunk

COLLECTION = "chunks"


def persistent_client(index_dir: Path) -> Any:
    import chromadb

    return chromadb.PersistentClient(path=str(index_dir / "chroma"))


def create_collection(client: Any, name: str = COLLECTION) -> Any:
    """(Re)create a cosine collection with no embedding function."""
    with contextlib.suppress(Exception):
        client.delete_collection(name)
    try:
        return client.create_collection(
            name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        return client.create_collection(
            name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )


def get_collection(client: Any, name: str = COLLECTION) -> Any:
    return client.get_collection(name, embedding_function=None)


def add_chunks(collection: Any, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    """Add chunks with precomputed embeddings, in batches."""
    batch = 500
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=embeddings[start : start + batch].tolist(),
            documents=[c.text for c in part],
            metadatas=[{"page_path": c.page_path} for c in part],
        )
