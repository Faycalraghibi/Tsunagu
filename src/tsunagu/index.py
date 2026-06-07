"""Build the dense index: chunk embeddings on disk and in Chroma."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from tsunagu.config import Settings
from tsunagu.embeddings import Embedder
from tsunagu.store import ChunkStore
from tsunagu.util import update_json
from tsunagu.vector_index import add_chunks, create_collection, persistent_client


def build_dense_index(settings: Settings, store: ChunkStore, embedder: Embedder) -> dict[str, Any]:
    """Embed every chunk, save `chunk_embeddings.npy` + `chunk_ids.json`, fill Chroma.

    Returns:
        Timing and size facts, also merged into `results/indexing.json`.
    """
    started = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in store.chunks])
    index_dir = settings.paths.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "chunk_embeddings.npy", vectors)
    (index_dir / "chunk_ids.json").write_text(json.dumps(store.ids), encoding="utf-8")
    collection = create_collection(persistent_client(index_dir))
    add_chunks(collection, store.chunks, vectors)
    facts = {
        "dense_index_seconds": round(time.perf_counter() - started, 1),
        "embed_model": settings.retrieval.embed_model,
        "n_chunks_indexed": len(store.ids),
        "embedding_dim": int(vectors.shape[1]),
    }
    update_json(settings.paths.results_dir / "indexing.json", facts)
    return facts
