from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from tsunagu.answer import Answer
from tsunagu.chunking import Chunk
from tsunagu.cli import app
from tsunagu.config import Settings, load_settings
from tsunagu.graph import Graph
from tsunagu.graph_retriever import GraphRetriever
from tsunagu.retrievers import DenseRetriever, HybridRetriever, RetrievalResult, RetrievedChunk
from tsunagu.store import ChunkStore
from tsunagu.systems import SYSTEMS, Resources, build_retriever


@pytest.fixture
def settings() -> Settings:
    return load_settings(Path("config.yaml"))


def test_systems_tuple() -> None:
    assert SYSTEMS == ("dense", "hybrid", "graph")


def test_build_retriever_unknown(settings: Settings) -> None:
    res = MagicMock(spec=Resources)
    with pytest.raises(ValueError, match="unknown system foo; known: dense, hybrid, graph"):
        build_retriever("foo", settings, res)


def test_build_retriever_dense(settings: Settings) -> None:
    res = MagicMock(spec=Resources)
    retriever = build_retriever("dense", settings, res)
    assert isinstance(retriever, DenseRetriever)
    assert retriever.name == "dense"


def test_build_retriever_hybrid(settings: Settings) -> None:
    res = MagicMock(spec=Resources)
    mock_store = MagicMock(spec=ChunkStore)
    mock_store.chunks = [Chunk("c1", "p", [], "test text", 2, [], [])]
    res.store = mock_store
    retriever = build_retriever("hybrid", settings, res)
    assert isinstance(retriever, HybridRetriever)
    assert retriever.name == "hybrid"


def test_build_retriever_graph(settings: Settings) -> None:
    res = MagicMock(spec=Resources)
    res.node_index = (["node_a"], np.zeros((1, 4), dtype=np.float32))
    res.chunk_matrix = np.zeros((1, 4), dtype=np.float32)
    retriever = build_retriever("graph", settings, res)
    assert isinstance(retriever, GraphRetriever)
    assert retriever.name == "graph"


def test_resources_chunk_matrix_validation(settings: Settings, tmp_path: Path) -> None:
    settings.paths.index_dir = tmp_path
    (tmp_path / "chunk_ids.json").write_text(json.dumps(["c1", "c2"]), encoding="utf-8")
    np.save(tmp_path / "chunk_embeddings.npy", np.zeros((2, 4), dtype=np.float32))

    res = Resources(settings)
    mock_store = MagicMock(spec=ChunkStore)
    mock_store.ids = ["c1", "c3"]
    with (
        patch.object(Resources, "store", new=mock_store),
        pytest.raises(RuntimeError, match="chunk_ids.json does not match chunks.jsonl"),
    ):
        _ = res.chunk_matrix


def test_resources_node_index_creates_and_caches(settings: Settings, tmp_path: Path) -> None:
    settings.paths.index_dir = tmp_path
    graph = Graph()
    graph.add_node("Node One", "Concept")
    graph.add_node("Node Two", "Concept")

    res = Resources(settings)
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    with (
        patch.object(Resources, "graph", new=graph),
        patch.object(Resources, "embedder", new=mock_embedder),
    ):
        ids, matrix = res.node_index
        assert ids == ["Concept:Node One", "Concept:Node Two"]
        assert matrix.shape == (2, 2)
        assert mock_embedder.embed_documents.call_count == 1
        assert (tmp_path / "node_ids.json").exists()
        assert (tmp_path / "node_embeddings.npy").exists()

        # Create a second Resources instance to verify disk cache hit
        res2 = Resources(settings)
        mock_embedder.reset_mock()
        with (
            patch.object(Resources, "graph", new=graph),
            patch.object(Resources, "embedder", new=mock_embedder),
        ):
            ids2, matrix2 = res2.node_index
            assert ids2 == ["Concept:Node One", "Concept:Node Two"]
            assert matrix2.shape == (2, 2)
            assert mock_embedder.embed_documents.call_count == 0


def test_cli_ask() -> None:
    runner = CliRunner()
    chunk = RetrievedChunk(
        chunk_id="p:0",
        page_path="langgraph/install",
        heading_path=["Install"],
        text="content",
        score=0.95,
    )
    result = RetrievalResult(
        "graph",
        [chunk],
        0.05,
        trace={
            "seeds": [("n1", "exact", 1.0)],
            "edges": [("n1", "CONNECTS", "n2", 1)],
            "pool_size": 1,
            "fallback": 0,
            "hop_of_result": {"p:0": 1},
        },
    )
    answer = Answer(
        answer="Install via pip install langgraph.",
        citations=["langgraph/install"],
        abstained=False,
        parse_ok=True,
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = result

    with (
        patch("tsunagu.cli._settings"),
        patch("tsunagu.systems.build_retriever", return_value=mock_retriever),
        patch("tsunagu.answer.answer_question", return_value=answer),
        patch("tsunagu.llm.build_llm"),
    ):
        res = runner.invoke(app, ["ask", "How do I install LangGraph?", "--show-path"])
        assert res.exit_code == 0
        assert "Seeds:" in res.stdout
        assert "n1" in res.stdout
        assert "Candidate pool: 1 chunks, dense fallback: 0" in res.stdout
        assert "Retrieved:" in res.stdout
        assert "0.950  langgraph/install#Install" in res.stdout
        assert "Answer: Install via pip install langgraph." in res.stdout
        assert "Citations: ['langgraph/install']" in res.stdout
        assert "Abstained: False" in res.stdout
