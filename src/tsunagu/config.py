"""Project settings, loaded from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CorpusConfig(BaseModel):
    repo: str
    commit: str
    pages: list[str]


class LLMConfig(BaseModel):
    provider: str = "nvidia"
    base_url: str
    model: str
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048
    rpm: int = 35
    timeout_s: float = 120.0
    max_retries: int = 5
    disable_thinking: bool = True


class ChunkingConfig(BaseModel):
    target_tokens: int = 500
    overlap_tokens: int = 50
    max_tokens: int = 900


class RetrievalConfig(BaseModel):
    top_k: int = 8
    candidate_k: int = 20
    rrf_k: int = 60
    embed_model: str = "BAAI/bge-small-en-v1.5"


class GraphConfig(BaseModel):
    extraction_backend: Literal["llamaindex", "json"] = "llamaindex"
    max_triplets_per_chunk: int = 10
    merge_threshold: float = 0.92
    hops: int = 2
    seed_vector_k: int = 5
    seed_min_sim: float = 0.6
    max_expand_degree: int = 40


class EvalConfig(BaseModel):
    repeats: int = 3
    systems: list[str] = Field(default_factory=lambda: ["dense", "hybrid", "graph"])


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/corpus/raw")
    processed_dir: Path = Path("data/processed")
    graph_dir: Path = Path("data/graph")
    index_dir: Path = Path("index")
    cache_dir: Path = Path("cache/llm")
    results_dir: Path = Path("results")
    questions: Path = Path("data/questions.jsonl")
    labels: Path = Path("data/labels_manual.csv")
    aliases: Path = Path("aliases.yaml")


class Settings(BaseModel):
    corpus: CorpusConfig
    llm: LLMConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_settings(path: Path = Path("config.yaml")) -> Settings:
    """Load and validate settings from a YAML file."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings.model_validate(data)
