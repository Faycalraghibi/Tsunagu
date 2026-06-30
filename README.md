# Tsunagu (つなぐ, "to connect")

> GraphRAG vs Dense and Hybrid RAG on cross-referenced technical documentation.

Does a graph layer (explicit page links, code symbols, LLM-extracted relations) improve answers to relational, multi-hop, and exact-reference questions on technical documentation compared with Dense RAG and Hybrid RAG (BM25 + dense with Reciprocal Rank Fusion)? And at what operational cost?

This prototype investigates this comparison on the LangGraph documentation corpus (`langchain-ai/docs` pinned at commit `23152b1133e7759381661e93bcdca778101a80bc`, MIT license).

---

## Architecture and Systems

Tsunagu implements and benchmarks three retrieval and generation architectures under an identical answer generation and evaluation harness:

1. **System A: Dense RAG**
   - Embedder: `BAAI/bge-small-en-v1.5` (384 dimensions, cosine distance).
   - Vector Store: Chroma collection storing 732 heading-aware chunks.
   - Top-8 chunk retrieval by dense similarity.

2. **System B: Hybrid RAG (BM25 + Dense RRF)**
   - Sparse: `BM25Okapi` with code-aware sub-tokenization (snake_case, camelCase, dotted paths).
   - Dense: Chroma vector search over top-20 candidates.
   - Fusion: Reciprocal Rank Fusion (RRF, $k=60$) combining ranks to return top-8 chunks.

3. **System C: GraphRAG**
   - **Deterministic Skeleton:** Explicit page links from MDX references + AST-extracted code symbols (`StateGraph`, `MemorySaver`, `MessagesState`, etc.).
   - **LLM Extraction:** Schema-guided extraction of relation triplets across all chunks (`src/tsunagu/extract.py`).
   - **Entity Resolution:** 4-stage pipeline combining string normalization, alias dictionary mapping, code symbol linking, and cosine embedding clustering ($> 0.92$).
   - **Query-Guided Expansion:** Dual-stage seed entity selection (exact match + vector similarity), degree-capped BFS (2-hop neighborhood, maximum hub degree 40), followed by dense ranking within the candidate pool.

---

## Evaluation and Results

Evaluated across 30 curated technical questions (7 `single_fact`, 8 `multi_hop`, 8 `relational`, 4 `exact_reference`, 3 `unanswerable`) with **3 evaluation repeats** per system (90 queries per system, 270 evaluations total).

All correctness scores include 95% Wilson confidence intervals ($n = \text{questions}$).

### Retrieval Performance (hit@8 / recall@8 / MRR)

| Question Type | Dense RAG | Hybrid RAG | GraphRAG |
|---|---|---|---|
| single_fact (n=7) | 1.00 / 1.00 / 0.93 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.93 |
| multi_hop (n=8) | 1.00 / 0.79 / 0.94 | 1.00 / 0.85 / 1.00 | 1.00 / 0.85 / 0.85 |
| relational (n=8) | 1.00 / 0.88 / 0.94 | 1.00 / 0.88 / 1.00 | 1.00 / 0.94 / 0.94 |
| exact_reference (n=4) | 1.00 / 1.00 / 0.67 | 1.00 / 1.00 / 0.50 | 0.75 / 0.62 / 0.46 |
| **All questions (n=27)** | **1.00 / 0.90 / 0.90** | **1.00 / 0.92 / 0.93** | **0.96 / 0.88 / 0.84** |

### Correctness (Majority pass across repeats, 95% Wilson CI)

| Question Type | Dense RAG | Hybrid RAG | GraphRAG |
|---|---|---|---|
| single_fact | 57% [25%, 84%] | 71% [36%, 92%] | 71% [36%, 92%] |
| multi_hop | 75% [41%, 93%] | 75% [41%, 93%] | 62% [31%, 86%] |
| relational | 75% [41%, 93%] | 50% [22%, 78%] | 75% [41%, 93%] |
| exact_reference | 100% [51%, 100%] | 100% [51%, 100%] | 75% [30%, 95%] |
| unanswerable (correct abstention) | 33% [6%, 79%] | 33% [6%, 79%] | 67% [21%, 94%] |
| **All answerable (n=27)** | **74% [55%, 87%]** | **70% [52%, 84%]** | **70% [52%, 84%]** |
| **Overall (n=30)** | **70% [52%, 83%]** | **67% [49%, 81%]** | **70% [52%, 83%]** |
| Mean score (0 to 2) | 1.66 | 1.61 | 1.61 |
| Repeat stability | 90% | 87% | 83% |

### Operations and Quality

| Metric | Dense RAG | Hybrid RAG | GraphRAG |
|---|---|---|---|
| Faithfulness (supported claims) | 100% (662/662) | 100% (715/718) | 100% (607/609) |
| Citation precision | 91% (119/131) | 92% (120/131) | 91% (116/127) |
| Correct abstention | 44% | 44% | 56% |
| Retrieval latency (p50 / p95) | 51 ms / 173 ms | 62 ms / 90 ms | 60 ms / 254 ms |
| End-to-end latency (p50 / p95) | 2.1 s / 8.3 s | 2.4 s / 6.5 s | 2.2 s / 6.8 s |

Full data, observations, and findings: [`results/results.md`](results/results.md) and [`results/observations.md`](results/observations.md).

---

## Installation

### Prerequisites

- Python 3.12+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Faycalraghibi/Tsunagu.git
cd Tsunagu

# Setup virtual environment and dependencies
./clean_setup.sh          # Linux / macOS / Git Bash
# or .\clean_setup.ps1     # Windows PowerShell

# Configure API key for NVIDIA NIM
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY
```

---

## Pipeline Execution

The CLI provides reproducible entry points for each stage:

```bash
# 1. Corpus ingestion
tsunagu fetch                                  # Fetch 47 pinned LangGraph doc pages
tsunagu ingest                                 # Clean MDX, heading-aware chunking, link extraction

# 2. Vector indexing
tsunagu index                                  # Generate BGE embeddings and build Chroma index

# 3. Knowledge graph construction
tsunagu extract-spike                          # Compare LlamaIndex vs JSON extractor backend
tsunagu build-graph                            # Extract entity-relation triplets across all chunks
tsunagu resolve                                # 4-stage entity resolution into graph_resolved.json
tsunagu merge-sample                           # Sample merges for human verification

# 4. Evaluation and benchmarks
tsunagu check-questions                        # Validate questions schema against corpus pages
tsunagu eval --retrieval-only                  # Benchmark retrieval metrics (hit@8, recall@8, MRR)
tsunagu eval --systems dense,hybrid,graph --repeats 3 # Full answer and LLM judge evaluation
tsunagu calib-sample                           # Generate blind 20-sample sheet for judge calibration
tsunagu calib                                  # Compute agreement and Cohen's kappa
tsunagu report                                 # Update results/results.md and results/observations.md

# 5. Ad-hoc querying
tsunagu ask "How do I save state in LangGraph?" --system graph --show-path
```

All LLM calls are deterministic and cached on disk in `cache/llm/` using SHA-256 keys, ensuring zero API consumption when re-running benchmarks or inspecting answers.

---

## Project Structure

```
.
├── src/tsunagu/
│   ├── config.py              # Pydantic settings and project paths
│   ├── corpus.py              # Pinned MDX corpus fetcher
│   ├── mdx.py                 # Mintlify MDX cleaner (fences, JSX removal)
│   ├── links.py               # Document link normalizer
│   ├── tokens.py              # GPT-4 / cl100k token counter
│   ├── chunking.py            # Heading-aware chunker preserving code blocks
│   ├── embeddings.py          # SentenceTransformer & test embedders
│   ├── store.py               # Parquet chunk store
│   ├── vector_index.py        # Chroma vector index
│   ├── retrievers.py          # Dense, BM25Okapi, and RRF Hybrid retrievers
│   ├── graph.py               # Property graph dataclasses and deterministic links
│   ├── extract.py             # Schema-guided relation extraction
│   ├── resolve.py             # 4-stage entity resolution and merge logging
│   ├── graph_retriever.py     # Seed entity resolution, BFS expansion, and fallback
│   ├── answer.py              # Cited answer generation and abstention handler
│   ├── judge.py               # Atomic claim faithfulness & correctness judge
│   ├── runner.py              # Repeat runner and evaluation metrics recorder
│   ├── calibration.py         # Blind manual calibration and Cohen's kappa
│   ├── report.py              # Auto-block and observations markdown generator
│   ├── systems.py             # System factory and unified query runner
│   └── cli.py                 # Typer CLI application
├── data/                      # Corpus, questions, and graph artifacts
├── results/                   # Benchmark rows, calibration keys, and results.md
├── tests/                     # 102 unit and integration tests (pytest)
└── pyproject.toml             # Dependencies, ruff, and mypy configuration
```

---

## Testing and Quality

```bash
# Run test suite (102 tests)
pytest

# Code formatting and linting
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

---

## License

Code is licensed under the MIT License. Corpus materials are derived from the LangGraph documentation repository (`langchain-ai/docs`), which is licensed under the MIT License.
