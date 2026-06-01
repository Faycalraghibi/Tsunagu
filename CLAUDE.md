# Tsunagu: GraphRAG vs vector RAG on cross-referenced technical docs

One-day personal prototype (2026-10-22). Compares dense RAG, hybrid RAG (BM25 + dense, RRF) and GraphRAG on the LangGraph docs (`langchain-ai/docs` @ `23152b1133e7759381661e93bcdca778101a80bc`).

- Spec: `docs/graphrag_prototype_spec.md`. Decisions: `docs/superpowers/specs/2026-10-22-tsunagu-design.md`. Plan: `docs/superpowers/plans/2026-10-22-tsunagu-mvp.md`.
- **Honesty contract:** numbers come only from `results/results.md`, which `tsunagu report` generates. Never write or edit `data/questions.jsonl` or `data/labels_manual.csv` (owner only). Never write the Findings, Limits or Safran sections.
- LLM: NVIDIA NIM, `nvidia/nemotron-3.5-lightning-30b-a3b`, key `NVIDIA_API_KEY` in `.env`. Every call goes through `tsunagu.llm.LLMClient` (disk cache in `cache/llm/`).

## Commands

```bash
./tsunagu-venv/Scripts/python -m pytest
tsunagu fetch | ingest | check-questions | index | build-graph | resolve | merge-sample
tsunagu eval --systems dense,hybrid,graph --repeats 3 [--retrieval-only]
tsunagu calib-sample | calib | report | usage
tsunagu ask "..." --system graph --show-path
```

## Conventions

- Python 3.12 venv `tsunagu-venv/`. Type hints, `from __future__ import annotations`, Google docstrings.
- No em dashes anywhere. Conventional Commits, no AI attribution trailers.
- Branch + PR; merge only after the owner writes "I greenlit". Stretch features need approval and their own branch.
- Run `ruff check`, `ruff format`, `mypy src/` and `pytest` before every commit; pre-commit must pass.
