# Tsunagu (つなぐ, "to connect")

Does a graph layer (explicit page links, code symbols, LLM-extracted relations) improve answers to relational, multi-hop and enumerative questions on cross-referenced technical documentation, compared with dense RAG and a cheap hybrid (BM25 + dense with RRF)? And at what cost?

One-day personal prototype. Corpus: LangGraph documentation, `langchain-ai/docs` @ `23152b1133e7759381661e93bcdca778101a80bc` (MIT).

## Setup

```bash
./clean_setup.sh          # or .\clean_setup.ps1
cp .env.example .env      # then set NVIDIA_API_KEY
```

## Run

See "Commands" in `CLAUDE.md`. Results: `results/results.md`.
