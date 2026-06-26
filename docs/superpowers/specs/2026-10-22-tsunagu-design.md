# Tsunagu: design addendum

The full design lives in `docs/graphrag_prototype_spec.md` (the owner's spec). This file records only the decisions taken on 2026-10-22 before implementation, where the spec left a choice open or where reality differs from it.

## Decisions taken with the owner

| Topic | Decision |
|---|---|
| LLM provider | NVIDIA NIM, `https://integrate.api.nvidia.com/v1`, model `nvidia/nemotron-3.5-lightning-30b-a3b` (same as Joshu), thinking disabled with `chat_template_kwargs.enable_thinking=false`. Key in `.env` as `NVIDIA_API_KEY`. Same model for extraction, answering and judging (a stated limit). |
| Graph extraction | LlamaIndex `SchemaLLMPathExtractor` (strict schema) through a cached `CustomLLM` wrapper. Fallback: our own schema-guided JSON extractor. A spike on 10 chunks decides (objective rule in the plan, Task 16). |
| Graph storage | Our own JSON property graph (`data/graph/*.json`), not `SimplePropertyGraphStore`. Reason: one file we can read, diff and traverse by hand. Neo4j export stays Stretch. |
| Question set | Planned as owner-written. Changed on 2026-10-22: at the owner's request the 30 questions and gold answers were drafted by the AI assistant from the cleaned corpus, before any eval result existed. See fact 5. |
| Git | Private GitHub repo `Faycalraghibi/Tsunagu`. Conventional commits per task on `feat/mvp`, PR with CI, merge only after the owner writes "I greenlit", squash into one MVP commit on `main`. No AI attribution trailers (owner convention). |

## Facts discovered while planning (differ from the spec)

1. **Docs moved and changed format.** LangGraph docs now live in `langchain-ai/docs` under `src/oss/langgraph/` as Mintlify `.mdx`, not plain Markdown. Pinned commit: `23152b1133e7759381661e93bcdca778101a80bc` (2026-10-21). Consequences:
   - A cleaning step is needed: front matter, `{/* */}` comments, JSX tags (`<Note>`, `<Tabs>`, `<Accordion>`), and `:::python` / `:::js` language blocks. We keep Python only, so the corpus is not duplicated in TypeScript.
   - Internal links are absolute (`/oss/langgraph/persistence`, sometimes `/oss/python/langgraph/...`). Normalised to page ids such as `langgraph/persistence`.
   - API references use `@[`StateGraph`]` or `@[`StateGraph`][StateGraph]`. They are a free, precise symbol signal and become code spans.
2. **Corpus selection: 47 pages**, about 1.1 MB of raw MDX, measured at 178,726 tokens (cl100k) and 732 chunks after dropping the JS blocks. Selected: all of `src/oss/langgraph/` except the two changelog stubs and `case-studies`, plus `src/oss/concepts/{memory,human-in-the-loop,context}.mdx`, `src/oss/langchain/human-in-the-loop.mdx` and `src/oss/langchain/multi-agent/{index,handoffs,custom-workflow}.mdx` (multi-agent moved to the LangChain section).
3. **Python 3.14 is the only interpreter installed.** The plan creates a Python 3.12 venv (`tsunagu-venv`) for torch, chromadb and llama-index compatibility, and falls back to 3.14 only if 3.12 cannot be installed.
4. **LlamaIndex extraction failed silently on the real model; the JSON extractor was used.** In the Task 16 spike (`results/extraction_spike.json`, 10 chunks, `nvidia/nemotron-3.5-lightning-30b-a3b`), `SchemaLLMPathExtractor` returned 0 triplets on all 10 chunks and raised no error, while our schema-guided JSON extractor returned triplets on 9 of 10 (`[10, 10, 10, 10, 9, 9, 1, 10, 10, 0]`). The LlamaIndex unit test with a fake LLM still passes, so the failure is specific to the real model's output against LlamaIndex's generated schema; the root cause was not investigated. Per the agreed fallback rule, `graph.extraction_backend` is `json` for the whole graph. Lesson: a framework that swallows extraction errors needs a measured check before you trust it.
5. **Questions and merge labels were drafted by the AI assistant, not the owner.** On 2026-10-22 the owner asked the assistant to write `data/questions.jsonl` (30 questions, gold answers, gold pages) and to fill the `correct` column of `results/merges_review.csv` (17 y, 13 n). Both were done before any retrieval or answer result existed. Consequences: (a) the spec's line "le jeu de questions ... sont de moi" is no longer true as written; (b) the same assistant also designed the graph retriever, so the question set may favour structures the graph handles well, which is a stated limit; (c) the false-merge rate reflects the assistant's judgement, not a human one. `data/labels_manual.csv` (judge calibration, G3) stays owner-only: a model labelling answers to calibrate a model judge would defeat its purpose.

## Design choices the spec left open

- **Graph node types.** Spec types (`Concept`, `Class`, `Function`, `Parameter`, `Feature`, `Page`) plus `Symbol` for deterministic code-span mentions. The resolver links a `Symbol` to an LLM entity with the same normalised name (`symbol_link` merge). This is the bridge between the free skeleton and the LLM layer.
- **MENTIONS.** Spec says `Chunk -MENTIONS-> Symbol`. We store it as `Page -MENTIONS-> entity` edges plus `chunk_ids` provenance on every node. Traversal collects the chunks recorded on reached nodes. The meaning is the same and the graph stays small.
- **Hub control.** Expansion from any node follows at most `max_expand_degree` (40) neighbours, chosen by name similarity to the query (query-guided expansion). The report gives the mean candidate pool size, so we can see when "graph" falls back to "dense over everything".
- **Statistics.** A question passes for a system when the judge gives 2 in a majority of the 3 repeats. Wilson intervals use n = 30 questions (not 90 runs, which would overstate independence). Repeat stability is reported separately.
- **Merge review.** 30 merges are drawn at random (seed 42) from the embedding merges, giving an unbiased false-merge estimate rather than a best-case or worst-case one.
- **Report safety.** `tsunagu report` rewrites only the block between `<!-- AUTO:START -->` and `<!-- AUTO:END -->` in `results/results.md`. Findings, Limits and the Safran section are written by the owner and never overwritten.
