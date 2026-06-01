# Tsunagu: GraphRAG vs vector RAG on cross-referenced technical documentation

*Spec for a one-day personal prototype, to be built on 22 October 2026, before the Safran Aircraft Engines interview.*
*Name suggestion: "Tsunagu" (つなぐ, "to connect"), in line with Joshu, Denryoku and Rikurita. Rename it freely.*

---

## 0. Honesty contract (read first)

- **You may talk about this project only once it is built and has run.** Every number you say comes from `results/results.md`, and nothing else.
- **How to present it:** "un prototype personnel d'une journée, monté pour comparer GraphRAG et RAG vectoriel sur une documentation technique très référencée. J'ai utilisé beaucoup d'assistance IA pour le code ; la conception, le jeu de questions, l'évaluation et les conclusions sont de moi."
- **If an interviewer asks when you built it:** say it plainly ("cette semaine, pour préparer ce sujet"). That reads as initiative, not weakness.
- **Mid-build cuts:** if you stop before a section is done, say exactly what was done. The spec marks **MVP** (must be done) and **Stretch** (only if time remains).

---

## 1. Goal and research question

**Question.** On a corpus of technical documentation full of cross-references, does a graph layer (entities, relations, explicit links between pages) improve answers to **relational, multi-hop and enumerative** questions compared with:

1. a dense vector RAG, and
2. a cheap hybrid RAG (BM25 + dense with reciprocal rank fusion)?

**And at what cost?** Measure indexing tokens, indexing time, query latency, and how much effort entity resolution takes.

**Why the hybrid control matters.** Many "GraphRAG wins" in blog posts are really "anything beats naive dense retrieval" wins. Comparing against hybrid tells you whether the *graph* adds value, or just *better retrieval* does. This single design choice is your strongest talking point.

**Why this corpus mirrors Safran's (use this bridge in the interview):**

| LangGraph docs | Safran technical documentation (inferred) |
|---|---|
| Links between pages | References between documents |
| API symbols (`StateGraph`, `MemorySaver`) | Exact references: part numbers, document IDs |
| Deprecated APIs and version notes | Revisions, applicability by version |
| "Feature X requires Y" | "Procedure X applies under condition Y" |

---

## 2. Corpus (MVP)

**Source:** the LangGraph documentation on GitHub, MIT licence.

**Where to find it.** It historically lived in `langchain-ai/langgraph` under `docs/docs/`. In 2025 LangChain moved its documentation to `langchain-ai/docs`. Check which repository currently holds the LangGraph pages and **pin a commit hash**; record it in `README.md` and `config.yaml`.

**Which pages:** 40 to 60 Markdown pages:

- core concepts: graph API, state, reducers, nodes and edges, persistence and checkpointers, memory, interrupts and human-in-the-loop, subgraphs, streaming, multi-agent, time travel, durable execution
- the how-to pages linked from those concepts

Skip changelogs and auto-generated API reference pages, which are too noisy.

**Why you can judge the answers:** you have used LangGraph for real (subgraphs, multi-agent, human in the loop, checkpointers, conditional routing in Denryoku and Rikurita), so you can tell good answers from bad ones.

**Size target:** about 150k to 400k tokens of text. Record the real count in `results.md`.

---

## 3. Architecture

```
docs (pinned commit)
   |
   v
[ingest]  parse Markdown: heading tree, code spans, links
   |--> chunks (heading-aware, ~400-600 tokens, small overlap)
   |--> explicit edges: Page -LINKS_TO-> Page        (deterministic, free)
   |--> symbol mentions: Chunk -MENTIONS-> Symbol    (regex on `code spans`)
   |--> LLM extraction: (entity, relation, entity)   (schema-guided)
   v
[resolve] normalise and merge duplicate entities
   v
Indexes:  A. dense vectors (ChromaDB)
          B. BM25 + dense, fused with RRF
          C. property graph + vectors on nodes and chunks
   v
[answer]  same LLM, same prompt, same top-k budget for A, B, C
          answer must cite page paths, or abstain
   v
[eval]    retrieval metrics, LLM judge (calibrated on your labels),
          latency, tokens, cost, reported with confidence intervals
```

**Key design choice: a two-source graph.** Explicit links and code symbols give a reliable skeleton at **zero LLM cost**. LLM extraction adds the semantic relations. This mirrors what you would do at Safran: use document references and metadata deterministically, and spend LLM calls only where structure is missing.

---

## 4. Components and specs

### 4.1 Ingestion (MVP)

- **Parsing:** use `markdown-it-py`, or a simple heading parser. Keep the heading path for each chunk ("Persistence > Checkpointers > Memory saver").
- **Chunking:** split on headings first, then by size (~500 tokens, 50-token overlap). **Never split a code block.**
- **Chunk metadata:**
  - `page_path`
  - `heading_path`
  - `chunk_id`
  - `symbols` (from code spans)
  - `out_links` (normalised target pages)
- **Link normalisation:** resolve relative links and anchors; drop external URLs (keep their count).

### 4.2 Graph extraction (MVP)

- **Framework:** LlamaIndex `PropertyGraphIndex`. Check the current API names in the docs before coding.
- **Extractors:**
  - `SchemaLLMPathExtractor` with a fixed schema:
    - entity types: `Concept`, `Class`, `Function`, `Parameter`, `Feature`, `Page`
    - relations: `USES`, `REQUIRES`, `CONFIGURES`, `PART_OF`, `ALTERNATIVE_TO`, `DEPRECATED_BY`, `RETURNS`, `LINKS_TO`, `MENTIONS`
    - `strict=True`, so only schema relations are kept
  - **plus** your own deterministic extractor that writes `LINKS_TO` and `MENTIONS` edges from the ingestion metadata
- **Store:**
  - MVP: in-memory `SimplePropertyGraphStore`, persisted to disk.
  - Stretch: Neo4j Community in Docker, which gives you real Cypher experience.
- **LLM for extraction:** your free OpenAI-compatible provider (NVIDIA Nemotron Lightning, or an OpenRouter free model), temperature 0. Use the same provider as for answering, unless a stronger cheap model is available. Record which one you used.
- **Cache every LLM call** to disk (`cache/` as JSON, keyed by a hash of model + prompt). This makes the runs reproducible and gives you the "tokens spent" number.

### 4.3 Entity resolution (MVP: a simple version)

1. **Normalise:** lowercase, strip backticks and `()`, singularise, collapse whitespace.
2. **Alias map:** a hand-written `aliases.yaml` (for example `checkpointer` / `checkpoint saver` / `saver`).
3. **Embedding merge:** merge entities of the same type whose name embeddings have cosine > 0.92. **Log every merge** to `results/merges.csv`, then review the top 30 merges by hand.
4. **Record** the number of entities before and after resolution, plus the false merges you found. This is an interview gold mine: entity resolution is the classic GraphRAG pain point.

### 4.4 Retrievers (MVP: A, B, C; Stretch: C2, D)

All retrievers return at most **8 chunks** to the answering step, so the comparison is fair.

- **A. Dense:**
  - embeddings: `BAAI/bge-small-en-v1.5`, local, through `sentence-transformers`
  - store: ChromaDB
  - top-k = 8
- **B. Hybrid:**
  - `rank_bm25` top-20 and dense top-20, fused with RRF: score = sum of 1 / (60 + rank)
  - keep the top 8
- **C. Graph:**
  1. find seed entities: entity-name match on symbols, plus vector search over node names
  2. expand 1 or 2 hops along the schema relations and `LINKS_TO`
  3. collect the chunks that `MENTIONS` the reached entities, plus the chunks of linked pages
  4. rank them by dense similarity to the query, then keep the top 8
  - use LlamaIndex's `VectorContextRetriever` and `LLMSynonymRetriever` where they fit; otherwise write the traversal yourself (about 40 lines, and easier to explain)
- **C2 (Stretch). Graph + hybrid:** the union of B and C, reranked, top 8.
- **D (Stretch). LightRAG:** HKUDS LightRAG on the same corpus, in `local`, `global` and `hybrid` modes, as an external reference point.
- **Not included:** Microsoft GraphRAG community summaries (Leiden). Too costly for one day. Explain it conceptually and say why you skipped it (indexing cost, and global questions are rare in documentary search).

### 4.5 Answering (MVP)

- **Same LLM, same prompt, temperature 0 for A, B and C.**
- **Prompt rules:**
  - answer only from the given context
  - cite `page_path#heading` for each claim
  - if the context does not contain the answer, reply exactly `INSUFFICIENT_CONTEXT`
- **Output format:** JSON `{answer, citations[], abstained}`, validated with Pydantic. On a parse failure, retry once.

### 4.6 Question set (MVP: 30 questions, written by you, from the docs)

Write each one yourself, reading the pages. For each question, record:

- `id`
- `type`
- `question`
- `gold_answer` (1 to 3 sentences)
- `gold_pages` (the 1 to 4 pages that support it)
- `notes`

| Type | Count | Example (adapt to the real docs) |
|---|---|---|
| Single fact | 7 | "What does a reducer do on a state key?" |
| Multi-hop | 8 | "To resume a graph after a human-in-the-loop interrupt, what must be configured when compiling it, and where is the state stored?" |
| Relational / enumerative | 8 | "Which features in the docs require a checkpointer?" |
| Exact reference | 4 | "Which class is recommended for in-memory checkpointing in tests?" |
| Unanswerable | 3 | "How do I deploy LangGraph on AWS Bedrock Agents?" (if the corpus does not cover it) |

Expectation: graph retrieval should shine on the multi-hop and relational types, hybrid on exact references, and dense on single facts. **These are hypotheses, not results.**

### 4.7 Evaluation (MVP)

**Retrieval** (using `gold_pages`):

- **hit@8:** is at least one gold page among the retrieved chunks?
- **recall@8 of gold pages:** the share of gold pages covered.
- **MRR:** the reciprocal rank of the first gold page.

**Generation**, scored by an LLM judge with a written rubric:

- **correctness** against `gold_answer`, on a 0 / 1 / 2 scale
- **faithfulness:** split the answer into claims; for each claim, is it supported by the retrieved context? Report the share supported.
- **citation precision:** the share of cited pages that are gold or that support the claim
- **abstention:** correct `INSUFFICIENT_CONTEXT` on unanswerable questions, and false abstentions on answerable ones

**Judge calibration:**

- Hand-label the correctness of 20 answers yourself (blind to the system).
- Compute the raw agreement and Cohen's kappa between you and the judge, and report both.
- If kappa is below about 0.6, fix the rubric before trusting the judge.

**Operations:**

- indexing time and indexing tokens (from the cache)
- query latency p50 and p95
- tokens per query

**Statistics:**

- Run generation **3 times** per system (temperature 0 still varies with some providers).
- Report pass rates with **95% Wilson intervals**.
- With 30 questions, differences under about 15 points are probably noise. **Say so.**

### 4.8 Ablations (Stretch, pick one or two)

- 1 hop vs 2 hops
- graph without the explicit `LINKS_TO` edges (is the LLM extraction alone enough?)
- without entity resolution
- top-k 4 vs 8

---

## 5. Repository layout

```
tsunagu/
  README.md              goal, corpus commit, how to run, honest scope
  config.yaml            provider, models, k, hops, thresholds, commit
  aliases.yaml
  src/tsunagu/
    ingest.py            parse, chunk, links, symbols
    graph.py             extraction, deterministic edges, store
    resolve.py           normalisation, alias map, embedding merge
    retrievers.py        dense, hybrid (RRF), graph (+ stretch)
    answer.py            prompt, JSON schema, abstention
    evaluate.py          retrieval metrics, judge, calibration, Wilson CI
    llm.py               OpenAI-compatible client + disk cache
    cli.py               typer: ingest | build-graph | resolve | ask | eval | report
  data/questions.jsonl
  data/labels_manual.csv
  results/results.md     the only source of numbers for the interview
  results/merges.csv
  results/per_question.csv
  tests/                 pytest with a fake LLM client: chunker keeps code
                         blocks, RRF, link normalisation, metrics, JSON parsing
```

**Commands:**

```
tsunagu ingest
tsunagu build-graph
tsunagu resolve
tsunagu eval --systems dense,hybrid,graph --repeats 3
tsunagu report
tsunagu ask "..." --system graph --show-path
```

`--show-path` prints the entities and edges the graph retriever traversed. That output is great to show on screen if the interview allows it.

**Dependencies:**

```
llama-index-core
llama-index-llms-openai-like
sentence-transformers
chromadb
rank-bm25
markdown-it-py
pydantic
typer
pyyaml
pytest
```

Stretch only: `neo4j`, `llama-index-graph-stores-neo4j`, `lightrag-hku`.

---

## 6. Timeline (about 6 hours)

| Block | Time | Done when |
|---|---|---|
| Setup, pin corpus, ingest + chunk + links + tests | 1 h | chunks and link edges written; chunker tests pass |
| Write the 30 questions with gold pages | 1 h | `questions.jsonl` complete (do it **before** seeing any results) |
| Dense + hybrid retrievers, answering, retrieval metrics | 1 h | A and B evaluated on retrieval |
| Graph extraction + deterministic edges + resolution | 1.5 h | graph built, `merges.csv` reviewed |
| Graph retriever + full eval with judge + 20 manual labels | 1 h | `results.md` with all metrics and CIs |
| One-page write-up | 0.5 h | `results.md` "Findings" and "What I would do at Safran" |

**If you run out of time:**

- Cut the ablations first, then the repeats (go to 1 run and say so).
- **Never** cut the hybrid control or the manual judge calibration.

---

## 7. `results.md` template (fill in with real values only)

```
Corpus: LangGraph docs @ <commit>, <N> pages, <T> tokens, <C> chunks
Graph: <E> entities before / <E'> after resolution, <R> relations,
       <L> explicit links; false merges found: <F>/30 reviewed
Indexing: <tokens> LLM tokens, <minutes> min (extraction model: <model>)

Retrieval (hit@8 / recall@8 / MRR), by question type: table for A, B, C
Correctness (judge), with 95% Wilson CI: table for A, B, C
Faithfulness, citation precision, abstention: table
Latency p50 / p95, tokens per query: table
Judge calibration: agreement <x>%, kappa <k> on 20 labels

Findings (3 to 5 bullets, honest, including where the graph lost)
Limits (corpus size, 30 questions, one LLM, judge bias)
What I would do at Safran (5 lines)
```

---

## 8. Interview narrative (fill the brackets from `results.md`)

**30 seconds, FR:**

> « Pour préparer ce sujet, j'ai monté cette semaine un petit prototype qui compare trois approches sur la documentation de LangGraph, un corpus technique très référencé, un peu comme une documentation d'ingénierie : RAG dense, RAG hybride BM25 plus dense, et GraphRAG. Le graphe combine des liens explicites entre pages, gratuits, et des relations extraites par LLM. Sur 30 questions que j'ai écrites, avec des pages de référence, [résultat principal]. Le hybride faisait déjà [x] ; le graphe n'apportait vraiment quelque chose que sur [types]. Le point dur, c'est la résolution d'entités : [y] fusions erronées sur 30 vérifiées. Et l'indexation a coûté [z] tokens. »

**30 seconds, EN:** the same content in English. Write it yourself after the run.

**The lesson to land (if the results allow it):** structure that already exists (references, metadata) is worth exploiting deterministically first; LLM-extracted graphs come after, measured against a strong hybrid baseline.

---

## 9. Anticipated questions (answer plans depend on your real results)

**Design**

1. Why GraphRAG at all, and why this corpus? → cross-references and relational questions; the mapping to Safran from §1.
2. Why did you compare against hybrid and not only dense? → to isolate the graph's real contribution.
3. How did you build the graph? → two sources: deterministic links and symbols, plus schema-guided LLM extraction; why `strict` schema.
4. Why a fixed schema rather than free extraction? → precision, fewer junk relations, an easier traversal; the trade-off is lower recall.
5. How do you pick seed entities for a query? → symbol match plus vector search on node names; failure case: a vague question with no entity.
6. 1 hop or 2? → [ablation result, or "not measured, my hypothesis is..."].

**Entity resolution**

7. How did you handle duplicates? → normalisation, alias map, embedding merge at 0.92, manual review: [numbers].
8. What happens with a false merge? → wrong expansions pull in wrong chunks; example: [real one from `merges.csv`].

**Evaluation**

9. How did you build the question set, and how do you avoid bias? → written before seeing results, stratified by type, gold pages.
10. Why trust an LLM judge? → calibrated on 20 manual labels, kappa [k]; and its limits.
11. Is your difference significant? → Wilson intervals, 30 questions, [honest answer].
12. How do you measure faithfulness? → claim splitting plus support check; its limits.
13. What about unanswerable questions? → `INSUFFICIENT_CONTEXT`, abstention rate [x].

**Cost and operations**

14. What did indexing cost, and would it scale to Safran's corpus? → [tokens]; extrapolate linearly; incremental updates are the real problem.
15. How do you update the graph when a document changes? → not implemented; the plan: hash per page, re-extract only changed pages, remove stale edges.
16. Latency? → [p50 / p95]; graph traversal is cheap, the LLM call dominates.

**Transfer to Safran and AWS**

17. How would you do this on AWS? → Bedrock for the LLM and embeddings; Neptune Analytics, or the GraphRAG option of Bedrock Knowledge Bases if it fits (verify in the docs); OpenSearch for hybrid search; your actual experience is the local prototype.
18. Would you use GraphRAG at Safran? → only where relations matter (references, applicability, revisions), after a strong hybrid baseline, and with deterministic edges first.
19. What about document rights? → filter at chunk level **before** traversal, so the graph cannot leak a restricted document through an edge.

**Honesty**

20. Did AI write the code? → "Beaucoup, oui ; le design, les questions, l'évaluation, la relecture et les conclusions sont de moi. Je peux vous expliquer chaque brique."
21. Have you used Neo4j or Cypher? → only if you did the Neo4j stretch; otherwise "pas encore, j'ai utilisé un store en mémoire".
