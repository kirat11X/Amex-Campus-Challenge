# ProfitLens — Premier Card Profitability Chatbot

An AI-powered chatbot that analyzes credit-card customer profitability for the Amex Campus Challenge 2026 dataset. Built with **Streamlit** + a local **Ollama** model + **pandas**.

All numbers come from deterministic pandas computation — the LLM only calls tools and explains results, never guesses values.

## Setup

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M   # chat + tool calling (~2.5 GB)
ollama pull nomic-embed-text                 # RAG embeddings

python -m venv .venv
source .venv/bin/activate
pip install -r profitlens/requirements.txt

# Data files must be in the project root:
#    - 6a3eb196bc7a3_campus_challenge_r1_data.csv
#    - 6a3eb1af6b994_feature_description.csv
#    - submission_FINAL.xlsx

cd profitlens && streamlit run app.py
```

`OLLAMA_HOST`, `LLM_MODEL` and `EMBED_MODEL` can be overridden via environment or `.env`.
The first launch embeds the knowledge base (~125 chunks, under a minute); it is cached afterwards.

## Tools & retrieval toggle

| | **ON** (default) | **OFF** |
|---|---|---|
| Model calls per message | up to 6 (tool-calling loop) | 1 |
| Tools / RAG retrieval | pandas tools + top knowledge-base chunks injected on every message | Skipped |
| Grounding | Live tool results, cited RAG chunks | System-instruction context only |
| Auto-rendered charts | Waterfall / histogram / term-comparison | None |

A 4B model is good at tool-grounded numbers (customer breakdowns, counts) but can misquote
line numbers or fine print from retrieved code — verify code citations against the source.

## Sample Questions

- "Profit breakdown for customer 1"
- "Why is customer 20 in the top 20%?"
- "Compare lounge usage between top 20% and rest"
- "Show the profit distribution"
- "Which cost term hurts profitability the most?"
- "Explain my profitability formula term by term"
- "What's the profit breakdown for customer 8?"
- "How many customers have collections calls (f3 >= 1)?"
- "Which line of the python file handles missing values?" *(retrieval)*
- "What does the framework say about validation?" *(retrieval)*
- "What columns does my uploaded file have?" *(retrieval over your own files)*

## Architecture

| File | Purpose |
|---|---|
| `config.py` | Weights (W dict), file paths, feature dictionary, model name |
| `data_engine.py` | Load CSV, score, cache to parquet, verify against submission |
| `tools.py` | 7 pandas tool functions for LLM function calling |
| `llm.py` | Ollama client, system prompt, multi-turn function-calling loop |
| `app.py` | Streamlit chat UI with plotly charts |

## Profitability Formula

```
profit = 0.024 × S_all + 0.331 × f1
       − (55 × f13 + 29 × f15 + 1.9 × f14 + 1.5 × f16)
       − 1.35 × f11 × f1
       − 70 × f2  − 8500 × f3
       + (−4.5) × f12

where S_all = f6 + f7 + f8 + f9 + f10
```

Top-20% flag: customers whose profit ranks in the 80th percentile or above.

## Grounding Context

`PROFITLENS_CONTEXT.md` is the **master grounding file** for the chatbot. It contains:

- Bot identity and 13 response rules
- Question taxonomy with tool-call recipes
- Challenge/product/dataset documentation
- The frozen profitability formula and coefficient rationale
- **Section 8**: precomputed ground-truth dataset statistics
- Glossary, gold-standard answer examples, and a hard "never" list

### How it works

The file is loaded **verbatim** (no summarization or truncation) as the `system_instruction` in every model call. It is cached in memory after the first read and never re-injected into chat history.

**Trust hierarchy:**
```
live tool result > Section 8 stats > RAG chunk (cited) > refuse to guess (never invent)
```

A **Runtime Verification** block is auto-generated at startup by recomputing 5 key stats from the live dataset and comparing them against the expected values hardcoded in `config.py`. A mismatch triggers a yellow "⚠ Context stats stale" badge in the sidebar but never crashes the app.

### Maintenance rule

> **If the formula, weights, or dataset ever change — regenerate Section 8 stats. Never hand-edit them.**

To regenerate:

```bash
cd /path/to/amex
source .venv/bin/activate
python profitlens/scripts/refresh_context_stats.py
```

The script prints the updated Section 8 block and the new `EXPECTED` dict for `config.py` in copy-paste format. Manually paste the output into `PROFITLENS_CONTEXT.md` (Section 8) and `config.py` (EXPECTED dict), then restart the app.

## Knowledge Base (RAG)

ProfitLens has a local retrieval layer built on local Ollama embeddings + FAISS. It is
exposed to the model as **one more tool** in the existing function-calling loop
(`search_knowledge_base`) — there is no separate RAG chain, chat, or model client.

### Two corpora

| Corpus | Contents | Built from |
|---|---|---|
| **core** | ~125 chunks: `trial.py` (code-aware, one chunk per `FRAMEWORK` entry), the feature dictionary, generated dataset & formula-output **profile docs**, the Profitability Framework sheet, `PROFITLENS_CONTEXT.md` sections, and the challenge deck (one chunk per page) | the project's own files |
| **user** | whatever you upload in the *Digest your own files?* tab | your uploads |

> **The 500K-row CSV and Predictions sheet are never chunked into vectors.**
> Row-level numbers belong to the pandas tools. The data enters retrieval only as
> compact generated *profile documents* — per-feature statistic cards, per-term
> P&L cards, a profit-distribution card, segment economics, and an integrity card.
> Uploaded CSV/XLSX files get the same treatment (schema + per-column stats +
> a 20-row preview; stats only above 50K rows).

### Trust hierarchy

```
live pandas tool result  >  Section 8 stats  >  RAG chunk (must cite source)  >  refuse
```

For **user-uploaded** files, retrieved chunks *are* the primary source, cited by
filename and location (`report.pdf, p.3`). Retrieval has a cosine similarity floor
of 0.5 — below it the tool returns "nothing relevant found" rather than letting the
model improvise.

### Using the Digest tab

1. Open **📂 Digest your own files?** and drop in `.pdf .docx .txt .md .csv .xlsx .py .json` (max 25 MB each).
2. Each file is chunked, embedded and added to the user index — you get a toast with the chunk count.
3. The table lists every digested file with a 🗑 delete button, plus **Clear all my files**.
4. Ask about them in the **💬 Chat** tab. Answers cite the filename and location.
5. The sidebar toggle **"Include my uploaded files in answers"** (visible on both tabs) forces retrieval to the core corpus when switched off.

Re-uploading identical content is a no-op (content-hash duplicate guard). The user
index persists in `rag_store/` across restarts.

### Rebuilding the core index

The core index rebuilds automatically at startup when any source file's sha256
changes (tracked in `rag_store/hashes.json`), when `W` changes, or when
`CORPUS_VERSION` in `rag/ingest.py` is bumped after a chunking-logic change.
Otherwise it loads from disk with no embedding calls.

To force a full rebuild:

```bash
rm -rf rag_store/          # drops the embedding cache too
# or, keeping the cache so nothing is re-embedded:
rm rag_store/hashes.json
```

Embeddings use `nomic-embed-text` (768-d, L2-normalized) served by local Ollama.
Override with `EMBED_MODEL` / `EMBED_DIM` in `.env` — and delete `rag_store/` after
changing the embedding model, since vectors from different models aren't comparable.
Every vector is cached by content hash in `rag_store/emb_cache.sqlite`,
so rebuilds and re-ingestion never re-embed unchanged text.

### Architecture (updated)

| File | Purpose |
|---|---|
| `config.py` | Weights (W dict), file paths, feature dictionary, model name, **EXPECTED Section-8 values**, **embedding + RAG settings** |
| `data_engine.py` | Load CSV, score, cache to parquet, verify against submission, **runtime verification** |
| `tools.py` | 8 tool functions for LLM function calling (7 pandas + **`search_knowledge_base`**) |
| `llm.py` | Context loader (verbatim), Ollama client, function-calling loop, **RAG guidance block** |
| `app.py` | Streamlit UI: **💬 Chat** and **📂 Digest your own files?** tabs, plotly charts, sidebar badges |
| `rag/chunker.py` | Hand-rolled chunkers: prose, markdown-by-header, statement-aware Python |
| `rag/profile_docs.py` | Dataset / formula-output / uploaded-table profile documents |
| `rag/embed.py` | Ollama embeddings — batching, retry, sqlite content-hash cache |
| `rag/store.py` | FAISS `IndexFlatIP` wrapper: add / search / delete-by-file / save / load |
| `rag/ingest.py` | Core-corpus build with hash caching; user upload, delete, clear |
| `PROFITLENS_CONTEXT.md` | Master grounding file — system instruction source of truth |
| `scripts/refresh_context_stats.py` | Read-only helper to recompute Section-8 stats for manual paste-in |
| `rag_store/` | Gitignored: FAISS indexes, JSONL metadata, hashes, embedding cache |
