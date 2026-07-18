# Amex Campus Challenge 2026 — Round 1: ProfitLens

Premier Card customer-profitability scoring for the Amex Campus Challenge 2026 (Round 1),
plus **ProfitLens** — a Streamlit + Gemini chatbot for exploring the scored dataset.

All profitability numbers are computed deterministically with pandas from a single,
frozen weights dict (`W`). The chatbot never invents numbers — it calls tools that read
straight from the scored data, or explains/cites what those tools return.

## Repository layout

```
amex/
├── 6a3eb196bc7a3_campus_challenge_r1_data.csv    # Raw challenge dataset (500K rows, f1-f23)
├── 6a3eb1af6b994_feature_description.csv         # Feature dictionary
├── 6a3ea743b1106_campus_challenge26_r1.pdf        # Challenge brief / deck
├── submission_FINAL.xlsx                         # Final submission workbook (built by main/trial.py)
├── .env                                          # GEMINI_API_KEY / MODEL — gitignored
│
├── main/                                         # Scoring pipeline + iteration history
│   ├── trial.py                                  # FINAL submission builder — source of truth for weights W
│   ├── endgame_pipeline.py, pipeline_v2.py, …     # Earlier pipeline iterations (kept for reference)
│   ├── try.ipynb                                 # Exploration notebook
│   └── submission_*.xlsx                         # Earlier submission variants (gitignored, see below)
│
├── preds/                                        # Prediction dumps from earlier iterations (gitignored)
│
├── profitlens/                                   # Streamlit chatbot app — see profitlens/README.md
│   ├── app.py                                    # Chat UI (Chat + Digest-your-files tabs)
│   ├── config.py                                 # Weights, paths, feature dictionary, model name
│   ├── data_engine.py                            # Load/score CSV, cache to parquet, verify submission
│   ├── tools.py                                  # Gemini function-calling tools (pandas + RAG search)
│   ├── llm.py                                    # Gemini client, system prompt, function-calling loop
│   ├── rag/                                      # Retrieval layer (chunking, embeddings, FAISS store)
│   ├── PROFITLENS_CONTEXT.md                     # Master grounding file (system instruction)
│   └── scripts/refresh_context_stats.py          # Regenerates the grounding file's stats section
│
└── rag_store/                                    # Gitignored: FAISS index + embedding cache for the RAG layer
```

## The profitability formula

Frozen — defined once in `main/trial.py`'s `W` dict and mirrored in `profitlens/config.py`:

```
profit = 0.024 × S_all + 0.331 × f1
       − (55 × f13 + 29 × f15 + 1.9 × f14 + 1.5 × f16)
       − 1.35 × f11 × f1
       − 70 × f2 − 8500 × f3
       + (−4.5) × f12

where S_all = f6 + f7 + f8 + f9 + f10
```

Top-20% flag: customers whose profit ranks in the 80th percentile or above.

## Regenerating the submission

```bash
cd main
python trial.py
```

Reads the raw CSV + submission template and writes `submission_FINAL.xlsx` to the repo root.
Change a weight in `W` and both the predictions and the stated equation string update together
(the equation is generated from `W`, never hardcoded).

## ProfitLens chatbot

An AI chat interface over the scored dataset: ask about any customer's profitability breakdown,
compare top-20% vs. the rest, explore distributions, dig into the challenge docs, or upload your
own files for it to answer questions about. Built with **Streamlit** + **Google Gemini** + **pandas**,
with a local RAG layer (FAISS + Gemini embeddings) for grounding on the project's own docs.

Full setup, architecture, and RAG details: **[profitlens/README.md](profitlens/README.md)**.

Quick start:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r profitlens/requirements.txt

# .env in this repo root:
#   GEMINI_API_KEY=your-api-key-here
#   MODEL=gemini-2.0-flash

cd profitlens
streamlit run app.py
```

### Gemini free tier

The Gemini free tier caps requests at ~20/day per model. ProfitLens has a
**"🆓 Free tier mode"** toggle in the sidebar (on by default) that skips the multi-round
tool-calling loop and knowledge-base retrieval, making exactly one direct model call per
message so you stay under quota. Turn it off for fully grounded, tool-backed answers
(customer lookups, live stats, RAG citations, auto-rendered charts) if you're on a paid plan.

## Environment

Create `.env` in the repo root (gitignored):

```
GEMINI_API_KEY=your-api-key-here
MODEL=gemini-2.0-flash
```
# Amex-Campus-Challenge
