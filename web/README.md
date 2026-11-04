# ProfitLens — in the browser

The ProfitLens chatbot (Amex Campus Challenge 2026, Premier Card profitability) as a **static web app**:
the language model, the embeddings and all eight tools run on the visitor's device. No server, no API key,
free to host (see [HOSTING.md](HOSTING.md)). The Streamlit + Ollama app in `../profitlens` is untouched.

| | Streamlit app (`../profitlens`) | this app (`web/`) |
|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 via Ollama | Qwen3 4B / 1.7B / 0.6B via **WebLLM** (WebGPU, Web Worker) — or your own Ollama |
| Embeddings | nomic-embed-text via Ollama + FAISS | bge-small-en-v1.5 via **transformers.js** (wasm, Web Worker) + in-memory cosine store |
| Data | 500K-row DataFrame in server memory | static shards; terms/profit recomputed in the browser, exact vs Python |
| Tools | 8 pandas tools | the same 8 tools in TypeScript (`src/tools`) |
| Uploads | server-side, FAISS on disk | in-browser parse + embed, IndexedDB |

## Run it

```bash
cd web
npm install
npm run dev        # http://localhost:5189
npm run build      # type-check + production build → dist/
npm run preview    # serve dist/ on http://localhost:5190
npm test           # 88+ tests (golden-checked against the Python tools)
```

Open it in a recent **Chrome/Edge**, pick a model in the sidebar (**Download & load** — a one-time download that
the browser caches) and ask something. No WebGPU? Use **My Ollama** (it connects to `http://localhost:11434`).

## Layout

```
web/
├─ scripts/export_data.py   Python → public/data (reuses ../profitlens modules; verifies the xlsx)
├─ scripts/embed_core.mjs   pre-embeds the knowledge base with the browser's own ONNX model
├─ public/data/             meta.json, context.md, tools_static.json, core_chunks.json, core_vectors.bin, customers/NNN.bin
├─ src/data/                shard decoding, frozen-formula scoring, Python-compatible rounding
├─ src/tools/               the 8 tools + a safe pandas-`query` evaluator
├─ src/rag/                 chunkers (ports of rag/chunker.py), vector store, embedding worker, uploads, IndexedDB
├─ src/llm/                 system prompt, tool-call loop, WebLLM + Ollama engines
├─ src/ui/                  React UI: chat, SVG charts, digest tab, model panel
└─ tests/                   vitest (unit + golden parity) and tests/e2e (Playwright smoke tests)
```

## Tests

* `npm test` — parity with the Python code: 170+ customer profiles, 45 tool calls (queries, histograms, comparisons),
  the markdown/python chunkers, number formatting, the prompt builders, the tool-calling loop and context fitting.
* `RAG_E2E=1 npx vitest run tests/rag.e2e.test.ts` — retrieval quality with the real embedding model (needs the model download).
* `node tests/e2e/no-gpu.mjs` / `node tests/e2e/ollama.mjs` — drive the built site in Chromium
  (`CHROME_PATH=… BASE_URL=http://localhost:5190`); the second one needs a local Ollama.
