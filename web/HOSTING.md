# Hosting ProfitLens for free (Vercel, static)

`amex/web` is a **static site**: HTML, JS, a few MB of JSON and ~23 MB of customer shards. There is no
backend and no API key. The language model and the embedding model run **in the visitor's browser**
(WebGPU / WebAssembly), so hosting costs nothing and every visitor brings their own compute.

```
visitor's browser ── loads ──▶ Vercel (static files only)
   │                               index.html, JS, /data/*.json, /data/customers/*.bin, /ort/*.wasm
   ├─ WebLLM (Qwen3, WebGPU, in a Web Worker) ◀── model weights, once, from Hugging Face → Cache API
   ├─ transformers.js (bge-small, wasm, in a Web Worker) ◀── 34 MB model, once, from Hugging Face
   ├─ the 8 tools (TypeScript): one customer shard per lookup, or the full table for ad-hoc queries
   └─ uploaded files: parsed + embedded locally, kept in IndexedDB (never sent anywhere)
```

## 1. Deploy to Vercel (Hobby plan)

The git root is `amex/`, so the project lives in the `web` sub-folder.

1. Commit and push the repo (including `web/package-lock.json` and `web/public/data/` — the data is part of the app).
2. <https://vercel.com/new> → **Import** the GitHub repo.
3. **Root Directory:** `web`  (Edit → pick the `web` folder).
4. Framework preset: **Vite** (auto-detected; `vercel.json` already sets install/build/output).
   Leave environment variables empty. Node 20+ (the default).
5. **Deploy.** The build is `npm ci && npm run build` (type-check + Vite, ~15 s). It does **not** run Python;
   the data in `public/data/` is committed.

Every later `git push` redeploys. Cache headers for `/data`, `/assets` and `/ort` are in `vercel.json`.

**Other static hosts** work the same way — Netlify, Cloudflare Pages, GitHub Pages. Build command
`npm run build`, publish directory `dist`. For a sub-path (GitHub Pages) build with
`npx vite build --base=/<repo>/`. Any host must serve `.wasm` as `application/wasm` (all of them do).

## 2. What visitors need

| | |
|---|---|
| Browser | Recent **Chrome or Edge** (WebGPU). Safari/Firefox: use the Ollama option below. |
| GPU | ≈ 4.2 GB free GPU memory for Qwen3-4B; ≈ 2.3 GB for 1.7B; ≈ 1.5 GB for 0.6B |
| First visit | Qwen3-4B ≈ **2.3 GB** (1.7B ≈ 1.1 GB), retrieval model 34 MB, ONNX runtime 14 MB |
| Next visits | Everything is cached by the browser; the model loads from disk in a few seconds |
| Phones / no-GPU laptops | The app still loads; chat needs the **My Ollama** option (below) |

The first customer lookup downloads one ~230 KB shard. An ad-hoc filter (`query_segment`) or a custom histogram
downloads the whole table once (≈ 23 MB, then cached) and runs it in the browser.

## 3. "My Ollama" fallback (no WebGPU, or you want the exact Streamlit model)

The page can talk straight to an Ollama on the visitor's own machine; inference stays local.

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
OLLAMA_ORIGINS="https://<your-app>.vercel.app" ollama serve     # allow your site's origin
```

Pages served from `localhost` are allowed by default, so `npm run dev` / `npm run preview` need nothing.
From an `https://` site Chrome may show a "access devices on your local network" prompt — allow it.
Firefox/Safari block `https` → `http://localhost` requests; use a Chromium browser or run the app locally.

## 4. Before you make it public — read this

* The site is **public** and ships the full customer feature table (as shards), the challenge deck's text,
  `trial.py` and the grounding file as knowledge-base chunks. Check the Campus Challenge's data and
  confidentiality terms before deploying. If in doubt, deploy privately (Vercel password protection / a
  private team) or drop the pieces you may not publish: delete `public/data/customers/` (customer lookups and
  ad-hoc queries stop working; everything else keeps working) and/or the `core_chunks.json` entries.
* Nothing the visitor types or uploads leaves their browser, except model downloads from Hugging Face.

## 5. Regenerating the data

Only needed if the scoring weights, the CSV or the corpus change. Needs the ProfitLens Python venv and the
repo-root data files (the Streamlit app's inputs).

```bash
cd web
npm run export-data   # ../.venv/bin/python scripts/export_data.py  (~1 min; verifies against submission_FINAL.xlsx)
npm run embed-core    # embeds the knowledge-base chunks with the same ONNX model the browser uses
npm test              # golden-checks the TypeScript tools against fresh Python output
git add public/data tests/golden
```

## 6. How it differs from the Streamlit app (honestly)

* **Same:** the 8 tools and their output shapes, the system prompt text, retrieval-before-answering, the 6-round
  tool loop, 12-message history, the charts, the "Digest your own files" tab, all the verification badges.
* **Model:** WebLLM ships Qwen3-4B (a hybrid thinking model, run with thinking off), not the Instruct-2507 build the
  Streamlit app uses through Ollama. Expect somewhat weaker tool use; the compact prompt and retrieval prefetch compensate.
* **Prompt:** the in-browser model gets a *compact* system prompt (~3K tokens: rules, formula, verified statistics) so it
  has room in an 8K window; the rest of `PROFITLENS_CONTEXT.md` is retrieved on demand. “Full” (verbatim file) is used
  with Ollama, or choose it under *Advanced*.
* **Embeddings:** `bge-small-en-v1.5` (384-d, ONNX) instead of `nomic-embed-text` via Ollama; the similarity floor stays 0.58.
* **Data:** customer profiles are recomputed in the browser from the raw features with the same frozen formula and are
  identical to the Python tool's output (tested on 170+ customers incl. ties, top-20% boundary, `f3 ≥ 1`).
  `query_segment` accepts the pandas `query` subset the model uses (comparisons, `and/or/not`, `& | ~`, arithmetic, chained
  comparisons, `in [...]`, `.isna()`), not arbitrary pandas.
* **Uploads:** pdf/docx/txt/md/csv/xlsx/py/json, parsed in the browser. Scanned (image-only) PDFs have no text, as in the Streamlit app.
