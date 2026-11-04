# 🏆 Amex Campus Challenge 2026 — Overall Campus Winner: ProfitLens

[![Overall Campus Winner](https://img.shields.io/badge/Award-OVERALL%20CAMPUS%20WINNER%20%F0%9F%8F%86-gold?style=for-the-badge)](README.md)
[![Vercel Demo](https://img.shields.io/badge/Vercel-Demo%20Live-brightgreen?style=for-the-badge&logo=vercel)](https://profitlens-amexcampuschallenge2026.vercel.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](Dockerfile)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?style=for-the-badge)](https://www.python.org/)
[![Vite + React](https://img.shields.io/badge/Frontend-Vite%20%2B%20React-purple.svg?style=for-the-badge)](web/)

> **🏆 OVERALL CAMPUS WINNER**: Official winning submission for the American Express Campus Challenge 2026 (PEC • Product Track).
>
> The challenge provided raw cardholder features (500,000 customers across 23 anonymized attributes) and tasked teams with formulating a profitability equation to identify the top 20% most valuable accounts.
>
> While the baseline brief only asked for an equation and a scored submission sheet, **this project won the Overall Campus Winner title by engineering an enterprise-grade AI analytics ecosystem**: pairing a mathematically rigorous, deterministic 10-term P&L model with a local LLM-powered financial analyst chatbot, automated P&L waterfall visualizations, local RAG document grounding, containerized microservices, and a 100% free serverless web app deployed to Vercel.

---

### 🌐 Live Interactive Demo
👉 **[profitlens-amexcampuschallenge2026.vercel.app](https://profitlens-amexcampuschallenge2026.vercel.app)**  
*Experience the instant client-side prototype directly in your browser — zero download, zero server latency, testable on mobile and desktop.*

---

## 📸 Screenshots & UI Walkthrough

| 1. Customer P&L Breakdown & Waterfall Chart |
| :--- |
| ![ProfitLens Customer 1 P&L Breakdown](assets/profitlens_customer_breakdown.png) |
| *Interactive horizontal waterfall chart detailing positive revenue drivers ($355.52 discount interchange) versus negative operational cost drags (airline credits $-372.40, cab benefits $-145.00, digital servicing $-261.00).* |

| 2. Formula Term-by-Term Explanation |
| :--- |
| ![ProfitLens Profitability Formula Explanation](assets/profitlens_formula_explanation.png) |
| *Complete decomposition of the 10-term economic formula with exact coefficients, signs, and business logic derived via single-lever response-curve analysis.* |

---

## 🌟 Why This Solution Won: Going Beyond The Prompt

| Challenge Requirement | Standard Baseline Approach | 🏆 Our Winning Solution (ProfitLens) |
| :--- | :--- | :--- |
| **Scoring Formula** | Generic regression or ungrounded formula | **10-Term Transparent P&L Economic Model**: Separates merchant discount interchange and net interest income from operational benefit costs, credit risk write-offs, and collections exclusions. Validated via single-lever response curves and ablation tests. |
| **Deliverables** | Static submission spreadsheet | **Interactive Financial Intelligence Platform**: Streamlit desktop dashboard + 100% free client-side static web application (`web/`). |
| **Explainability** | High-level summary text | **Conversational AI Financial Analyst**: Multi-turn local LLM (Ollama / Qwen3 / Gemini) function-calling chatbot that breaks down any customer's P&L dynamically without hallucinating figures. |
| **Visual Analytics** | None / manual plotting | **Auto-Rendered P&L Waterfall Charts**: Dynamic horizontal waterfall visualizations showing green positive revenue drivers vs red cost drags, distribution histograms, and segment comparisons. |
| **Knowledge Grounding** | Isolated script | **Local RAG (Retrieval-Augmented Generation)**: FAISS vector retrieval across challenge rules, methodology documents, and an interactive "Digest Your Own Files" tab for custom user documents (PDF/DOCX/XLSX). |
| **Deployment & Scale** | Difficult to demo without local setup | **1-Click Free Hosting on Vercel (`web/`)**: Custom binary sharding (~230KB per lookup) and client-side execution, making the demo instantly shareable on Vercel with zero server costs. |

---

## 🚀 Live Demo on Vercel (Free 1-Click Hosting)

The `web/` directory contains an instant, interactive client-side application hosted on Vercel:
👉 **[profitlens-amexcampuschallenge2026.vercel.app](https://profitlens-amexcampuschallenge2026.vercel.app)**

### How to deploy your own instance to Vercel for free:
1. Push this repository to your GitHub: `https://github.com/kirat11X/Amex-Campus-Challenge`.
2. Go to [Vercel Dashboard](https://vercel.com/new) and click **"Add New Project"** -> **"Import"** your GitHub repository.
3. In the project setup:
   - **Root Directory**: Click *Edit* and select `web`
   - **Framework Preset**: Auto-detected as **Vite**
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Click **Deploy**!
5. In ~30 seconds, your live URL is active.

### Live Hostable Modes:
- **🟢 Instant Demo Mode (Default)**: Completely serverless & zero-download. Visitors on any device (phone, tablet, Safari, Chrome) can immediately click all sample prompts, inspect customer 1, customer 20, or type any customer ID (1–500,000) to see instant P&L breakdowns and interactive charts.
- **🟣 In-Browser WebLLM**: Optional WebGPU mode running local Qwen3 models directly in the visitor's browser.
- **🦙 My Ollama**: Connects directly to a local Ollama instance on `localhost:11434`.

---

## 🧮 The Winning Profitability Formula

The scoring weights are frozen in `main/trial.py` (`W` dict) and mirrored in `profitlens/config.py`:

```
profit = 0.024 × S_all + 0.331 × f1
       − (55 × f13 + 29 × f15 + 1.9 × f14 + 1.5 × f16)
       − 1.35 × f11 × f1
       − 70 × f2 − 8500 × f3
       + (−4.5) × f12

where S_all = f6 + f7 + f8 + f9 + f10
```

### Term-by-Term Business Meaning
| Term | Formula | Sign | Business Meaning |
| :--- | :--- | :--- | :--- |
| **rev_discount** | `0.0240 × S_all` | `+` | Merchant discount (Interchange) revenue on billed business (`f6+f7+f8+f9+f10`) |
| **rev_interest** | `0.331 × f1` | `+` | Net Interest Income (NII) on revolving balance (`f1`) |
| **cost_lounge** | `−55 × f13` | `−` | Lounge access cost at fully-loaded issuer cost ($55/visit) |
| **cost_cab** | `−29 × f15` | `−` | Cab credit cost at $29 per utilized month |
| **cost_airline_credit** | `−1.9 × f14` | `−` | Airline fee credit cost at 1.9× loaded multiplier |
| **cost_entertainment_credit** | `−1.5 × f16` | `−` | Entertainment credit cost at 1.5× loaded multiplier |
| **cost_credit_loss** | `−1.35 × f11 × f1` | `−` | Expected write-offs: risk rate (`f11`) × revolving exposure (`f1`) |
| **cost_cancel_calls** | `−70 × f2` | `−` | Retention cost per cancellation call (`f2`) |
| **cost_collections** | `−8500 × f3` | `−` | Collections-driven cancellation penalty — soft exclusion |
| **cost_servicing** | `−4.5 × f12` | `−` | Digital servicing cost per website login (`f12`) |

**Top-20% Cutoff:** Customers with profit at or above the 80th percentile (**$2,012.86**). Mean profit of top-20% is **$4,354.85** (vs. -$922.40 for bottom 80%).

---

## 🐳 Docker Deployment

The repo includes a multi-container Docker configuration with Streamlit and an embedded Ollama service:

```bash
# 1. Build and run with CPU
docker compose up --build

# 2. Or run with NVIDIA GPU acceleration (requires nvidia-container-toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Access Streamlit at:
# http://localhost:8501
```

To test the standalone application container directly:
```bash
docker build -t profitlens:latest .
docker run -p 8501:8501 profitlens:latest
```

---

## 💻 Local Setup (Streamlit App)

```bash
# 1. Pull required Ollama models
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull nomic-embed-text

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r profitlens/requirements.txt

# 3. Launch ProfitLens
cd profitlens
streamlit run app.py
```

---

## 📁 Repository Layout

```
amex/
├── assets/                                       # UI screenshots & visual assets
│   ├── profitlens_customer_breakdown.png
│   └── profitlens_formula_explanation.png
├── 6a3eb196bc7a3_campus_challenge_r1_data.csv    # Raw dataset (500K rows, f1-f23)
├── 6a3eb1af6b994_feature_description.csv         # Feature dictionary
├── 6a3ea743b1106_campus_challenge26_r1.pdf       # Challenge brief
├── submission_FINAL.xlsx                         # Final verified submission workbook
├── Dockerfile                                    # Optimized Docker container definition
├── docker-compose.yml                            # Orchestrates app + Ollama service
├── docker-compose.gpu.yml                        # GPU reservation override for Ollama
│
├── profitlens/                                   # Streamlit chatbot application
│   ├── app.py                                    # Streamlit UI with Plotly charts
│   ├── config.py                                 # Single source of truth (W dict, paths, metadata)
│   ├── data_engine.py                            # Scoring, parquet caching, runtime verification
│   ├── tools.py                                  # Guarded analytical tools
│   ├── llm.py                                    # Ollama client & function-calling loop
│   ├── rag/                                      # FAISS vector store & embeddings
│   └── PROFITLENS_CONTEXT.md                     # System instruction & dataset grounding context
│
├── web/                                          # Dummy hostable web app for Vercel
│   ├── src/                                      # React + TypeScript frontend
│   │   ├── llm/demo.ts                           # Instant Hostable Demo Engine
│   │   ├── ui/Charts.tsx                         # SVG P&L waterfall & histogram charts
│   │   ├── ui/Sidebar.tsx                        # Dataset cards & try asking prompts
│   │   └── data/dataset.ts                       # Bit-exact client-side customer scoring
│   ├── public/data/                              # Static binary shards & metadata
│   ├── vercel.json                               # Vercel deployment configuration
│   └── HOSTING.md                                # Deployment guide
│
└── main/                                         # Challenge exploration & submission pipeline
    ├── trial.py                                  # Generates submission_FINAL.xlsx
    └── pipeline_v2.py, pipeline_v3.py            # Historical iterations
```

---

## 🛠️ Sample Questions to Try

- `Profit breakdown for customer 1`
- `Why is customer 20 in the top 20%?`
- `Compare lounge usage between top 20% and rest`
- `Show the profit distribution`
- `Which cost term hurts profitability the most?`
- `Explain my profitability formula term by term`
