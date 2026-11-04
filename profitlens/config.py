"""
ProfitLens — Configuration (single source of truth)
Weights, file paths, model name, feature dictionary,
and Section-8 expected values for runtime verification.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root (one level up from profitlens/) ──────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# Local inference via Ollama (no API key, nothing leaves the laptop).
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

# Embedding model for the RAG layer, also served by Ollama.
# nomic-embed-text produces 768-dim vectors.
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))

# ── Data file paths (relative to the repo root) ─────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "6a3eb196bc7a3_campus_challenge_r1_data.csv"
FEATURE_CSV = ROOT_DIR / "6a3eb1af6b994_feature_description.csv"
SUBMISSION_XLSX = ROOT_DIR / "submission_FINAL.xlsx"
SCORED_PARQUET = ROOT_DIR / "profitlens" / "scored.parquet"
CONTEXT_MD_PATH = ROOT_DIR / "profitlens" / "PROFITLENS_CONTEXT.md"

# Corpus sources that live outside the app package
TRIAL_PY = ROOT_DIR / "main" / "trial.py"
CHALLENGE_PDF = ROOT_DIR / "6a3ea743b1106_campus_challenge26_r1.pdf"

# ── RAG store (gitignored, rebuilt from sources on demand) ───────────────
RAG_STORE_DIR = ROOT_DIR / "rag_store"
RAG_SIM_FLOOR = 0.58         # cosine floor (nomic-embed-text) below which a hit is "not relevant"
RAG_MAX_UPLOAD_MB = 25

# ── Master weights (frozen — do NOT modify) ──────────────────────────────
W = dict(
    ic=0.0240,      # discount (interchange) rate on billed business
    nii=0.331,      # net interest yield on revolving balance f1
    lng=55.0,       # lounge cost per visit (f13)
    cab=29.0,       # cab-credit cost per utilized month (f15)
    b14=1.9,        # airline-credit multiplier (f14, $)
    b16=1.5,        # entertainment-credit multiplier (f16, $)
    rs=1.35,        # expected-loss scale on f11 * f1
    a2=70.0,        # cancellation-call charge (f2)
    a3=8500.0,      # collections-call charge (f3) — effective exclusion
    w12=-4.5,       # servicing cost per website login (f12)
)

# ── Feature dictionary ───────────────────────────────────────────────────
FEATURE_DICT = {
    "f1":  "Average Revolve Balance in last 12m",
    "f2":  "Cancellation Calls in last 12m",
    "f3":  "Cancellation Calls due to Collection",
    "f4":  "Rewards Points Balance",
    "f5":  "Total Spend in last 12m",
    "f6":  "Airlines Spend in 12m",
    "f7":  "Other Spend in 12m",
    "f8":  "Entertainment Spend in 12m",
    "f9":  "Lodging Spend in 12m",
    "f10": "Dining Spend in 12m",
    "f11": "Average Risk Score in 12m",
    "f12": "Login Counts to website",
    "f13": "Lounge Access Count",
    "f14": "Credits used in airlines ($)",
    "f15": "Cab benefits usage (months utilized)",
    "f16": "Entertainment Credit Used Amount ($)",
    "f17": "Total Lend Line Amount",
    "f18": "Total Consumer Lend Line Amount",
    "f19": "Number of Supplementary Accounts",
    "f20": "Count of Active Charge Cards",
    "f21": "Rewards points redeemed in 12 months",
    "f22": "Emails Open in Last 6 months",
    "f23": "Emails Clicked in Last 6 months",
}

# ── Term decomposition metadata ─────────────────────────────────────────
TERM_COLUMNS = [
    "rev_discount", "rev_interest",
    "cost_lounge", "cost_cab", "cost_airline_credit", "cost_ent_credit",
    "cost_credit_loss", "cost_cancel_calls", "cost_collections", "cost_servicing",
]

TERM_MEANINGS = {
    "rev_discount":       "Discount/interchange revenue on spend",
    "rev_interest":       "Net interest income on revolving balance",
    "cost_lounge":        "Lounge benefit cost",
    "cost_cab":           "Cab credit cost",
    "cost_airline_credit":"Airline credit cost",
    "cost_ent_credit":    "Entertainment credit cost",
    "cost_credit_loss":   "Expected credit loss (risk × revolving exposure)",
    "cost_cancel_calls":  "Attrition/retention cost",
    "cost_collections":   "Collections-cancellation penalty (near-disqualifying)",
    "cost_servicing":     "Digital servicing cost",
}

# ── Equation string (generated from W, never hardcoded) ──────────────────
def build_equation_string() -> str:
    """Build the profit equation as a human-readable string from W."""
    return (
        f"profit = {W['ic']}×S_all + {W['nii']}×f1"
        f" − ({W['lng']}×f13 + {W['cab']}×f15 + {W['b14']}×f14 + {W['b16']}×f16)"
        f" − {W['rs']}×f11×f1"
        f" − {W['a2']}×f2"
        f" − {W['a3']}×f3"
        f" + ({W['w12']})×f12"
        f"\nwhere S_all = f6 + f7 + f8 + f9 + f10"
    )


# ── Section-8 expected values (hardcoded from PROFITLENS_CONTEXT.md §8) ──
# Update this dict ONLY when regenerating Section 8 via scripts/refresh_context_stats.py.
# These are compared against live-computed values at startup for the runtime verification badge.
EXPECTED = {
    "row_count":          500_000,
    "top20_cutoff":       2012.86,    # p80 of profit
    "mean_top20_profit":  4354.85,    # mean profit for top-20%
    "mean_rest_profit":   -922.40,    # mean profit for bottom-80%
    "pct_negative":       45.2,       # % customers with profit < 0
    "f3_in_top20":        0,          # count of customers with f3>=1 in top-20%
}
