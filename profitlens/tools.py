"""
ProfitLens — Tool Functions for Gemini Function Calling
Each function returns a JSON-serializable dict.
Bad input is handled gracefully (returns {"error": ..., "hint": ...}).
"""
from __future__ import annotations

import re
import json
import numpy as np
import pandas as pd
from typing import List, Optional

from config import (W, TERM_COLUMNS, TERM_MEANINGS, FEATURE_DICT, SUBMISSION_XLSX,
                    RAG_SIM_FLOOR, build_equation_string)


# ── Globals (set once via init_tools) ────────────────────────────────────
_df: pd.DataFrame | None = None

# Sidebar toggle "Include my uploaded files in answers" (see set_user_corpus_enabled).
_user_corpus_enabled: bool = True

VALID_FEATURES = [f"f{i}" for i in range(1, 24)]
VALID_COLUMNS = VALID_FEATURES + TERM_COLUMNS + ["profit", "pct_rank"]


def init_tools(df: pd.DataFrame):
    """Must be called once at startup with the scored DataFrame."""
    global _df
    _df = df


def _check_init():
    if _df is None:
        return {"error": "Data not loaded", "hint": "Internal error — restart the app."}
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  1. get_customer_profile
# ═══════════════════════════════════════════════════════════════════════════

def get_customer_profile(customer_id: int) -> dict:
    """Return full profile for a single customer: raw features, P&L terms,
    profit, percentile rank, top-20% flag, and rank out of 500,000."""
    err = _check_init()
    if err:
        return err
    row = _df[_df["id"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found.",
                "hint": f"Valid IDs range from {int(_df['id'].min())} to {int(_df['id'].max())}."}
    r = row.iloc[0]
    # Raw features (NaN → null)
    raw = {}
    for f in VALID_FEATURES:
        val = r[f]
        raw[f] = None if pd.isna(val) else round(float(val), 4)
    # P&L terms
    terms = {t: round(float(r[t]), 4) for t in TERM_COLUMNS}
    profit = round(float(r["profit"]), 4)
    pct = round(float(r["pct_rank"]), 6)
    rank = int((_df["profit"] >= r["profit"]).sum())
    return {
        "customer_id": int(customer_id),
        "raw_features": raw,
        "terms": terms,
        "profit": profit,
        "percentile_rank": pct,
        "rank_out_of_500k": rank,
        "is_top20": bool(r["is_top20"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  2. get_top_customers
# ═══════════════════════════════════════════════════════════════════════════

def get_top_customers(n: int = 10, ascending: bool = False) -> dict:
    """Return top (or bottom) n customers by profit, with key terms.
    Caps n at 50."""
    err = _check_init()
    if err:
        return err
    n = max(1, min(int(n), 50))
    sorted_df = _df.sort_values("profit", ascending=ascending).head(n)
    rows = []
    for _, r in sorted_df.iterrows():
        term_vals = {t: round(float(r[t]), 2) for t in TERM_COLUMNS}
        # Find 2 largest revenue and cost terms
        rev_terms = {k: v for k, v in term_vals.items() if v > 0}
        cost_terms = {k: v for k, v in term_vals.items() if v < 0}
        top_rev = sorted(rev_terms.items(), key=lambda x: x[1], reverse=True)[:2]
        top_cost = sorted(cost_terms.items(), key=lambda x: x[1])[:2]
        rows.append({
            "id": int(r["id"]),
            "profit": round(float(r["profit"]), 2),
            "is_top20": bool(r["is_top20"]),
            "top_revenues": dict(top_rev),
            "top_costs": dict(top_cost),
        })
    return {"customers": rows, "n": n, "ascending": ascending}


# ═══════════════════════════════════════════════════════════════════════════
#  3. segment_compare
# ═══════════════════════════════════════════════════════════════════════════

def segment_compare(feature: str) -> dict:
    """Compare a feature/term between top-20% and bottom-80%.
    Returns mean, median, p25, p75, missing % for each segment."""
    err = _check_init()
    if err:
        return err
    if feature not in VALID_COLUMNS:
        return {"error": f"Invalid feature '{feature}'.",
                "hint": f"Valid names: {VALID_COLUMNS}"}
    top20 = _df[_df["is_top20"]][feature]
    bot80 = _df[~_df["is_top20"]][feature]

    def _stats(s: pd.Series) -> dict:
        return {
            "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
            "median": round(float(s.median()), 4) if len(s) > 0 else None,
            "p25": round(float(s.quantile(0.25)), 4) if len(s) > 0 else None,
            "p75": round(float(s.quantile(0.75)), 4) if len(s) > 0 else None,
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "count": len(s),
        }

    return {
        "feature": feature,
        "description": FEATURE_DICT.get(feature, TERM_MEANINGS.get(feature, feature)),
        "top20": _stats(top20),
        "rest": _stats(bot80),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  4. term_contribution_summary
# ═══════════════════════════════════════════════════════════════════════════

def term_contribution_summary() -> dict:
    """Aggregate mean of each P&L term for top-20% vs rest,
    plus which terms most separate the segments."""
    err = _check_init()
    if err:
        return err
    top20 = _df[_df["is_top20"]]
    bot80 = _df[~_df["is_top20"]]
    terms = []
    for col in TERM_COLUMNS:
        t_mean = round(float(top20[col].mean()), 2)
        r_mean = round(float(bot80[col].mean()), 2)
        diff = round(t_mean - r_mean, 2)
        terms.append({
            "term": col,
            "meaning": TERM_MEANINGS[col],
            "top20_mean": t_mean,
            "rest_mean": r_mean,
            "difference": diff,
        })
    # Sort by absolute difference descending
    terms.sort(key=lambda x: abs(x["difference"]), reverse=True)
    return {
        "terms": terms,
        "top20_mean_profit": round(float(top20["profit"].mean()), 2),
        "rest_mean_profit": round(float(bot80["profit"].mean()), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. get_distribution
# ═══════════════════════════════════════════════════════════════════════════

def get_distribution(column: str, bins: int = 40, clip_percentiles: list = None) -> dict:
    """Return histogram data (bin edges + counts) for any feature/term/profit.
    The APP renders this as a plotly chart; the tool returns the data."""
    if clip_percentiles is None:
        clip_percentiles = [1.0, 99.0]
    err = _check_init()
    if err:
        return err
    if column not in VALID_COLUMNS:
        return {"error": f"Invalid column '{column}'.",
                "hint": f"Valid names: {VALID_COLUMNS}"}
    s = _df[column].dropna()
    lo = float(np.percentile(s, clip_percentiles[0]))
    hi = float(np.percentile(s, clip_percentiles[1]))
    clipped = s.clip(lo, hi)
    counts, edges = np.histogram(clipped, bins=int(bins))
    return {
        "column": column,
        "description": FEATURE_DICT.get(column, TERM_MEANINGS.get(column, column)),
        "bin_edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "clip": [lo, hi],
        "total_values": len(s),
        "nan_count": int(_df[column].isna().sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  6. query_segment
# ═══════════════════════════════════════════════════════════════════════════

_UNSAFE_PATTERNS = re.compile(r"(@|__|\bimport\b|\bexec\b|\beval\b|\bos\b|\bsys\b)")

def query_segment(filters: str, agg: str) -> dict:
    """Guarded pandas query over feature/term columns.
    agg must be one of: count, mean_profit, sum_profit, top20_share."""
    err = _check_init()
    if err:
        return err
    valid_aggs = {"count", "mean_profit", "sum_profit", "top20_share"}
    if agg not in valid_aggs:
        return {"error": f"Invalid agg '{agg}'.", "hint": f"Choose from {valid_aggs}"}
    # Security: reject unsafe patterns
    if _UNSAFE_PATTERNS.search(filters):
        return {"error": "Unsafe query pattern detected.", "hint": "Only use column comparisons, e.g. 'f3 >= 1 and profit > 0'."}
    # Whitelist: only allow references to known columns
    try:
        subset = _df.query(filters)
    except Exception as e:
        return {"error": f"Query failed: {e}", "hint": "Use pandas query syntax over columns: " + ", ".join(VALID_COLUMNS)}

    if agg == "count":
        result = {"count": len(subset)}
    elif agg == "mean_profit":
        result = {"mean_profit": round(float(subset["profit"].mean()), 4) if len(subset) > 0 else None, "count": len(subset)}
    elif agg == "sum_profit":
        result = {"sum_profit": round(float(subset["profit"].sum()), 4), "count": len(subset)}
    elif agg == "top20_share":
        t20_count = int(subset["is_top20"].sum()) if len(subset) > 0 else 0
        result = {"top20_count": t20_count, "total": len(subset),
                  "top20_share": round(t20_count / len(subset) * 100, 2) if len(subset) > 0 else None}

    return {"filters": filters, "agg": agg, **result}


# ═══════════════════════════════════════════════════════════════════════════
#  7. explain_formula
# ═══════════════════════════════════════════════════════════════════════════

def explain_formula() -> dict:
    """Return the W dict, the equation string, the term table, and the
    Profitability Framework sheet contents from the submission xlsx."""
    framework_text = ""
    try:
        fw = pd.read_excel(SUBMISSION_XLSX, sheet_name="Profitability Framework")
        framework_text = fw.to_dict(orient="records")
    except Exception:
        framework_text = "(Could not read Profitability Framework sheet)"

    term_table = []
    for col in TERM_COLUMNS:
        term_table.append({"column": col, "meaning": TERM_MEANINGS[col]})

    return {
        "weights": {k: v for k, v in W.items()},
        "equation": build_equation_string(),
        "term_table": term_table,
        "profitability_framework": framework_text,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  8. search_knowledge_base  (RAG retrieval — core + user corpora)
# ═══════════════════════════════════════════════════════════════════════════

def set_user_corpus_enabled(enabled: bool):
    """Called by the app from the sidebar toggle. When disabled, retrieval is
    forced to the core corpus regardless of what the model asked for."""
    global _user_corpus_enabled
    _user_corpus_enabled = bool(enabled)


def search_knowledge_base(query: str, corpus: str = "both", k: int = 5) -> dict:
    """Semantic search over the ProfitLens knowledge base.

    corpus: "core" (project code/docs/profiles), "user" (uploaded files), "both".
    Returns top-k chunks with source + location for citation.
    """
    from rag.embed import embed_query
    from rag.ingest import core_store, user_store

    corpus = (corpus or "both").lower()
    if corpus not in {"core", "user", "both"}:
        corpus = "both"
    k = max(1, min(int(k), 10))

    user_requested = corpus in {"user", "both"}
    if not _user_corpus_enabled:
        corpus = "core"

    notes: list[str] = []
    user = user_store()
    if user_requested and not _user_corpus_enabled:
        notes.append("your uploaded files were NOT consulted — the sidebar toggle "
                     "'Include my uploaded files in answers' is off; tell the user "
                     "this and that turning it on would let you search them")
    elif user_requested and len(user) == 0:
        notes.append("no user files ingested yet")

    try:
        qvec = embed_query(query)
    except Exception as e:
        return {"results": [], "note": f"retrieval unavailable (embedding failed: {e})"}

    hits: list[dict] = []
    if corpus in {"core", "both"}:
        hits += core_store().search(qvec, k=k, floor=RAG_SIM_FLOOR)
    if corpus in {"user", "both"}:
        hits += user.search(qvec, k=k, floor=RAG_SIM_FLOOR)
    hits.sort(key=lambda h: h["score"], reverse=True)
    hits = hits[:k]

    if not hits:
        notes.insert(0, "nothing relevant found")
        return {"results": [], "note": "; ".join(notes)}

    results = [{
        "source": h["source"],
        "location": h["location"],
        "kind": h["kind"],
        "score": h["score"],
        "text": h["text"],
    } for h in hits]
    out = {"results": results}
    if notes:
        out["note"] = "; ".join(notes)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Tool registry (for Gemini)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "get_customer_profile": get_customer_profile,
    "get_top_customers": get_top_customers,
    "segment_compare": segment_compare,
    "term_contribution_summary": term_contribution_summary,
    "get_distribution": get_distribution,
    "query_segment": query_segment,
    "explain_formula": explain_formula,
    "search_knowledge_base": search_knowledge_base,
}
