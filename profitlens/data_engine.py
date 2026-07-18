"""
ProfitLens — Data Engine
Load CSV, compute profitability scores & term decomposition,
cache to parquet, verify against submission, build stats pack.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path

from config import W, DATA_CSV, SUBMISSION_XLSX, SCORED_PARQUET, TERM_COLUMNS, EXPECTED


# ═══════════════════════════════════════════════════════════════════════════
#  Core scoring
# ═══════════════════════════════════════════════════════════════════════════

def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add P&L term columns, profit, percentile rank, top-20% flag."""
    d = df.copy()
    # Fill NaN with 0 for scoring
    f = {c: d[c].fillna(0) for c in [f"f{i}" for i in range(1, 24)]}

    # S_all = total on-card billed business
    s_all = f["f6"] + f["f7"] + f["f8"] + f["f9"] + f["f10"]

    # Term decomposition
    d["rev_discount"]       =  W["ic"]  * s_all
    d["rev_interest"]       =  W["nii"] * f["f1"]
    d["cost_lounge"]        = -W["lng"] * f["f13"]
    d["cost_cab"]           = -W["cab"] * f["f15"]
    d["cost_airline_credit"]= -W["b14"] * f["f14"]
    d["cost_ent_credit"]    = -W["b16"] * f["f16"]
    d["cost_credit_loss"]   = -W["rs"]  * f["f11"] * f["f1"]
    d["cost_cancel_calls"]  = -W["a2"]  * f["f2"]
    d["cost_collections"]   = -W["a3"]  * f["f3"]
    d["cost_servicing"]     =  W["w12"] * f["f12"]

    # Profit = sum of all terms
    d["profit"] = d[TERM_COLUMNS].sum(axis=1)

    # Percentile rank & top-20% flag
    d["pct_rank"] = d["profit"].rank(pct=True)
    d["is_top20"] = (d["pct_rank"] >= 0.80)

    return d


# ═══════════════════════════════════════════════════════════════════════════
#  Load / cache
# ═══════════════════════════════════════════════════════════════════════════

def load_scored() -> pd.DataFrame:
    """Load scored DataFrame, using parquet cache if fresh."""
    csv_mtime = DATA_CSV.stat().st_mtime if DATA_CSV.exists() else 0
    parquet_mtime = SCORED_PARQUET.stat().st_mtime if SCORED_PARQUET.exists() else 0

    if SCORED_PARQUET.exists() and parquet_mtime > csv_mtime:
        return pd.read_parquet(SCORED_PARQUET)

    df = pd.read_csv(DATA_CSV)
    df = _compute_scores(df)
    # Ensure directory exists
    SCORED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCORED_PARQUET, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Integrity verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_against_submission(df: pd.DataFrame) -> dict:
    """
    Compare computed profit against submission_FINAL.xlsx Predictions sheet.
    Returns {"match_count": int, "total": int, "mismatches": int, "ok": bool, "message": str}.
    Never raises — returns an error message if something goes wrong.
    """
    try:
        sub = pd.read_excel(SUBMISSION_XLSX, sheet_name="Predictions")
        sub.columns = [c.strip() for c in sub.columns]
        # Normalise column names
        if "ID" in sub.columns:
            sub = sub.rename(columns={"ID": "id"})
        if "Prediction" not in sub.columns:
            return {"ok": False, "message": "No 'Prediction' column in xlsx.", "match_count": 0, "total": 0, "mismatches": 0}

        merged = df[["id", "profit"]].merge(sub[["id", "Prediction"]], on="id", how="inner")
        total = len(merged)

        # Round profit to 4 decimal places before comparing
        diffs = (merged["profit"].round(4) - merged["Prediction"]).abs()
        matches = (diffs < 1e-6).sum()
        mismatches = total - matches

        if mismatches == 0:
            msg = f"✅ Formula matches submitted predictions ({total:,}/{total:,})"
        else:
            msg = f"⚠️ {mismatches:,} mismatches out of {total:,} rows"

        return {"ok": bool(mismatches == 0), "match_count": int(matches), "total": total, "mismatches": int(mismatches), "message": msg}

    except Exception as e:
        return {"ok": False, "message": f"Verification error: {e}", "match_count": 0, "total": 0, "mismatches": 0}


# ═══════════════════════════════════════════════════════════════════════════
#  Stats pack (compact, for Gemini system prompt — <1,500 tokens)
# ═══════════════════════════════════════════════════════════════════════════

def build_stats_pack(df: pd.DataFrame) -> dict:
    """Build a compact statistics summary for the system prompt."""
    top20 = df[df["is_top20"]]
    bot80 = df[~df["is_top20"]]
    cutoff = df.loc[df["is_top20"], "profit"].min()

    # Mean of each P&L term for top-20% vs rest
    term_means = {}
    for col in TERM_COLUMNS:
        term_means[col] = {
            "top20_mean": round(float(top20[col].mean()), 2),
            "rest_mean":  round(float(bot80[col].mean()), 2),
        }

    # Missing-value % per feature
    feature_cols = [f"f{i}" for i in range(1, 24)]
    missing_pct = {}
    for c in feature_cols:
        pct = round(float(df[c].isna().mean() * 100), 2)
        if pct > 0:
            missing_pct[c] = pct

    return {
        "row_count": len(df),
        "top20_cutoff": round(float(cutoff), 4),
        "top20_count": len(top20),
        "profit_stats": {
            "top20": {
                "mean": round(float(top20["profit"].mean()), 2),
                "median": round(float(top20["profit"].median()), 2),
            },
            "rest": {
                "mean": round(float(bot80["profit"].mean()), 2),
                "median": round(float(bot80["profit"].median()), 2),
            },
        },
        "term_means": term_means,
        "missing_pct": missing_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Runtime verification (context-stats self-consistency guard)
# ═══════════════════════════════════════════════════════════════════════════

def compute_runtime_verification(df: pd.DataFrame) -> dict:
    """
    Compute 5 live stats and compare against EXPECTED (Section 8 values).

    Returns:
        {
          "live": {...live computed values...},
          "mismatches": [list of mismatched key names],
          "all_match": bool,
          "block": str,  # text to append to system instruction
        }
    Never raises.
    """
    try:
        top20 = df[df["is_top20"]]
        bot80 = df[~df["is_top20"]]

        # 1. row count
        row_count = len(df)
        # 2. top-20% cutoff (p80 of profit)
        top20_cutoff = round(float(top20["profit"].min()), 2)
        # 3. mean profit top-20% vs rest
        mean_top20 = round(float(top20["profit"].mean()), 2)
        mean_rest  = round(float(bot80["profit"].mean()), 2)
        # 4. % customers with profit < 0
        pct_negative = round(float((df["profit"] < 0).mean() * 100), 1)
        # 5. count of customers with f3 >= 1 in top-20%
        # raw CSV f3 is NOT stored in the parquet; use cost_collections proxy:
        # cost_collections = -8500 * f3, so f3>=1 iff cost_collections <= -8500
        f3_in_top20 = int((top20["cost_collections"] <= -8500.0).sum())

        live = {
            "row_count":         row_count,
            "top20_cutoff":      top20_cutoff,
            "mean_top20_profit": mean_top20,
            "mean_rest_profit":  mean_rest,
            "pct_negative":      pct_negative,
            "f3_in_top20":       f3_in_top20,
        }

        # Compare each live value against EXPECTED
        mismatches = []
        tolerances = {
            "row_count":         0,
            "top20_cutoff":      0.5,    # $0.50 tolerance for float rounding
            "mean_top20_profit": 1.0,
            "mean_rest_profit":  1.0,
            "pct_negative":      0.1,
            "f3_in_top20":       0,
        }
        for key, exp_val in EXPECTED.items():
            live_val = live[key]
            tol = tolerances.get(key, 0)
            if abs(float(live_val) - float(exp_val)) > tol:
                mismatches.append(key)

        all_match = len(mismatches) == 0
        if all_match:
            status_line = "Status: ALL MATCH Section 8"
        else:
            status_line = f"Status: MISMATCH on: {', '.join(mismatches)} — trust these live values over Section 8."

        block = (
            "\n\n## RUNTIME VERIFICATION (auto-generated at startup)\n"
            f"Live-computed on the loaded dataset: "
            f"N={row_count:,}, cutoff=${top20_cutoff:,.2f}, "
            f"mean_top20=${mean_top20:,.2f}, mean_rest=${mean_rest:,.2f}, "
            f"pct_negative={pct_negative}%, f3_in_top20={f3_in_top20}\n"
            f"{status_line}\n"
        )

        return {
            "live": live,
            "mismatches": mismatches,
            "all_match": all_match,
            "block": block,
        }

    except Exception as e:
        return {
            "live": {},
            "mismatches": ["error"],
            "all_match": False,
            "block": f"\n\n## RUNTIME VERIFICATION\nError during verification: {e}\n",
        }
