"""
ProfitLens RAG — generated profile documents.

Raw 500K-row tables NEVER enter the vector store: row-level lookup belongs to
the pandas tools. Instead the dataset and the formula's scored output enter RAG
as compact, retrieval-friendly text cards computed from them. The same
treatment is applied to user-uploaded CSV/XLSX files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (FEATURE_DICT, TERM_COLUMNS, TERM_MEANINGS, W,
                    build_equation_string)
from rag.chunker import make_chunk

FEATURES = [f"f{i}" for i in range(1, 24)]

# Which weight drives which feature — used to state each feature's role.
_FEATURE_ROLE = {
    "f1":  f"revenue: net interest at {W['nii']}×f1, and exposure in the credit-loss term",
    "f2":  f"cost: cancellation-call charge at ${W['a2']:.0f} per call",
    "f3":  f"cost: collections-cancellation charge at ${W['a3']:.0f} per call (effectively disqualifying)",
    "f6":  f"revenue: part of S_all, discount rate {W['ic']}",
    "f7":  f"revenue: part of S_all, discount rate {W['ic']}",
    "f8":  f"revenue: part of S_all, discount rate {W['ic']}",
    "f9":  f"revenue: part of S_all, discount rate {W['ic']}",
    "f10": f"revenue: part of S_all, discount rate {W['ic']}",
    "f11": f"cost: expected-loss scale {W['rs']} applied to f11×f1",
    "f12": f"cost: digital servicing at ${abs(W['w12'])} per website login",
    "f13": f"cost: lounge benefit at ${W['lng']:.0f} per visit",
    "f14": f"cost: airline-credit multiplier {W['b14']}",
    "f15": f"cost: cab credit at ${W['cab']:.0f} per utilized month",
    "f16": f"cost: entertainment-credit multiplier {W['b16']}",
}


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):,.4g}"


def _numeric_card(s: pd.Series, name: str, desc: str, extra: str = "") -> str:
    v = s.dropna()
    if v.empty:
        return f"{name} — {desc}. All values missing ({len(s):,} rows)."
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return (
        f"{name} — {desc}.\n"
        f"Count: {len(v):,} non-missing of {len(s):,} rows "
        f"(missing {s.isna().mean() * 100:.2f}%).\n"
        f"mean={_fmt(v.mean())}, median={_fmt(q[0.5])}, "
        f"p5={_fmt(q[0.05])}, p25={_fmt(q[0.25])}, p75={_fmt(q[0.75])}, "
        f"p95={_fmt(q[0.95])}, min={_fmt(v.min())}, max={_fmt(v.max())}, "
        f"std={_fmt(v.std())}."
        + (f"\n{extra}" if extra else "")
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Core corpus — dataset profile docs
# ═══════════════════════════════════════════════════════════════════════════

def dataset_profile_docs(df: pd.DataFrame) -> list[dict]:
    """One card per feature f1–f23, one for S_all, one dataset overview."""
    src = "dataset profile (6a3eb196bc7a3_campus_challenge_r1_data.csv)"
    chunks: list[dict] = []
    top20 = df["is_top20"]

    for f in FEATURES:
        if f not in df.columns:
            continue
        s = df[f]
        t_mean = s[top20].mean()
        r_mean = s[~top20].mean()
        role = _FEATURE_ROLE.get(f, "not used in the profitability formula")
        extra = (f"Role in formula: {role}.\n"
                 f"Mean for top-20% customers: {_fmt(t_mean)}; "
                 f"for the rest: {_fmt(r_mean)}.")
        chunks.append(make_chunk(
            _numeric_card(s, f, FEATURE_DICT.get(f, f), extra),
            src, "data_profile", f"feature {f}"))

    s_all = df[["f6", "f7", "f8", "f9", "f10"]].fillna(0).sum(axis=1)
    chunks.append(make_chunk(
        _numeric_card(s_all, "S_all",
                      "total on-card billed business = f6+f7+f8+f9+f10",
                      f"Mean for top-20%: {_fmt(s_all[top20].mean())}; "
                      f"rest: {_fmt(s_all[~top20].mean())}. "
                      f"Discount revenue = {W['ic']}×S_all."),
        src, "data_profile", "derived S_all"))

    missing = {f: round(float(df[f].isna().mean() * 100), 2)
               for f in FEATURES if f in df.columns and df[f].isna().any()}
    chunks.append(make_chunk(
        f"Dataset overview — Amex Campus Challenge 2026 Round 1 data.\n"
        f"{len(df):,} customer rows, one row per unique identifier `id`.\n"
        f"Schema: id + features f1–f23 (all numeric), plus the P&L term columns "
        f"and profit computed by ProfitLens.\n"
        f"Missing-value semantics: a missing feature means NO recorded activity "
        f"and is scored as 0 — it is not an unknown to impute.\n"
        f"Features with missing values (%): "
        f"{', '.join(f'{k}={v}' for k, v in missing.items()) or 'none'}.\n"
        f"Row-level values are served by the pandas tools, not by retrieval.",
        src, "data_profile", "dataset overview"))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Core corpus — formula-output profile docs
# ═══════════════════════════════════════════════════════════════════════════

def formula_output_docs(df: pd.DataFrame, verification: dict | None = None) -> list[dict]:
    """One card per P&L term, plus profit-distribution, segment-economics and
    integrity cards, all computed from the scored output."""
    src = "scored output (profit model)"
    chunks: list[dict] = []
    top20, rest = df["is_top20"], ~df["is_top20"]
    profit = df["profit"]

    total_sep = sum(abs(df.loc[top20, c].mean() - df.loc[rest, c].mean())
                    for c in TERM_COLUMNS) or 1.0

    for col in TERM_COLUMNS:
        t_mean = float(df.loc[top20, col].mean())
        r_mean = float(df.loc[rest, col].mean())
        share = abs(t_mean - r_mean) / total_sep * 100
        chunks.append(make_chunk(
            f"P&L term {col} — {TERM_MEANINGS[col]}.\n"
            f"Definition in the frozen equation: see `{build_equation_string().splitlines()[0]}`.\n"
            f"Mean for top-20%: ${t_mean:,.2f}; for the rest: ${r_mean:,.2f}; "
            f"difference ${t_mean - r_mean:,.2f}.\n"
            f"Share of total top-vs-rest separation: {share:.1f}%.\n"
            f"Overall mean across all customers: ${float(df[col].mean()):,.2f}.",
            src, "data_profile", f"term {col}"))

    pcts = [1, 5, 10, 25, 50, 75, 80, 90, 95, 99]
    qs = profit.quantile([p / 100 for p in pcts])
    chunks.append(make_chunk(
        "Profit distribution across all customers.\n"
        + "\n".join(f"p{p} = ${float(qs[p / 100]):,.2f}" for p in pcts) + "\n"
        + f"mean = ${float(profit.mean()):,.2f}, "
          f"min = ${float(profit.min()):,.2f}, max = ${float(profit.max()):,.2f}.\n"
          f"Top-20% cutoff (p80 of profit) = ${float(df.loc[top20, 'profit'].min()):,.2f}.\n"
          f"{float((profit < 0).mean() * 100):.1f}% of customers have negative profit.",
        src, "data_profile", "profit distribution"))

    chunks.append(make_chunk(
        f"Segment economics — top-20% vs the rest.\n"
        f"Top-20%: {int(top20.sum()):,} customers, mean profit "
        f"${float(df.loc[top20, 'profit'].mean()):,.2f}, median "
        f"${float(df.loc[top20, 'profit'].median()):,.2f}.\n"
        f"Rest (bottom 80%): {int(rest.sum()):,} customers, mean profit "
        f"${float(df.loc[rest, 'profit'].mean()):,.2f}, median "
        f"${float(df.loc[rest, 'profit'].median()):,.2f}.\n"
        f"Segmentation rule: customers are rank-ordered descending by profit; "
        f"percentile rank ≥ 0.80 is the predicted most-profitable segment.\n"
        f"Total profit pool: ${float(profit.sum()):,.0f}; the top-20% contribute "
        f"${float(df.loc[top20, 'profit'].sum()):,.0f}.",
        src, "data_profile", "segment economics"))

    if verification:
        chunks.append(make_chunk(
            f"Integrity check — submission_FINAL.xlsx Predictions sheet.\n"
            f"The Predictions sheet holds the profit value rounded to 4 decimal "
            f"places, one row per unique identifier.\n"
            f"Verified match: {verification.get('match_count', 0):,}/"
            f"{verification.get('total', 0):,} rows agree with the live-computed "
            f"profit within 1e-6.\n"
            f"Status: {verification.get('message', 'n/a')}",
            src, "data_profile", "integrity check"))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  User uploads — tabular profile docs
# ═══════════════════════════════════════════════════════════════════════════

STATS_ONLY_ROWS = 50_000
PREVIEW_ROWS = 20


def tabular_profile_docs(df: pd.DataFrame, source: str, label: str) -> list[dict]:
    """Profile-doc treatment for an uploaded CSV or one XLSX sheet:
    schema + dtypes + per-column stats + a small preview. Never full rows."""
    chunks: list[dict] = []
    big = len(df) > STATS_ONLY_ROWS

    schema = "\n".join(
        f"- {c} ({df[c].dtype}), missing {df[c].isna().mean() * 100:.1f}%"
        for c in df.columns)
    overview = (
        f"Uploaded table `{label}` — schema overview.\n"
        f"{len(df):,} rows × {len(df.columns)} columns.\n"
        f"Columns:\n{schema}"
    )
    if big:
        overview += (f"\n(Table exceeds {STATS_ONLY_ROWS:,} rows — "
                     f"summary statistics only, no row preview.)")
    chunks.append(make_chunk(overview, source, "user", f"{label}: schema"))

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            text = _numeric_card(s, f"`{label}`.{col}", f"numeric column ({s.dtype})")
        else:
            vc = s.astype("string").value_counts().head(10)
            text = (
                f"`{label}`.{col} — non-numeric column ({s.dtype}).\n"
                f"{s.notna().sum():,} non-missing of {len(s):,} rows "
                f"(missing {s.isna().mean() * 100:.2f}%), "
                f"{s.nunique(dropna=True):,} distinct values.\n"
                "Most frequent values:\n"
                + "\n".join(f"- {k}: {v:,}" for k, v in vc.items())
            )
        chunks.append(make_chunk(text, source, "user", f"{label}: column {col}"))

    if not big and len(df) > 0:
        head = df.head(PREVIEW_ROWS)
        try:
            preview = head.to_markdown(index=False)
        except ImportError:      # tabulate not installed
            preview = head.to_string(index=False)
        chunks.append(make_chunk(
            f"Uploaded table `{label}` — first "
            f"{min(PREVIEW_ROWS, len(df))} rows:\n\n{preview}",
            source, "user", f"{label}: preview rows 1-{min(PREVIEW_ROWS, len(df))}"))
    return chunks
