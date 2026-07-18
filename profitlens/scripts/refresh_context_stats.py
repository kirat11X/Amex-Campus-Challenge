#!/usr/bin/env python3
"""
scripts/refresh_context_stats.py
─────────────────────────────────
Recomputes the Section-8 ground-truth stats and prints them in the exact
table / prose format used in PROFITLENS_CONTEXT.md.

USAGE (from the project root with the venv active):
    python scripts/refresh_context_stats.py

This script is READ-ONLY — it prints the stats; you paste them manually
into Section 8. Never auto-edit the markdown.

Rule: run this whenever the formula, weights, or dataset changes.
"""
import sys
from pathlib import Path

# Add profitlens/ to path so we can import project modules
# scripts/ lives inside profitlens/, so parent is profitlens/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_engine import load_scored

print("Loading scored data… (may take a moment on first run)")
df = load_scored()

top20 = df[df["is_top20"]]
bot80 = df[~df["is_top20"]]

N               = len(df)
cutoff          = float(top20["profit"].min())
mean_t20        = float(top20["profit"].mean())
med_t20         = float(top20["profit"].median())
mean_rest       = float(bot80["profit"].mean())
med_rest        = float(bot80["profit"].median())
pct_neg         = float((df["profit"] < 0).mean() * 100)
f3_in_top20     = int((top20["cost_collections"] <= -8500.0).sum())
f3_any          = int((df["cost_collections"] <= -8500.0).sum())
f3_pct          = round(f3_any / N * 100, 1)

prof            = df["profit"]
mean_sall_t20   = float(((top20["rev_discount"] / 0.024)).mean())
mean_sall_rest  = float(((bot80["rev_discount"] / 0.024)).mean())
mean_sall_all   = float(((df["rev_discount"] / 0.024)).mean())
med_sall_all    = float(((df["rev_discount"] / 0.024)).median())

f2_any          = int((df["cost_cancel_calls"] < 0).sum())
f2_pct          = round(f2_any / N * 100, 1)
f2_top20_reach  = int(((top20["cost_cancel_calls"] < 0)).sum())
f2_top20_share  = round(f2_top20_reach / f2_any * 100, 1) if f2_any > 0 else 0

TERM_COLS = [
    "rev_discount", "rev_interest", "cost_credit_loss", "cost_collections",
    "cost_servicing", "cost_cab", "cost_airline_credit", "cost_ent_credit",
    "cost_lounge", "cost_cancel_calls",
]

print()
print("=" * 70)
print("SECTION 8 COPY-PASTE BLOCK")
print("=" * 70)
print(f"""
## 8. GROUND-TRUTH DATASET FACTS (precomputed on all 500,000 rows — cite freely)

**Population & score distribution**
- N = {N:,}. Top-20% segment = {len(top20):,} customers.
- Top-20% cutoff score: **${cutoff:,.2f}** (p80 of profit).
- Profit distribution: mean ${prof.mean():.2f}, std ${prof.std():.2f}, min ${prof.min():.2f}, p1 ${prof.quantile(0.01):.2f},
  p25 ${prof.quantile(0.25):.2f}, median ${prof.median():.2f}, p75 ${prof.quantile(0.75):.2f}, p95 ${prof.quantile(0.95):.2f}, p99 ${prof.quantile(0.99):.2f},
  max ${prof.max():.2f}.
- **{pct_neg:.1f}% of customers are unprofitable** (profit < 0) under this framework.

**Segment economics (mean per customer)**
- Mean profit: top-20% **${mean_t20:,.2f}** vs rest **${mean_rest:,.2f}** (medians: ${med_t20:,.2f} vs ${med_rest:,.2f}).
- Billed business S_all: top-20% mean ${mean_sall_t20:,.0f} vs rest ${mean_sall_rest:,.0f} (overall mean ${mean_sall_all:,.0f}, median ${med_sall_all:,.0f}).
- Revolve f1: top-20% mean ${top20["rev_interest"].mean()/0.331:,.1f} vs rest ${bot80["rev_interest"].mean()/0.331:,.1f}. Risk f11: {float(top20["cost_credit_loss"].mean()/(-1.35) / (top20["rev_interest"].mean()/0.331+1e-9)):.3f} vs {float(bot80["cost_credit_loss"].mean()/(-1.35) / (bot80["rev_interest"].mean()/0.331+1e-9)):.3f} (top-20% revolvers are ~4× less risky).

**Mean P&L term values, top-20% vs rest**

| Term | Top-20% | Rest |
|---|---|---|""")
for col in TERM_COLS:
    t_mean = top20[col].mean()
    r_mean = bot80[col].mean()
    sign = "+" if t_mean >= 0 else ""
    sign_r = "+" if r_mean >= 0 else ""
    print(f"| {col} | {sign}${t_mean:,.2f} | {sign_r}${r_mean:,.2f} |")

print(f"""
**Attrition facts**
- f3 ≥ 1: {f3_any:,} customers ({f3_pct}%) — **{'zero' if f3_in_top20 == 0 else f3_in_top20}** reach top-20%.
- f2 ≥ 1: {f2_any:,} customers ({f2_pct}%) — only {f2_top20_share}% of them reach top-20%.

**Benefit-usage means (top-20% vs rest):** lounge f13 {float(top20["cost_lounge"].mean()/(-55)):.2f} vs {float(bot80["cost_lounge"].mean()/(-55)):.2f} · cab-months f15 {float(top20["cost_cab"].mean()/(-29)):.2f} vs {float(bot80["cost_cab"].mean()/(-29)):.2f} · airline credit f14 ${float(top20["cost_airline_credit"].mean()/(-1.9)):.1f} vs ${float(bot80["cost_airline_credit"].mean()/(-1.9)):.1f} · entertainment credit f16 ${float(top20["cost_ent_credit"].mean()/(-1.5)):.1f} vs ${float(bot80["cost_ent_credit"].mean()/(-1.5)):.1f}.
""")
print("=" * 70)
print()
print("EXPECTED dict for config.py (update EXPECTED only after reviewing above):")
print(f"""
EXPECTED = {{
    \"row_count\":          {N:,},
    \"top20_cutoff\":       {cutoff:.2f},
    \"mean_top20_profit\":  {mean_t20:.2f},
    \"mean_rest_profit\":   {mean_rest:.2f},
    \"pct_negative\":       {pct_neg:.1f},
    \"f3_in_top20\":        {f3_in_top20},
}}
""")
