"""
Amex Campus Challenge 2026 - Round 1
Baseline Cardmember Profitability Framework
--------------------------------------------
Profit(CM) = Revenue - Cost, dollarized per component, one weight per lever.
Tune WEIGHTS between submissions; everything else stays fixed.
"""

import pandas as pd
import numpy as np

DATA = "/mnt/user-data/uploads/6a3eb196bc7a3_campus_challenge_r1_data.csv"
OUT  = "/home/claude/submission_v1.csv"

# ----------------------------------------------------------------------
# WEIGHTS — the only thing you tune between submissions (10 max, spend wisely)
# ----------------------------------------------------------------------
W = {
    # -------- REVENUE --------
    "interchange_rate":   0.025,   # ~2.5% discount revenue on on-card spend (f5)
    "nii_margin":         0.12,    # net interest margin on avg revolve balance (f1)
    "fee_per_card":       625.0,   # annual fee midpoint x active charge cards (f20)
    "fee_per_supp":       100.0,   # supplementary account fee (f19 - 1)

    # -------- COST --------
    "point_cost":         0.012,   # $ per CC point (issuer cost, 1-2c face)
    "mult_5x_share":      True,    # estimate 5x share from industry mix (f6,f9)
    "lounge_cost":        40.0,    # $ per lounge visit (f13)
    "cab_cost_per_month": 15.0,    # $15/mo cab credit (f15 = months used)
    "airline_credit":     1.0,     # f14 already in $ used
    "ent_credit":         1.0,     # f16 already in $ used
    "risk_exposure_revolve": 1.0,  # expected loss = f11 x (revolve + k x monthly spend)
    "risk_exposure_spend":   1.0,  # 1 month of spend at risk for charge card
    "servicing_per_login":   0.0,  # negligible, keep 0

    # -------- RETENTION HAIRCUT --------
    "attrition_cancel":   0.25,    # profit haircut if cancellation call (f2)
    "attrition_collect":  0.15,    # extra haircut if collections-driven call (f3)
}

# ----------------------------------------------------------------------
df = pd.read_csv(DATA)

# Missing = no activity for activity features (safe baseline assumption)
f = df.fillna(0.0)

# --- Revenue ---
interchange = W["interchange_rate"] * f["f5"]
nii         = W["nii_margin"] * f["f1"]
fees        = W["fee_per_card"] * f["f20"].clip(lower=1) \
            + W["fee_per_supp"] * (f["f19"] - 1).clip(lower=0)
revenue = interchange + nii + fees

# --- Rewards cost ---
# f6-f10 are industry-level spends (wallet view), weakly correlated with on-card
# spend f5. Use them only to estimate the CM's 5x-category share, then apply
# that mix to on-card spend f5 to get points earned.
if W["mult_5x_share"]:
    industry_total = f[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1)
    share_5x = np.where(industry_total > 0,
                        (f["f6"] + f["f9"]) / industry_total.replace(0, np.nan),
                        0.0)
    share_5x = np.nan_to_num(share_5x)
else:
    share_5x = 0.0
points_earned = f["f5"] * (5 * share_5x + 1 * (1 - share_5x))
rewards_cost  = W["point_cost"] * points_earned

# --- Benefit costs (real $ out the door) ---
benefit_cost = (W["lounge_cost"]        * f["f13"]
              + W["cab_cost_per_month"] * f["f15"]
              + W["airline_credit"]     * f["f14"]
              + W["ent_credit"]         * f["f16"])

# --- Credit risk cost: PD proxy (f11) x exposure ---
exposure  = (W["risk_exposure_revolve"] * f["f1"]
           + W["risk_exposure_spend"]   * f["f5"] / 12 * 12)  # f5 is annual; 1yr view
risk_cost = f["f11"] * exposure

# --- Servicing ---
servicing = W["servicing_per_login"] * f["f12"]

cost = rewards_cost + benefit_cost + risk_cost + servicing

# --- Profit with retention haircut ---
profit = revenue - cost
retention = 1.0 - W["attrition_cancel"] * f["f2"] - W["attrition_collect"] * f["f3"]
profit = profit * retention.clip(lower=0.4)

# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
comp = pd.DataFrame({
    "interchange": interchange, "nii": nii, "fees": fees,
    "rewards_cost": -rewards_cost, "benefit_cost": -benefit_cost,
    "risk_cost": -risk_cost, "profit": profit,
})
top = profit.rank(pct=True) >= 0.8
print("=== Component means: top 20% vs rest ===")
print(comp.groupby(top).mean().T.rename(columns={True: "top20", False: "rest"}).round(1))
print("\nProfit percentiles:")
print(profit.describe(percentiles=[.1, .25, .5, .75, .8, .9, .99]).round(1))

# ----------------------------------------------------------------------
# Submission (match template: ID, Prediction)
# ----------------------------------------------------------------------
sub = pd.DataFrame({"ID": df["id"], "Prediction": profit})
sub.to_csv(OUT, index=False)
print(f"\nSaved {OUT}  shape={sub.shape}")
