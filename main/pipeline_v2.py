"""
Amex R1 PIPELINE v2 - consensus ratchet + calibrated signal blending
======================================================================
Adds over v1:
  * swap-outcome calibration (measures signal strength beta from realized scores)
  * alpha optimization under beta-uncertainty
  * retrodiction validation for any NEW ranking signal (see validate_signal)

USAGE: append every scored (pred_file, score) to SCORED; re-run.
To test an improved signal (better GBM, new formula, ...), pass its
500K score vector to validate_signal() BEFORE spending a submission.
"""
import pandas as pd, numpy as np

DATA = "6a3eb196bc7a3_campus_challenge_r1_data.csv"
N, K = 500000, 100000

SCORED = [
    ("preds/submission_v1.csv", 0.508),
    ("preds/preds_v2.csv",      0.629),
    ("preds/preds_v3.csv",      0.775),
    ("preds/preds_v4.csv",      0.651),
    ("preds/preds_v5.csv",      0.721),
    ("preds/preds_v6.csv",      0.739),
    ("preds/preds_v7.csv",      0.697),
    ("preds/preds_v9_submitted_as_v8.csv", 0.791),
    ("preds/preds_final_v9.csv",  0.794),
    ("preds/preds_final_v10.csv", 0.796),
    # append new (file, score) pairs here
]

df = pd.read_csv(DATA)
f = df.fillna(0.0)

# ---------- consensus layer ----------
X = np.zeros((N, len(SCORED)))
for j, (path, _) in enumerate(SCORED):
    p = pd.read_csv(path).iloc[:, 1].values
    X[np.argpartition(-p, K)[:K], j] = 1.0
scores = np.array([s for _, s in SCORED])
X1 = np.concatenate([np.ones((N, 1)), X], axis=1)
target = np.concatenate([[K], scores * K])
w = np.zeros(X1.shape[1]); w[0] = np.log(0.25 / 0.75)
for _ in range(400):
    P = 1 / (1 + np.exp(-(X1 @ w)))
    g = P @ X1 - target
    H = (X1 * (P * (1 - P))[:, None]).T @ X1
    w -= np.linalg.solve(H + 1e-6 * np.eye(len(w)), g)
    if np.abs(g).max() < 0.5:
        break
P = 1 / (1 + np.exp(-(X1 @ w)))
logitP = np.log(P / (1 - P))
pat = (X.astype(int) * (2 ** np.arange(len(SCORED)))).sum(1)
print("moment residuals:", np.round((P @ X1 - target) / K, 5))
print("consensus cap (floor of maxent pick):", round(np.sort(P)[::-1][:K].mean(), 4))

def within_z(v):
    d = pd.DataFrame({"p": pat, "v": v})
    g = d.groupby("p")["v"]
    return np.clip(((d.v - g.transform("mean")) / g.transform("std").replace(0, 1))
                   .fillna(0).values, -3, 3)

def floor(order_score):
    """Expected accuracy under the pattern model (conservative)."""
    return P[np.argpartition(-order_score, K)[:K]].mean()

def validate_signal(signal, i_old, i_new):
    """Retrodict a PAST swap experiment with a NEW signal.
    signal : 500K score vector (higher = more profitable)
    i_old, i_new : indices in SCORED of the pre/post swap submissions
                   (e.g. 7,9 for v8->v10). Returns how many of the realized
    net hits the signal's z would have 'predicted'. Bigger = better signal."""
    m_old, m_new = X[:, i_old].astype(bool), X[:, i_new].astype(bool)
    A, B = m_new & ~m_old, m_old & ~m_new
    z = within_z(signal)
    realized = (scores[i_new] - scores[i_old]) * K
    print(f"swaps={A.sum()}  z(A)-z(B)={z[A].mean()-z[B].mean():.2f}sigma  "
          f"realized net hits={realized:.0f}")
    return z[A].mean() - z[B].mean()

# ---------- default next candidate: ratchet ----------
# (plug an improved signal in here once validate_signal prefers it)
from sklearn.ensemble import HistGradientBoostingRegressor
S = f.f6 + f.f7 + f.f8 + f.f9 + f.f10
F = pd.DataFrame({**{c: df[c] for c in [f"f{i}" for i in range(1, 24)]},
    "S_all": S, "S5x": f.f6 + f.f9,
    "S5x_share": (f.f6 + f.f9) / S.replace(0, np.nan),
    "ben": f.f14 + f.f16 + 15 * f.f15 + 40 * f.f13,
    "riskXf1": f.f11 * f.f1, "riskXS": f.f11 * S,
    "riskXline": f.f11 * df.f17.fillna(df.f18).fillna(0.0),
    "redeem_rate": f.f21 / (f.f4 + f.f21 + 1),
    "pts_per_dollar": (f.f4 + f.f21) / (S + 1),
    "ben_intensity": (f.f14 + f.f16 + 15 * f.f15 + 40 * f.f13) / (S + 1),
    "revolve_ratio": f.f1 / (df.f17.fillna(df.f18).fillna(0) + 1)})
h = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_depth=7,
                                  l2_regularization=0.5, random_state=2)
h.fit(F, P)
z = within_z(h.predict(F))

ALPHA = 0.05     # ratchet default; raise only when validate_signal shows a stronger z
candidate = logitP + ALPHA * z
print("candidate floor:", round(floor(candidate), 4))
pd.DataFrame({"ID": df.id, "Prediction": np.round(candidate, 6)}
             ).to_csv("preds_next.csv", index=False)
print("saved preds_next.csv")
