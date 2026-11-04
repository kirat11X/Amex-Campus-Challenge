#!/usr/bin/env python
"""
Export everything the static ProfitLens web app needs into web/public/data/.

Reuses the Streamlit app's own modules (read-only), so the web build computes
nothing differently from the Python one:

  meta.json            weights, dictionaries, stats pack, runtime-verification block,
                       xlsx integrity result, customer-shard manifest
  context.md           PROFITLENS_CONTEXT.md, verbatim (the system-prompt source)
  tools_static.json    exact tool outputs that need the full 500K rows to compute:
                       term_contribution_summary, explain_formula, top/bottom 50,
                       segment_compare and default get_distribution for every column
  core_chunks.json     the core knowledge-base chunks (same builders as rag/ingest.py)
  customers/NNN.bin    gzip'd 5,000-customer shards: the 23 raw features, bit-exact,
                       plus the tie-aware rank counts the browser needs for
                       rank / percentile / top-20% (profit itself is recomputed in JS)
  tests/golden/*.json  Python tool outputs the JS ports are tested against

Run (needs the ProfitLens venv and the scored data in the repo root):

    ../.venv/bin/python scripts/export_data.py            # from amex/web
    ../.venv/bin/python scripts/export_data.py --skip-xlsx  # skip the slow xlsx check
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WEB = Path(__file__).resolve().parent.parent
PROFITLENS = WEB.parent / "profitlens"
sys.path.insert(0, str(PROFITLENS))

import config as cfg                      # noqa: E402
import data_engine as de                  # noqa: E402
import tools                              # noqa: E402
from rag import ingest as ing             # noqa: E402
from rag.chunker import (chunk_markdown, chunk_python, make_chunk,   # noqa: E402
                         split_prose)
from rag.profile_docs import (_fmt, dataset_profile_docs, formula_output_docs,  # noqa: E402
                              tabular_profile_docs)

OUT = WEB / "public" / "data"
GOLDEN = WEB / "tests" / "golden"
SHARD_ROWS = 5000
FEATURES = [f"f{i}" for i in range(1, 24)]


# ── helpers ────────────────────────────────────────────────────────────────

def clean(o):
    """JSON-safe copy: numpy → python, NaN/inf → None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def dump(path: Path, obj, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, ensure_ascii=False, allow_nan=False, default=str,
                  indent=indent, separators=None if indent else (",", ":"))
    print(f"  {path.relative_to(WEB)}  ({path.stat().st_size / 1024:,.1f} KB)")


# ── customer shards ────────────────────────────────────────────────────────

def pick_encoding(x: np.ndarray) -> dict:
    """Smallest lossless encoding for a feature column.

    Integer-valued (after an optional power-of-ten scale) columns become i8/i16/i32
    with the type's minimum as the NaN sentinel; anything else stays float64.
    The scaled value must divide back to the *identical* double.
    """
    valid = x[~np.isnan(x)]
    for d in range(0, 10):
        scale = 10.0 ** d
        y = np.round(valid * scale)
        if not np.array_equal(y / scale, valid):
            continue
        peak = float(np.abs(y).max()) if len(y) else 0.0
        for kind, lim in (("i8", 127), ("i16", 32767), ("i32", 2_147_483_647)):
            if peak < lim:
                return {"kind": kind, "scale": int(scale)}
        break
    return {"kind": "f64", "scale": 1}


_DTYPE = {"i8": "<i1", "i16": "<i2", "i32": "<i4", "f64": "<f8"}


def encode_column(x: np.ndarray, enc: dict) -> bytes:
    if enc["kind"] == "f64":
        return x.astype("<f8").tobytes()
    dt = np.dtype(_DTYPE[enc["kind"]])
    sentinel = np.iinfo(dt).min
    y = np.where(np.isnan(x), sentinel, np.round(x * enc["scale"])).astype(dt)
    return y.tobytes()


def export_customers(df: pd.DataFrame) -> dict:
    n = len(df)
    assert df["id"].is_monotonic_increasing and df["id"].iloc[0] == 0 \
        and df["id"].iloc[-1] == n - 1, "shards assume ids 0..N-1 in order"

    encodings = {f: pick_encoding(df[f].to_numpy(float)) for f in FEATURES}

    # Tie-aware rank counts (all the browser needs for rank / pct_rank / top-20%).
    profit = df["profit"].to_numpy(float)
    sp = np.sort(profit)
    less = np.searchsorted(sp, profit, side="left").astype("<u4")
    eq = (np.searchsorted(sp, profit, side="right") - less).astype("<u4")
    # pandas rank(pct=True): average rank / N, with average rank = less + (eq+1)/2
    pct = (2.0 * less + eq + 1.0) / 2.0 / n
    assert np.array_equal(pct, df["pct_rank"].to_numpy(float)), "pct_rank derivation drifted"
    assert np.array_equal(pct >= 0.80, df["is_top20"].to_numpy(bool)), "top-20% derivation drifted"

    folder = OUT / "customers"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    total = 0
    n_shards = math.ceil(n / SHARD_ROWS)
    for s in range(n_shards):
        lo, hi = s * SHARD_ROWS, min(n, (s + 1) * SHARD_ROWS)
        parts = [encode_column(df[f].to_numpy(float)[lo:hi], encodings[f]) for f in FEATURES]
        parts += [less[lo:hi].tobytes(), eq[lo:hi].tobytes()]
        blob = gzip.compress(b"".join(parts), 9, mtime=0)
        (folder / f"{s:03d}.bin").write_bytes(blob)
        total += len(blob)
    print(f"  customers/  {n_shards} shards, {total / 1e6:,.1f} MB gzip'd")

    return {
        "shard_rows": SHARD_ROWS, "n_shards": n_shards, "rows": n,
        "columns": [{"name": f, **encodings[f]} for f in FEATURES],
        "bytes": total,
    }


# ── core knowledge-base chunks (mirrors rag.ingest.build_core, minus embedding) ─

def core_chunks(df: pd.DataFrame, verification: dict) -> list[dict]:
    chunks: list[dict] = []
    if cfg.TRIAL_PY.exists():
        chunks += chunk_python(cfg.TRIAL_PY.read_text(encoding="utf-8"), "trial.py")
    chunks += ing._feature_dictionary_chunks()
    chunks += dataset_profile_docs(df)
    chunks += formula_output_docs(df, verification)
    chunks += ing._framework_sheet_chunks()
    if cfg.CONTEXT_MD_PATH.exists():
        chunks += chunk_markdown(cfg.CONTEXT_MD_PATH.read_text(encoding="utf-8"),
                                 "PROFITLENS_CONTEXT.md", kind="context")
    if cfg.CHALLENGE_PDF.exists():
        for page_no, text in ing._pdf_pages(cfg.CHALLENGE_PDF.read_bytes()):
            for i, piece in enumerate(split_prose(text, target=1_200)):
                chunks.append(make_chunk(piece, cfg.CHALLENGE_PDF.name, "slide",
                                         f"p.{page_no}" if i == 0 else f"p.{page_no} ({i + 1})"))
    seen, unique = set(), []
    for c in chunks:
        if c["chunk_id"] not in seen and c["text"]:
            seen.add(c["chunk_id"])
            unique.append(c)
    return unique


# ── golden outputs for the JS tests ────────────────────────────────────────

def golden(df: pd.DataFrame):
    rng = np.random.default_rng(7)
    ids = {0, 1, 2, 8, 20, len(df) - 1}
    ids |= set(rng.choice(len(df), 160, replace=False).tolist())
    ids |= set(df.loc[df["cost_collections"] < 0, "id"].head(10).tolist())      # f3 >= 1
    ids |= set(df.sort_values("profit").head(5)["id"].tolist())                  # extremes
    ids |= set(df.sort_values("profit").tail(5)["id"].tolist())
    cut = df.loc[df["is_top20"], "profit"].min()
    ids |= set(df.loc[(df["profit"] - cut).abs().nsmallest(8).index, "id"].tolist())  # cutoff
    nan_heavy = df[FEATURES].isna().sum(axis=1).nlargest(5).index
    ids |= set(df.loc[nan_heavy, "id"].tolist())
    profiles = {int(i): tools.get_customer_profile(int(i)) for i in sorted(ids)}
    dump(GOLDEN / "profiles.json", profiles)

    queries = [
        ("f3 >= 1", "count"), ("f3 >= 1 and profit > 0", "count"),
        ("f3 >= 1", "top20_share"), ("f2 >= 1", "top20_share"),
        ("profit < 0", "count"), ("profit < 0", "mean_profit"),
        ("is_top20", "sum_profit"), ("is_top20 == True and f1 > 5000", "mean_profit"),
        ("f1 > 0 & f11 < 0.05", "mean_profit"), ("f13 >= 2 | f15 >= 10", "count"),
        ("not (f2 >= 1)", "count"), ("~(f12 > 20)", "count"),
        ("(f6 + f7) * 2 > 100000", "count"), ("f4 > 100000", "top20_share"),
        ("1 <= f19 < 3", "count"), ("f23 == 1", "count"), ("pct_rank >= 0.95", "count"),
        ("f1 > 0 and f1 < 1000 and f14 >= 0", "sum_profit"), ("cost_lounge < -55", "count"),
        ("f16 > 60", "mean_profit"), ("f5 > 1e4", "count"),
        ("id < 1000", "mean_profit"), ("f3 > 99", "count"),
        ("f1 > 1000000", "mean_profit"), ("f1 > 1000000", "top20_share"),
        ("f4 != f4", "count"),
    ]
    tool_cases = [{"tool": "query_segment", "args": {"filters": q, "agg": a},
                   "result": tools.query_segment(q, a)} for q, a in queries]
    for col, bins, clip in [("profit", 40, None), ("f1", 20, None), ("f13", 10, [0, 100]),
                            ("cost_collections", 40, [0.5, 99.5]), ("f6", 25, [5, 95]),
                            ("pct_rank", 40, None), ("f16", 40, [1, 99])]:
        args = {"column": col, "bins": bins}
        if clip:
            args["clip_percentiles"] = clip
        tool_cases.append({"tool": "get_distribution", "args": args,
                           "result": tools.get_distribution(**args)})
    for feat in ("f1", "f13", "profit", "cost_lounge", "pct_rank", "f4", "f23"):
        tool_cases.append({"tool": "segment_compare", "args": {"feature": feat},
                           "result": tools.segment_compare(feat)})
    for n, asc in ((5, False), (10, True), (3, False)):
        tool_cases.append({"tool": "get_top_customers", "args": {"n": n, "ascending": asc},
                           "result": tools.get_top_customers(n, asc)})
    dump(GOLDEN / "tools.json", tool_cases)
    golden_chunkers()


def golden_chunkers():
    """Python chunker / formatter outputs the TypeScript ports must reproduce."""
    import io

    def brief(chunks):
        return [{"kind": c["kind"], "location": c["location"], "text": c["text"]} for c in chunks]

    ctx = cfg.CONTEXT_MD_PATH.read_text(encoding="utf-8")
    trial = cfg.TRIAL_PY.read_text(encoding="utf-8")
    prose = ("The first paragraph is short.\n\n" + " ".join(
        f"Sentence number {i} talks about card profitability and revolving balances." for i in range(60))
        + "\n\nA closing paragraph. It has two sentences!")
    csv_text = ("id,amount,name,region\n1,10.5,alpha,north\n2,,beta,south\n3,7.25,alpha,north\n"
                "4,1200000,gamma,\n5,0.00012,beta,south\n6,-3,alpha,north\n")
    df = pd.read_csv(io.StringIO(csv_text))
    fmt_vals = [0, 1, 12, 123.456, 1234.5678, 12345.678, 0.5, 0.00012345, 123456789.0, -0.5, -1234.5,
                1e-5, 99999.0, 9999.5, 4354.85, 0.010, 19358.123]
    dump(GOLDEN / "chunkers.json", {
        "context_md": brief(chunk_markdown(ctx, "PROFITLENS_CONTEXT.md", kind="context")),
        "trial_py": brief(chunk_python(trial, "trial.py")),
        "prose": {"text": prose, "pieces": {str(t): split_prose(prose, target=t) for t in (300, 800, 1200)}},
        "fmt": [[v, _fmt(v)] for v in fmt_vals],
        "csv": {"text": csv_text, "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "chunks": brief(tabular_profile_docs(df, "sample.csv", "sample.csv"))},
    })


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xlsx", action="store_true",
                    help="skip the (slow) submission_FINAL.xlsx integrity check")
    ap.add_argument("--no-golden", action="store_true")
    args = ap.parse_args()

    print("loading scored data …")
    df = de.load_scored()
    tools.init_tools(df)

    print("verifying against submission_FINAL.xlsx …")
    if args.skip_xlsx:
        verification = {"ok": False, "message": "Submission check skipped at export time.",
                        "match_count": 0, "total": 0, "mismatches": 0}
    else:
        verification = de.verify_against_submission(df)
    print("  ", verification["message"])

    stats = de.build_stats_pack(df)
    runtime = de.compute_runtime_verification(df)

    print("writing files:")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": cfg.W,
        "equation": cfg.build_equation_string(),
        "term_columns": cfg.TERM_COLUMNS,
        "term_meanings": cfg.TERM_MEANINGS,
        "feature_dict": cfg.FEATURE_DICT,
        "valid_columns": tools.VALID_COLUMNS,
        "rag_sim_floor_python": cfg.RAG_SIM_FLOOR,
        "id_min": int(df["id"].min()), "id_max": int(df["id"].max()),
        "stats": stats,
        "verification": verification,
        "runtime": runtime,
        "customers": export_customers(df),
    }
    dump(OUT / "meta.json", meta, indent=1)

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg.CONTEXT_MD_PATH, OUT / "context.md")
    print(f"  public/data/context.md  ({cfg.CONTEXT_MD_PATH.stat().st_size / 1024:,.1f} KB)")

    static = {
        "term_contribution_summary": tools.term_contribution_summary(),
        "explain_formula": tools.explain_formula(),
        "top_customers": {"top": tools.get_top_customers(50),
                          "bottom": tools.get_top_customers(50, ascending=True)},
        "segment_compare": {c: tools.segment_compare(c) for c in tools.VALID_COLUMNS},
        "distributions": {c: tools.get_distribution(c) for c in tools.VALID_COLUMNS},
    }
    dump(OUT / "tools_static.json", static)

    chunks = core_chunks(df, verification)
    dump(OUT / "core_chunks.json", chunks)
    ref = cfg.RAG_STORE_DIR / "core_meta.jsonl"
    if ref.exists():
        old = {json.loads(l)["chunk_id"] for l in ref.read_text().splitlines() if l.strip()}
        new = {c["chunk_id"] for c in chunks}
        print(f"  core chunks: {len(chunks)} (Streamlit index: {len(old)}; "
              f"{'identical' if old == new else f'{len(old ^ new)} differ'})")

    if not args.no_golden:
        print("golden test data:")
        golden(df)
    print("done.")


if __name__ == "__main__":
    main()
