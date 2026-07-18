"""
ProfitLens RAG — Gemini embeddings.

Same google-genai SDK and same GEMINI_API_KEY as the chat loop.
Batched (100/request), 429 exponential backoff, sqlite content-hash cache
so re-ingestion and index rebuilds never re-pay embedding calls.
All vectors are L2-normalized before they leave this module.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time

import numpy as np
from google.genai import types

from config import GEMINI_EMBED_MODEL, EMBED_DIM, RAG_STORE_DIR

logger = logging.getLogger(__name__)

# The free tier meters embed_content per *content*, not per request, at 100/min —
# so batches stay under that and the ingest loop paces itself between batches.
BATCH_SIZE = 50
RETRY_DELAYS = [2, 4, 8, 16, 32]
INTER_BATCH_PAUSE = 1.0
_CACHE_PATH = RAG_STORE_DIR / "emb_cache.sqlite"

_cache_hits = 0
_cache_misses = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════════════════════

def _conn() -> sqlite3.Connection:
    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
    return con


def _key(text: str, task_type: str) -> str:
    payload = f"{GEMINI_EMBED_MODEL}|{EMBED_DIM}|{task_type}|{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_stats() -> dict:
    return {"hits": _cache_hits, "misses": _cache_misses}


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


def _call_api(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a single batch, retrying on rate limits."""
    from llm import get_client  # imported lazily: shares the chat client factory

    client = get_client()
    config = types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=EMBED_DIM)
    for delay in RETRY_DELAYS + [None]:
        try:
            resp = client.models.embed_content(
                model=GEMINI_EMBED_MODEL, contents=texts, config=config)
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:
            err = str(e).lower()
            if delay and ("429" in err or "rate" in err or "resource" in err
                          or "unavailable" in err or "503" in err):
                # Honour the server's own retryDelay when it sends one.
                m = re.search(r"retry in ([\d.]+)s", err)
                wait = max(delay, float(m.group(1)) + 1 if m else 0)
                logger.warning("Embedding rate-limited; retrying in %.0fs", wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Return an (n, EMBED_DIM) float32 array of L2-normalized embeddings."""
    global _cache_hits, _cache_misses
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    keys = [_key(t, task_type) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)

    con = _conn()
    try:
        for i, k in enumerate(keys):
            row = con.execute("SELECT vec FROM emb WHERE key=?", (k,)).fetchone()
            if row is not None:
                out[i] = np.frombuffer(row[0], dtype="float32")
                _cache_hits += 1

        todo = [i for i, v in enumerate(out) if v is None]
        _cache_misses += len(todo)

        for start in range(0, len(todo), BATCH_SIZE):
            if start:
                time.sleep(INTER_BATCH_PAUSE)
            idxs = todo[start:start + BATCH_SIZE]
            vecs = _call_api([texts[i] for i in idxs], task_type)
            for i, vec in zip(idxs, vecs):
                arr = np.asarray(vec, dtype="float32")
                out[i] = arr
                con.execute("INSERT OR REPLACE INTO emb VALUES (?,?)",
                            (keys[i], arr.tobytes()))
            con.commit()
    finally:
        con.close()

    return _normalize(np.vstack([v for v in out]))


def embed_query(query: str) -> np.ndarray:
    """Embed a single query with RETRIEVAL_QUERY task type."""
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
