"""
ProfitLens RAG — local embeddings via Ollama.

Batched requests to Ollama's /api/embed, with a sqlite content-hash cache
so re-ingestion and index rebuilds never re-embed unchanged text.
All vectors are L2-normalized before they leave this module.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time

import numpy as np
import requests

from config import OLLAMA_HOST, EMBED_MODEL, EMBED_DIM, RAG_STORE_DIR

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
RETRY_DELAYS = [1, 2, 4]
# nomic-embed-text expects a task prefix on every input.
_PREFIX = {"RETRIEVAL_DOCUMENT": "search_document: ", "RETRIEVAL_QUERY": "search_query: "}
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
    payload = f"{EMBED_MODEL}|{EMBED_DIM}|{task_type}|{text}"
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
    """Embed a single batch via Ollama, retrying transient failures."""
    prefix = _PREFIX.get(task_type, "")
    payload = {"model": EMBED_MODEL, "input": [prefix + t for t in texts],
               "truncate": True}
    for delay in RETRY_DELAYS + [None]:
        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/embed", json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except requests.RequestException as e:
            if delay is None:
                raise RuntimeError(
                    f"Ollama embedding failed ({e}). Is Ollama running and "
                    f"`ollama pull {EMBED_MODEL}` done?") from e
            logger.warning("Embedding request failed; retrying in %ss", delay)
            time.sleep(delay)
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
