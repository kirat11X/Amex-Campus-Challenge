"""
ProfitLens RAG — FAISS vector store.

Cosine similarity via IndexFlatIP over L2-normalized vectors, with a JSONL
metadata sidecar. No external DB, no server. Deletion rebuilds the index from
the vectors FAISS already holds (reconstruct_n), so no re-embedding is needed.
"""
from __future__ import annotations

import json
import threading

import faiss
import numpy as np

from config import RAG_STORE_DIR, EMBED_DIM


class VectorStore:
    """One named corpus: `<name>.index` + `<name>_meta.jsonl` in rag_store/."""

    def __init__(self, name: str, dim: int = EMBED_DIM):
        self.name = name
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metas: list[dict] = []
        self._lock = threading.Lock()

    # ── paths ────────────────────────────────────────────────────────────
    @property
    def index_path(self):
        return RAG_STORE_DIR / f"{self.name}.index"

    @property
    def meta_path(self):
        return RAG_STORE_DIR / f"{self.name}_meta.jsonl"

    # ── size ─────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.metas)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for m in self.metas:
            if m["source"] not in seen:
                seen.append(m["source"])
        return seen

    # ── mutation ─────────────────────────────────────────────────────────
    def add(self, vectors: np.ndarray, chunks: list[dict]) -> int:
        """Add vectors + their chunk metadata. Skips chunk_ids already present."""
        if len(chunks) == 0:
            return 0
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[0] != len(chunks):
            raise ValueError("vector/chunk count mismatch")
        with self._lock:
            known = {m["chunk_id"] for m in self.metas}
            keep = [i for i, c in enumerate(chunks) if c["chunk_id"] not in known]
            if not keep:
                return 0
            self.index.add(vectors[keep])
            self.metas.extend(chunks[i] for i in keep)
            return len(keep)

    def delete_by_source(self, source: str) -> int:
        """Drop every chunk from `source`, rebuilding the index in place."""
        with self._lock:
            keep = [i for i, m in enumerate(self.metas) if m["source"] != source]
            removed = len(self.metas) - len(keep)
            if removed == 0:
                return 0
            vectors = (self.index.reconstruct_n(0, self.index.ntotal)
                       if self.index.ntotal else np.zeros((0, self.dim), dtype="float32"))
            new_index = faiss.IndexFlatIP(self.dim)
            if keep:
                new_index.add(np.ascontiguousarray(vectors[keep], dtype="float32"))
            self.index = new_index
            self.metas = [self.metas[i] for i in keep]
            return removed

    def clear(self):
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metas = []

    # ── search ───────────────────────────────────────────────────────────
    def search(self, qvec: np.ndarray, k: int = 5, floor: float = 0.0) -> list[dict]:
        """Top-k chunks above the cosine floor, each with its score."""
        if self.index.ntotal == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < floor:
                continue
            meta = self.metas[int(idx)]
            hits.append({**meta, "score": round(float(score), 4)})
        return hits

    # ── persistence ──────────────────────────────────────────────────────
    def save(self):
        RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            for m in self.metas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self) -> bool:
        """Load from disk. Returns False (and stays empty) if absent or stale."""
        if not (self.index_path.exists() and self.meta_path.exists()):
            return False
        try:
            index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                metas = [json.loads(line) for line in fh if line.strip()]
            if index.d != self.dim or index.ntotal != len(metas):
                return False
            self.index, self.metas = index, metas
            return True
        except Exception:
            return False
