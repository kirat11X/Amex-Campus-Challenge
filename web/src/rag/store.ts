import type { Chunk, Hit } from './types';

/**
 * In-memory cosine-similarity store (vectors are L2-normalised, so cosine = dot).
 * Mirrors profitlens/rag/store.py: add skips known chunk_ids, delete rebuilds in place.
 */
export class VectorStore {
  private vecs: Float32Array[] = [];
  private metas: Chunk[] = [];

  constructor(readonly dim: number) {}

  get size(): number {
    return this.metas.length;
  }

  get sources(): string[] {
    return [...new Set(this.metas.map((m) => m.source))];
  }

  all(): { chunks: Chunk[]; vectors: Float32Array[] } {
    return { chunks: [...this.metas], vectors: [...this.vecs] };
  }

  add(vectors: Float32Array[], chunks: Chunk[]): number {
    if (vectors.length !== chunks.length) throw new Error('vector/chunk count mismatch');
    const known = new Set(this.metas.map((m) => m.chunk_id));
    let added = 0;
    chunks.forEach((c, i) => {
      if (known.has(c.chunk_id)) return;
      if (vectors[i].length !== this.dim) throw new Error(`expected ${this.dim}-d vectors, got ${vectors[i].length}`);
      known.add(c.chunk_id);
      this.vecs.push(vectors[i]);
      this.metas.push(c);
      added++;
    });
    return added;
  }

  deleteBySource(source: string): number {
    const keep = this.metas.map((m) => m.source !== source);
    const removed = keep.filter((k) => !k).length;
    if (removed) {
      this.vecs = this.vecs.filter((_, i) => keep[i]);
      this.metas = this.metas.filter((_, i) => keep[i]);
    }
    return removed;
  }

  clear(): void {
    this.vecs = [];
    this.metas = [];
  }

  search(q: Float32Array, k: number, floor = 0): Hit[] {
    const scored: { i: number; s: number }[] = [];
    for (let i = 0; i < this.vecs.length; i++) {
      const v = this.vecs[i];
      let s = 0;
      for (let j = 0; j < v.length; j++) s += v[j] * q[j];
      if (s >= floor) scored.push({ i, s });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, Math.max(0, k)).map(({ i, s }) => ({ ...this.metas[i], score: Math.round(s * 1e4) / 1e4 }));
  }
}
