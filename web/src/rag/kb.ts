/**
 * The knowledge base: a "core" corpus (the project's own code, docs and data profiles —
 * shipped pre-embedded) and a "user" corpus (files digested in the browser and kept in
 * IndexedDB; nothing is uploaded anywhere).
 */
import type { ByteSource } from '../data/dataset';
import type { EmbedConfig } from '../data/types';
import type { Embedder } from './embedder';
import { parseUpload, MAX_UPLOAD_BYTES } from './ingest';
import { kvGet, kvSet } from './persist';
import { VectorStore } from './store';
import type { Chunk, Hit, KnowledgeBase } from './types';

export interface UserFile {
  filename: string;
  type: string;
  chunks: number;
  ingested_at: string;
  content_hash: string;
}

interface Saved {
  manifest: UserFile[];
  chunks: Chunk[];
  vectors: Float32Array[];
}

export interface IngestResult {
  ok: boolean;
  duplicate?: boolean;
  chunks?: number;
  message: string;
}

const sha = async (algo: 'SHA-1' | 'SHA-256', data: ArrayBuffer) =>
  [...new Uint8Array(await crypto.subtle.digest(algo, data))].map((b) => b.toString(16).padStart(2, '0')).join('');

export class Knowledge implements KnowledgeBase {
  manifest: UserFile[] = [];
  private listeners = new Set<() => void>();

  constructor(
    readonly core: VectorStore,
    readonly user: VectorStore,
    private embedder: Embedder,
    private cfg: EmbedConfig,
  ) {}

  static async load(bytes: ByteSource, embedder: Embedder, cfg: EmbedConfig): Promise<Knowledge> {
    const [chunkBytes, vecBytes] = await Promise.all([bytes('data/core_chunks.json'), bytes('data/core_vectors.bin')]);
    const chunks: Chunk[] = JSON.parse(new TextDecoder().decode(chunkBytes));
    const core = new VectorStore(cfg.dim);
    const flat = new Float32Array(vecBytes);
    const fresh = (await sha('SHA-1', chunkBytes)) === cfg.chunks_sha1;
    if (fresh && flat.length === chunks.length * cfg.dim) {
      core.add(chunks.map((_, i) => flat.slice(i * cfg.dim, (i + 1) * cfg.dim)), chunks);
    } else {
      // Stale pre-computed vectors (corpus changed without `npm run embed-core`): embed here instead.
      console.warn('core_vectors.bin does not match core_chunks.json — re-embedding in the browser');
      core.add(await embedder.embed(chunks.map((c) => c.text), 'document'), chunks);
    }
    const kb = new Knowledge(core, new VectorStore(cfg.dim), embedder, cfg);
    const saved = await kvGet<Saved>('user-corpus');
    if (saved && saved.chunks.length === saved.vectors.length && saved.vectors.every((v) => v.length === cfg.dim)) {
      kb.manifest = saved.manifest;
      kb.user.add(saved.vectors, saved.chunks);
    }
    return kb;
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private async persist() {
    const { chunks, vectors } = this.user.all();
    await kvSet('user-corpus', { manifest: this.manifest, chunks, vectors } satisfies Saved);
    this.listeners.forEach((fn) => fn());
  }

  userChunkCount(): number {
    return this.user.size;
  }

  async search(query: string, corpus: 'core' | 'user', k: number): Promise<Hit[]> {
    const store = corpus === 'core' ? this.core : this.user;
    if (store.size === 0) return [];
    const [q] = await this.embedder.embed([query], 'query');
    return store.search(q, k, this.cfg.sim_floor);
  }

  async ingestFile(name: string, data: ArrayBuffer): Promise<IngestResult> {
    if (data.byteLength > MAX_UPLOAD_BYTES) {
      return { ok: false, message: `${name} is ${(data.byteLength / 1e6).toFixed(1)} MB — the limit is 25 MB.` };
    }
    const hash = await sha('SHA-256', data);
    if (this.manifest.some((e) => e.content_hash === hash)) {
      return { ok: true, duplicate: true, chunks: 0, message: `${name} is already digested — skipped re-embedding.` };
    }
    let chunks: Chunk[];
    try {
      chunks = await parseUpload(name, data);
    } catch (e) {
      return { ok: false, message: `Could not read ${name}: ${e instanceof Error ? e.message : e}` };
    }
    if (!chunks.length) return { ok: false, message: `${name} contained no indexable text.` };

    let vectors: Float32Array[];
    try {
      // Embed before replacing, so a failed re-upload leaves the previous version intact.
      vectors = await this.embedder.embed(chunks.map((c) => c.text), 'document');
    } catch (e) {
      return { ok: false, message: `Embedding failed for ${name}: ${e instanceof Error ? e.message : e}` };
    }
    this.user.deleteBySource(name);
    this.manifest = this.manifest.filter((e) => e.filename !== name);
    const added = this.user.add(vectors, chunks);
    const now = new Date();
    const p = (n: number) => String(n).padStart(2, '0');
    this.manifest.push({
      filename: name,
      type: name.split('.').pop()!.toLowerCase(),
      chunks: added,
      ingested_at: `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())} ${p(now.getHours())}:${p(now.getMinutes())}`,
      content_hash: hash,
    });
    await this.persist();
    return { ok: true, duplicate: false, chunks: added, message: `✓ ${name} digested — ${added} chunks` };
  }

  async deleteFile(name: string): Promise<number> {
    const removed = this.user.deleteBySource(name);
    this.manifest = this.manifest.filter((e) => e.filename !== name);
    await this.persist();
    return removed;
  }

  async clearUser(): Promise<number> {
    const removed = this.user.size;
    this.user.clear();
    this.manifest = [];
    await this.persist();
    return removed;
  }
}
