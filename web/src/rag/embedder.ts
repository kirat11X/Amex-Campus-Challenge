import type { EmbedConfig } from '../data/types';

export type EmbedStatus =
  | { state: 'idle' }
  | { state: 'loading'; file?: string; progress?: number }
  | { state: 'ready' }
  | { state: 'error'; message: string };

/**
 * Client for the embedding worker. The model (~34 MB, cached by the browser) is fetched
 * on first use; document vectors for the core corpus ship pre-computed, so only questions
 * and uploaded files go through here.
 */
export class Embedder {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: Float32Array[]) => void; reject: (e: Error) => void }>();
  private ready: Promise<void> | null = null;
  status: EmbedStatus = { state: 'idle' };
  private listeners = new Set<(s: EmbedStatus) => void>();

  constructor(private cfg: EmbedConfig) {}

  onStatus(fn: (s: EmbedStatus) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private set(s: EmbedStatus) {
    this.status = s;
    this.listeners.forEach((fn) => fn(s));
  }

  private call(msg: Record<string, unknown>): Promise<Float32Array[]> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ id, ...msg });
    });
  }

  /** Start loading the model (idempotent). */
  init(): Promise<void> {
    if (this.ready) return this.ready;
    this.worker = new Worker(new URL('./embed.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress') {
        const p = m.progress ?? {};
        if (p.status === 'progress' || p.status === 'download') this.set({ state: 'loading', file: p.file, progress: p.progress });
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.ok) p.resolve(m.vectors ?? []);
      else p.reject(new Error(m.error));
    };
    this.set({ state: 'loading' });
    const ortBase = new URL(import.meta.env.BASE_URL, location.origin).href + 'ort/';
    this.ready = this.call({ type: 'init', model: this.cfg.model, dtype: this.cfg.dtype, ortBase }).then(
      () => this.set({ state: 'ready' }),
      (e) => {
        this.ready = null;
        this.set({ state: 'error', message: e.message });
        throw e;
      },
    );
    return this.ready;
  }

  async embed(texts: string[], kind: 'document' | 'query'): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    await this.init();
    const prefix = kind === 'query' ? this.cfg.query_prefix : '';
    const out: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += 16) {
      out.push(...(await this.call({ type: 'embed', texts: texts.slice(i, i + 16).map((t) => prefix + t), pooling: this.cfg.pooling })));
    }
    return out;
  }
}
