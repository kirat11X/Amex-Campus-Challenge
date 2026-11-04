/**
 * The customer data, served as static shards.
 *
 * Only the 23 raw features (bit-exact) and two rank counts per customer are stored;
 * the P&L terms and profit are recomputed here with the same frozen formula as
 * profitlens/data_engine.py, so a lookup downloads one ~230 KB shard, not 500K rows.
 */
import { round } from './numfmt';
import type { ColumnEnc, CustomerProfile, Meta, ToolsStatic } from './types';

export type ByteSource = (path: string) => Promise<ArrayBuffer>;

export const FEATURES = Array.from({ length: 23 }, (_, i) => `f${i + 1}`);
const TOP20_PCT = 0.8;

export { round };

export async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const b = new Uint8Array(buf);
  // Some static hosts transparently gunzip; accept both.
  if (!(b.length > 2 && b[0] === 0x1f && b[1] === 0x8b)) return buf;
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

// ── shard decoding ─────────────────────────────────────────────────────────

const SIZE: Record<ColumnEnc['kind'], number> = { i8: 1, i16: 2, i32: 4, f64: 8 };

export interface Shard {
  rows: number;
  features: Float64Array[]; // 23 columns, NaN = no recorded activity
  less: Uint32Array;
  eq: Uint32Array;
}

export function decodeShard(raw: ArrayBuffer, columns: ColumnEnc[], rows: number): Shard {
  let off = 0;
  const take = (bytes: number) => {
    const part = raw.slice(off, off + bytes); // copy => correctly aligned for typed arrays
    off += bytes;
    return part;
  };
  const features = columns.map((c) => {
    const part = take(rows * SIZE[c.kind]);
    if (c.kind === 'f64') return new Float64Array(part);
    const ints = c.kind === 'i8' ? new Int8Array(part) : c.kind === 'i16' ? new Int16Array(part) : new Int32Array(part);
    const sentinel = c.kind === 'i8' ? -128 : c.kind === 'i16' ? -32768 : -2147483648;
    const out = new Float64Array(rows);
    for (let i = 0; i < rows; i++) out[i] = ints[i] === sentinel ? NaN : ints[i] / c.scale;
    return out;
  });
  const less = new Uint32Array(take(rows * 4));
  const eq = new Uint32Array(take(rows * 4));
  if (off !== raw.byteLength) throw new Error(`shard size mismatch: read ${off} of ${raw.byteLength} bytes`);
  return { rows, features, less, eq };
}

// ── scoring (mirror of data_engine._compute_scores) ────────────────────────

export interface Scored {
  terms: Record<string, number>;
  profit: number;
}

/** `f` = the 23 raw features in order, NaN for missing. Missing counts as 0. */
export function scoreRow(f: ArrayLike<number>, W: Record<string, number>, termColumns: string[]): Scored {
  const z = (i: number) => (Number.isNaN(f[i - 1]) ? 0 : f[i - 1]);
  const sAll = z(6) + z(7) + z(8) + z(9) + z(10);
  const terms: Record<string, number> = {
    rev_discount: W.ic * sAll,
    rev_interest: W.nii * z(1),
    cost_lounge: -W.lng * z(13),
    cost_cab: -W.cab * z(15),
    cost_airline_credit: -W.b14 * z(14),
    cost_ent_credit: -W.b16 * z(16),
    cost_credit_loss: -W.rs * z(11) * z(1),
    cost_cancel_calls: -W.a2 * z(2),
    cost_collections: -W.a3 * z(3),
    cost_servicing: W.w12 * z(12),
  };
  let profit = 0;
  for (const c of termColumns) profit += terms[c];
  return { terms, profit };
}

export const pctRank = (less: number, eq: number, n: number) => (2 * less + eq + 1) / 2 / n;

// ── full table (only loaded for ad-hoc queries / custom histograms) ────────

export class FullTable {
  readonly n: number;
  private cache = new Map<string, Float64Array>();

  constructor(
    private meta: Meta,
    private features: Float64Array[],
    private less: Uint32Array,
    private eq: Uint32Array,
  ) {
    this.n = meta.customers.rows;
  }

  columns(): string[] {
    return ['id', ...FEATURES, ...this.meta.term_columns, 'profit', 'pct_rank', 'is_top20'];
  }

  has(name: string): boolean {
    return this.columns().includes(name);
  }

  col(name: string): Float64Array {
    const hit = this.cache.get(name);
    if (hit) return hit;
    let out: Float64Array;
    const fi = FEATURES.indexOf(name);
    if (fi >= 0) {
      out = this.features[fi];
    } else if (name === 'id') {
      out = Float64Array.from({ length: this.n }, (_, i) => i);
    } else if (name === 'pct_rank' || name === 'is_top20') {
      const pct = new Float64Array(this.n);
      const top = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        pct[i] = pctRank(this.less[i], this.eq[i], this.n);
        top[i] = pct[i] >= TOP20_PCT ? 1 : 0;
      }
      this.cache.set('pct_rank', pct);
      this.cache.set('is_top20', top);
      out = name === 'pct_rank' ? pct : top;
    } else if (name === 'profit' || this.meta.term_columns.includes(name)) {
      const tc = this.meta.term_columns;
      const arrays = Object.fromEntries([...tc, 'profit'].map((c) => [c, new Float64Array(this.n)]));
      const row = new Float64Array(23);
      for (let i = 0; i < this.n; i++) {
        for (let j = 0; j < 23; j++) row[j] = this.features[j][i];
        const s = scoreRow(row, this.meta.weights, tc);
        for (const c of tc) arrays[c][i] = s.terms[c];
        arrays.profit[i] = s.profit;
      }
      for (const [c, a] of Object.entries(arrays)) this.cache.set(c, a);
      out = arrays[name];
    } else {
      throw new Error(`unknown column ${name}`);
    }
    this.cache.set(name, out);
    return out;
  }
}

// ── the dataset facade ─────────────────────────────────────────────────────

export class Dataset {
  private shards = new Map<number, Promise<Shard>>();
  private full: Promise<FullTable> | null = null;

  constructor(
    readonly meta: Meta,
    readonly staticTools: ToolsStatic,
    private bytes: ByteSource,
  ) {}

  static async load(bytes: ByteSource): Promise<Dataset> {
    const json = async <T>(p: string) => JSON.parse(new TextDecoder().decode(await bytes(p))) as T;
    const [meta, staticTools] = await Promise.all([json<Meta>('data/meta.json'), json<ToolsStatic>('data/tools_static.json')]);
    return new Dataset(meta, staticTools, bytes);
  }

  get rows(): number {
    return this.meta.customers.rows;
  }

  isLoadedFully(): boolean {
    return this.full !== null;
  }

  private shard(index: number): Promise<Shard> {
    let p = this.shards.get(index);
    if (!p) {
      const c = this.meta.customers;
      const rows = Math.min(c.shard_rows, c.rows - index * c.shard_rows);
      p = this.bytes(`data/customers/${String(index).padStart(3, '0')}.bin`)
        .then(gunzip)
        .then((raw) => decodeShard(raw, c.columns, rows));
      p.catch(() => this.shards.delete(index));
      this.shards.set(index, p);
      // Keep a handful of decoded shards, not all 100 (~1 MB each decoded).
      if (this.shards.size > 6) this.shards.delete(this.shards.keys().next().value as number);
    }
    return p;
  }

  async customer(id: number): Promise<CustomerProfile | null> {
    const { shard_rows } = this.meta.customers;
    if (!Number.isInteger(id) || id < this.meta.id_min || id > this.meta.id_max) return null;
    const sh = await this.shard(Math.floor(id / shard_rows));
    const i = id % shard_rows;
    const f = sh.features.map((c) => c[i]);
    const { terms, profit } = scoreRow(f, this.meta.weights, this.meta.term_columns);
    const n = this.rows;
    const pct = pctRank(sh.less[i], sh.eq[i], n);
    return {
      customer_id: id,
      raw_features: Object.fromEntries(FEATURES.map((name, j) => [name, Number.isNaN(f[j]) ? null : round(f[j], 4)])),
      terms: Object.fromEntries(this.meta.term_columns.map((c) => [c, round(terms[c], 4)])),
      profit: round(profit, 4),
      percentile_rank: round(pct, 6),
      rank_out_of_500k: n - sh.less[i],
      is_top20: pct >= TOP20_PCT,
    };
  }

  /** Download every shard (~23 MB, cached by the browser) for queries the precomputed tables can't answer. */
  fullTable(onProgress?: (done: number, total: number) => void): Promise<FullTable> {
    if (!this.full) {
      this.full = this.loadFull(onProgress);
      this.full.catch(() => (this.full = null));
    }
    return this.full;
  }

  private async loadFull(onProgress?: (done: number, total: number) => void): Promise<FullTable> {
    const c = this.meta.customers;
    const feats = FEATURES.map(() => new Float64Array(c.rows));
    const less = new Uint32Array(c.rows);
    const eq = new Uint32Array(c.rows);
    let done = 0;
    let next = 0;
    const worker = async () => {
      for (;;) {
        const s = next++;
        if (s >= c.n_shards) return;
        const sh = await this.shard(s);
        const at = s * c.shard_rows;
        sh.features.forEach((col, j) => feats[j].set(col, at));
        less.set(sh.less, at);
        eq.set(sh.eq, at);
        onProgress?.(++done, c.n_shards);
      }
    };
    await Promise.all(Array.from({ length: 6 }, worker));
    this.shards.clear();
    return new FullTable(this.meta, feats, less, eq);
  }
}
