/** numpy-compatible helpers (linear-interpolated percentile, histogram, accurate sums). */

/** Neumaier-compensated sum: stays within an ulp or two of the exact sum over 500K values. */
export function sum(a: ArrayLike<number>): number {
  let s = 0;
  let c = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return s + c;
}

export const mean = (a: ArrayLike<number>) => (a.length ? sum(a) / a.length : NaN);

/** np.percentile(a, q) with method="linear". `sorted` must be ascending. */
export function percentileSorted(sorted: Float64Array, q: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const t = pos - lo;
  const a = sorted[lo];
  const b = sorted[hi];
  return t >= 0.5 ? b - (b - a) * (1 - t) : a + (b - a) * t;
}

export function dropNaN(a: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  let n = 0;
  for (let i = 0; i < a.length; i++) if (!Number.isNaN(a[i])) out[n++] = a[i];
  return out.slice(0, n);
}

/** np.histogram(values, bins) over [min, max] (the last bin is closed). */
export function histogram(values: Float64Array, bins: number): { counts: number[]; edges: number[] } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (values.length === 0) {
    lo = 0;
    hi = 1;
  }
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins);
  edges[bins] = hi;
  const counts = new Array<number>(bins).fill(0);
  const norm = bins / (hi - lo);
  for (const v of values) {
    let idx = Math.floor((v - lo) * norm);
    if (idx >= bins) idx = bins - 1;
    // numpy's float-error correction against the true edges
    if (v < edges[idx]) idx--;
    else if (v >= edges[idx + 1] && idx !== bins - 1) idx++;
    counts[idx]++;
  }
  return { counts, edges };
}
