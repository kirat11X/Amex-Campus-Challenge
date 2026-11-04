import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ByteSource } from '../src/data/dataset';

const PUBLIC = join(__dirname, '..', 'public');

/** Serve the static files straight from public/ (what the browser would fetch). */
export const diskBytes: ByteSource = async (path) => {
  const buf = await readFile(join(PUBLIC, path));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
};

/** Deep compare with numeric tolerance (abs 1e-6 + rel 1e-9, or `tol`). */
export function approx(a: unknown, b: unknown, tol = 1e-6, path = '$'): string | null {
  if (typeof a === 'number' && typeof b === 'number') {
    return Math.abs(a - b) <= tol + 1e-9 * Math.abs(b) ? null : `${path}: ${a} != ${b}`;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path}: length ${a.length} != ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const r = approx(a[i], b[i], tol, `${path}[${i}]`);
      if (r) return r;
    }
    return null;
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a).sort();
    const kb = Object.keys(b).sort();
    if (ka.join() !== kb.join()) return `${path}: keys ${ka} != ${kb}`;
    for (const k of ka) {
      const r = approx((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], tol, `${path}.${k}`);
      if (r) return r;
    }
    return null;
  }
  return a === b ? null : `${path}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`;
}
