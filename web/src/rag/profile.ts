/**
 * Profile-doc treatment for uploaded tables, ported from profitlens/rag/profile_docs.py:
 * schema + per-column statistics + a short preview — never row-by-row.
 */
import { fmtG } from '../data/numfmt';
import { makeChunk } from './chunker';
import type { Chunk } from './types';

export type Cell = string | number | boolean | null;
export interface Table {
  columns: string[];
  rows: Cell[][];
}

export const STATS_ONLY_ROWS = 50_000;
export const PREVIEW_ROWS = 20;

export { fmtG };

const commas = (n: number) => n.toLocaleString('en-US');

export interface ColumnInfo {
  name: string;
  dtype: 'int64' | 'float64' | 'bool' | 'object';
  values: (number | string | boolean)[]; // non-missing
  total: number;
}

export function inferColumns(t: Table): ColumnInfo[] {
  return t.columns.map((name, j) => {
    const raw = t.rows.map((r) => r[j] ?? null).map((v) => (typeof v === 'string' && v.trim() === '' ? null : v));
    const present = raw.filter((v): v is Exclude<Cell, null> => v !== null);
    const nums = present.map((v) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN));
    let dtype: ColumnInfo['dtype'] = 'object';
    let values: ColumnInfo['values'] = present;
    if (present.length > 0 && nums.every(Number.isFinite)) {
      dtype = present.length === raw.length && nums.every(Number.isInteger) ? 'int64' : 'float64';
      values = nums;
    } else if (present.length > 0 && present.every((v) => typeof v === 'boolean' || (typeof v === 'string' && /^(true|false)$/i.test(v)))) {
      dtype = 'bool';
    } else if (present.length > 0) {
      values = present.map(String);
    }
    return { name, dtype, values, total: raw.length };
  });
}

function quantile(sorted: number[], q: number): number {
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function numericCard(col: ColumnInfo, name: string, desc: string): string {
  const v = (col.values as number[]).slice().sort((a, b) => a - b);
  if (v.length === 0) return `${name} — ${desc}. All values missing (${commas(col.total)} rows).`;
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const std = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : NaN;
  const q = (p: number) => quantile(v, p);
  return (
    `${name} — ${desc}.\n` +
    `Count: ${commas(v.length)} non-missing of ${commas(col.total)} rows ` +
    `(missing ${(((col.total - v.length) / col.total) * 100).toFixed(2)}%).\n` +
    `mean=${fmtG(mean)}, median=${fmtG(q(0.5))}, p5=${fmtG(q(0.05))}, p25=${fmtG(q(0.25))}, ` +
    `p75=${fmtG(q(0.75))}, p95=${fmtG(q(0.95))}, min=${fmtG(v[0])}, max=${fmtG(v[v.length - 1])}, std=${fmtG(std)}.`
  );
}

function markdownTable(columns: string[], rows: Cell[][]): string {
  const esc = (c: Cell) => String(c ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
  return [
    `| ${columns.map(esc).join(' | ')} |`,
    `|${columns.map(() => ' --- ').join('|')}|`,
    ...rows.map((r) => `| ${columns.map((_, j) => esc(r[j])).join(' | ')} |`),
  ].join('\n');
}

export function tabularProfileDocs(t: Table, source: string, label: string): Chunk[] {
  const chunks: Chunk[] = [];
  const cols = inferColumns(t);
  const n = t.rows.length;
  const big = n > STATS_ONLY_ROWS;

  const schema = cols.map((c) => `- ${c.name} (${c.dtype}), missing ${(((c.total - c.values.length) / (c.total || 1)) * 100).toFixed(1)}%`).join('\n');
  let overview = `Uploaded table \`${label}\` — schema overview.\n${commas(n)} rows × ${cols.length} columns.\nColumns:\n${schema}`;
  if (big) overview += `\n(Table exceeds ${commas(STATS_ONLY_ROWS)} rows — summary statistics only, no row preview.)`;
  chunks.push(makeChunk(overview, source, 'user', `${label}: schema`));

  for (const c of cols) {
    let text: string;
    if (c.dtype === 'int64' || c.dtype === 'float64') {
      text = numericCard(c, `\`${label}\`.${c.name}`, `numeric column (${c.dtype})`);
    } else {
      const counts = new Map<string, number>();
      for (const v of c.values) counts.set(String(v), (counts.get(String(v)) ?? 0) + 1);
      const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
      text =
        `\`${label}\`.${c.name} — non-numeric column (${c.dtype}).\n` +
        `${commas(c.values.length)} non-missing of ${commas(c.total)} rows ` +
        `(missing ${(((c.total - c.values.length) / (c.total || 1)) * 100).toFixed(2)}%), ` +
        `${commas(counts.size)} distinct values.\nMost frequent values:\n` +
        top.map(([k, v]) => `- ${k}: ${commas(v)}`).join('\n');
    }
    chunks.push(makeChunk(text, source, 'user', `${label}: column ${c.name}`));
  }

  if (!big && n > 0) {
    const k = Math.min(PREVIEW_ROWS, n);
    chunks.push(makeChunk(
      `Uploaded table \`${label}\` — first ${k} rows:\n\n${markdownTable(t.columns, t.rows.slice(0, k))}`,
      source, 'user', `${label}: preview rows 1-${k}`));
  }
  return chunks;
}

/** RFC-4180 CSV → table (first row = header). */
export function parseCsv(text: string): Table {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      rows.push(row); row = [];
    } else field += ch;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const clean = rows.filter((r) => !(r.length === 1 && r[0] === ''));
  if (clean.length === 0) return { columns: [], rows: [] };
  const [head, ...body] = clean;
  return { columns: head.map((h, i) => h.trim() || `column_${i + 1}`), rows: body };
}
