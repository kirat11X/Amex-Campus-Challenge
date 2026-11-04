/**
 * Chunking, ported from profitlens/rag/chunker.py (same sizes, same split rules),
 * so uploaded files are chunked exactly as the Streamlit app would chunk them.
 * Every chunk: {chunk_id, source, kind, location, text}. Never splits mid-sentence
 * or mid-line-of-code.
 */
import type { Chunk } from './types';

export const TARGET_CHARS = 800;
export const OVERLAP_CHARS = 120;
export const MAX_CHARS = 1_200;

/** cyrb53 — small synchronous 53-bit string hash; ids only need to be unique. */
function hashId(s: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < s.length; i++) {
    const ch = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, '0');
}

export function makeChunk(text: string, source: string, kind: string, location: string): Chunk {
  text = text.trim();
  return { chunk_id: hashId(`${source}|${location}|${text}`), source, kind, location, text };
}

// ── prose ──────────────────────────────────────────────────────────────────

export function splitProse(text: string, target = TARGET_CHARS, overlap = OVERLAP_CHARS): string[] {
  text = text.trim();
  if (!text) return [];
  if (text.length <= target) return [text];

  const units: string[] = [];
  for (let para of text.split(/\n\s*\n/)) {
    para = para.trim();
    if (!para) continue;
    if (para.length <= target) {
      units.push(para);
      continue;
    }
    let buf = '';
    for (const sent of para.split(/(?<=[.!?])\s+/)) {
      if (buf && buf.length + sent.length + 1 > target) {
        units.push(buf.trim());
        buf = sent;
      } else {
        buf = `${buf} ${sent}`.trim();
      }
    }
    if (buf) units.push(buf.trim());
  }

  const pieces: string[] = [];
  let buf = '';
  for (const unit of units) {
    if (buf && buf.length + unit.length + 2 > target) {
      pieces.push(buf.trim());
      const tail = overlap ? buf.slice(-overlap) : '';
      buf = tail ? `${tail}\n\n${unit}` : unit;
    } else {
      buf = buf ? `${buf}\n\n${unit}` : unit;
    }
  }
  if (buf.trim()) pieces.push(buf.trim());
  return pieces;
}

export function chunkPlaintext(text: string, source: string, kind = 'user'): Chunk[] {
  return splitProse(text).map((piece, i) => makeChunk(piece, source, kind, `part ${i + 1}`));
}

// ── markdown (header-aware) ────────────────────────────────────────────────

export function chunkMarkdown(text: string, source: string, kind = 'context'): Chunk[] {
  const matches = [...text.matchAll(/^(#{1,6})\s+(.*)$/gm)];
  if (matches.length === 0) {
    return splitProse(text).map((piece, i) => makeChunk(piece, source, kind, `part ${i + 1}`));
  }
  const chunks: Chunk[] = [];
  const preamble = text.slice(0, matches[0].index).trim();
  if (preamble) splitProse(preamble).forEach((piece, i) => chunks.push(makeChunk(piece, source, kind, `preamble ${i + 1}`)));

  matches.forEach((m, i) => {
    const header = m[0].trim();
    const title = m[2].trim();
    const start = (m.index as number) + m[0].length;
    const end = i + 1 < matches.length ? (matches[i + 1].index as number) : text.length;
    const body = text.slice(start, end).trim();
    if (!body) return;
    const pieces = splitProse(body, MAX_CHARS);
    pieces.forEach((piece, j) => {
      const loc = pieces.length === 1 ? title : `${title} (${j + 1}/${pieces.length})`;
      chunks.push(makeChunk(`${header}\n\n${piece}`, source, kind, loc));
    });
  });
  return chunks;
}

// ── python source (code-aware) ─────────────────────────────────────────────

const SECTION_HINTS: [RegExp, string][] = [
  [/\bfillna\b|\bdropna\b|missing/i, 'missing-value handling'],
  [/^W\s*=\s*dict\(/m, 'W weights'],
  [/^FRAMEWORK\s*=/m, 'framework dict'],
  [/\bprofit\s*=|\brank\(pct/i, 'scoring'],
  [/eq_text|equation/i, 'equation-string builder'],
  [/openpyxl|to_excel|wb\.save|shutil\.copy/i, 'submission writer'],
  [/^\s*(import|from)\s/m, 'imports/config'],
];

function sectionLabel(body: string): string | null {
  for (const [re, label] of SECTION_HINTS) if (re.test(body)) return label;
  return null;
}

/** Line indices where a new top-level statement begins; tracks brackets and triple quotes. */
function statementStarts(lines: string[], start: number): number[] {
  const starts: number[] = [];
  let depth = 0;
  let inStr = false;
  let quote = '';
  let expecting = true;
  for (let i = start; i < lines.length; i++) {
    const line = lines[i];
    if (expecting && !inStr && depth === 0 && line.trim()) {
      starts.push(i);
      expecting = false;
    }
    let j = 0;
    while (j < line.length) {
      if (inStr) {
        if (line.startsWith(quote, j)) {
          inStr = false;
          j += 3;
          continue;
        }
      } else {
        if (line.startsWith('"""', j) || line.startsWith("'''", j)) {
          inStr = true;
          quote = line.slice(j, j + 3);
          j += 3;
          continue;
        }
        const ch = line[j];
        if (ch === '#') break;
        if ('([{'.includes(ch)) depth++;
        else if (')]}'.includes(ch)) depth = Math.max(0, depth - 1);
        else if (ch === '"' || ch === "'") {
          const end = line.indexOf(ch, j + 1);
          j = end === -1 ? line.length : end;
        }
      }
      j++;
    }
    if (!inStr && depth === 0 && !line.trimEnd().endsWith('\\')) expecting = true;
  }
  return starts;
}

const DICT_KEY = /^\s{1,}["']([^"']+)["']\s*:/;

function splitDictLiteral(lines: string[], lo: number, hi: number, source: string, name: string): Chunk[] | null {
  const keyLines: number[] = [];
  for (let i = lo + 1; i < hi; i++) if (DICT_KEY.test(lines[i])) keyLines.push(i);
  if (keyLines.length < 2) return null;
  return keyLines.map((k, n) => {
    const end = n + 1 < keyLines.length ? keyLines[n + 1] : hi;
    const key = (DICT_KEY.exec(lines[k]) as RegExpExecArray)[1];
    return makeChunk(`${name} entry — ${key}:\n` + lines.slice(k, end).join('\n'), source, 'code',
      `lines ${k + 1}-${end} (${name}: ${key})`);
  });
}

export function chunkPython(text: string, source: string): Chunk[] {
  const lines = text.split(/\r\n|\r|\n/);
  if (lines.length && lines[lines.length - 1] === '') lines.pop(); // str.splitlines() drops the trailing empty piece
  const chunks: Chunk[] = [];
  let start = 0;

  const stripped = text.trimStart();
  if (stripped.startsWith('"""') || stripped.startsWith("'''")) {
    const quote = stripped.slice(0, 3);
    const offset = text.length - stripped.length;
    const endQ = text.indexOf(quote, offset + 3);
    if (endQ !== -1) {
      const docEndLine = text.slice(0, endQ + 3).split('\n').length - 1;
      chunks.push(makeChunk(lines.slice(0, docEndLine + 1).join('\n'), source, 'code',
        `lines 1-${docEndLine + 1} (module docstring)`));
      start = docEndLine + 1;
    }
  }

  const cuts = [...new Set([...statementStarts(lines, start), lines.length])].sort((a, b) => a - b);
  if (cuts.length < 2) return chunks.filter((c) => c.text);

  const emit = (lo: number, hi: number) => {
    const body = lines.slice(lo, hi).join('\n').replace(/^\n+|\n+$/g, '');
    if (!body.trim()) return;
    const label = sectionLabel(body);
    chunks.push(makeChunk(body, source, 'code', `lines ${lo + 1}-${hi}` + (label ? ` (${label})` : '')));
  };

  let segStart = cuts[0];
  for (let idx = 1; idx < cuts.length; idx++) {
    const segEnd = cuts[idx];
    const m = /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:dict\(|\{)/.exec(lines[segStart]);
    if (m && segEnd - segStart > 8) {
      const perKey = splitDictLiteral(lines, segStart, segEnd, source, m[1]);
      if (perKey) {
        chunks.push(...perKey);
        segStart = segEnd;
        continue;
      }
    }
    const body = lines.slice(segStart, segEnd).join('\n');
    const nextLen = idx + 1 < cuts.length ? lines.slice(segStart, cuts[idx + 1]).join('\n').length : 1e9;
    if (body.length >= TARGET_CHARS || nextLen > MAX_CHARS || idx === cuts.length - 1) {
      emit(segStart, segEnd);
      segStart = segEnd;
    }
  }
  return chunks.filter((c) => c.text);
}
