/** Turn an uploaded file into chunks (ported from profitlens/rag/ingest.py:_parse_upload). */
import { chunkMarkdown, chunkPlaintext, chunkPython, makeChunk, splitProse } from './chunker';
import { parseCsv, tabularProfileDocs, type Cell } from './profile';
import type { Chunk } from './types';

export const SUPPORTED_EXTS = ['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'py', 'json'];
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const decode = (data: ArrayBuffer) => new TextDecoder('utf-8', { fatal: false }).decode(data);

async function pdfPages(data: ArrayBuffer): Promise<[number, string][]> {
  const pdfjs = await import('pdfjs-dist');
  const workerUrl = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default;
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(data.slice(0)) }).promise;
  const pages: [number, string][] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const content = await (await doc.getPage(i)).getTextContent();
    let text = '';
    for (const it of content.items) {
      if (!('str' in it)) continue;
      text += it.str + (it.hasEOL ? '\n' : ' ');
    }
    text = text.replace(/[ \t]+\n/g, '\n').trim();
    if (text) pages.push([i, text]);
  }
  return pages;
}

export async function parseUpload(name: string, data: ArrayBuffer): Promise<Chunk[]> {
  const ext = name.includes('.') ? name.split('.').pop()!.toLowerCase() : '';
  if (!SUPPORTED_EXTS.includes(ext)) {
    throw new Error(`Unsupported file type '.${ext}'. Supported: ${SUPPORTED_EXTS.map((e) => '.' + e).join(', ')}`);
  }

  if (ext === 'pdf') {
    const pages = await pdfPages(data);
    if (!pages.length) throw new Error('No extractable text (scanned or image-only PDF?)');
    return pages.flatMap(([no, text]) =>
      splitProse(text, 1_200).map((piece, i) => makeChunk(piece, name, 'user', i === 0 ? `p.${no}` : `p.${no} (${i + 1})`)));
  }

  if (ext === 'docx') {
    const mammoth = await import('mammoth/mammoth.browser.js');
    const { value } = await mammoth.extractRawText({ arrayBuffer: data });
    const paragraphs = value.split(/\n+/).map((p: string) => p.trim()).filter(Boolean);
    if (!paragraphs.length) throw new Error('Document contains no text paragraphs');
    return splitProse(paragraphs.join('\n\n')).map((piece, i) => makeChunk(piece, name, 'user', `part ${i + 1}`));
  }

  if (ext === 'csv') return tabularProfileDocs(parseCsv(decode(data)), name, name);

  if (ext === 'xlsx') {
    const readXlsx = (await import('read-excel-file/universal')).default;
    const sheets = await readXlsx(data);
    const chunks = sheets.flatMap(({ sheet, data: grid }) => {
      const rows = (grid as unknown[][]).filter((r) => r.some((c) => c !== null && c !== undefined && c !== ''));
      if (!rows.length) return [];
      const columns = rows[0].map((h, i) => (h === null || h === undefined || h === '' ? `column_${i + 1}` : String(h)));
      const body: Cell[][] = rows.slice(1).map((r) => columns.map((_, j) => {
        const v = r[j];
        return v === undefined || v === null ? null : v instanceof Date ? v.toISOString().slice(0, 10) : (v as Cell);
      }));
      return tabularProfileDocs({ columns, rows: body }, name, `${name} / ${sheet}`);
    });
    if (!chunks.length) throw new Error('Workbook contains no readable sheets');
    return chunks;
  }

  const text = decode(data);
  if (ext === 'py') return chunkPython(text, name);
  if (ext === 'md') return chunkMarkdown(text, name, 'user');
  let body = text;
  if (ext === 'json') {
    try { body = JSON.stringify(JSON.parse(text), null, 2); } catch { /* not valid JSON — index as plain text */ }
  }
  return chunkPlaintext(body, name, 'user');
}
