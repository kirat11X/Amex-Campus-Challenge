import { describe, expect, it } from 'vitest';
import { chunkMarkdown, chunkPython, splitProse } from '../src/rag/chunker';
import { fmtG, inferColumns, parseCsv, tabularProfileDocs } from '../src/rag/profile';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import golden from './golden/chunkers.json';

const brief = (cs: { kind: string; location: string; text: string }[]) => cs.map(({ kind, location, text }) => ({ kind, location, text }));
const read = (p: string) => readFileSync(join(__dirname, '..', p), 'utf8');

describe('chunkers are faithful ports of profitlens/rag/chunker.py', () => {
  it('markdown: the grounding file splits into identical chunks', () => {
    const got = brief(chunkMarkdown(read('public/data/context.md'), 'PROFITLENS_CONTEXT.md', 'context'));
    expect(got.length).toBe(golden.context_md.length);
    expect(got).toEqual(golden.context_md);
  });
  it('python: the submission builder splits into identical chunks', () => {
    const trial = readFileSync(join(__dirname, '..', '..', 'main', 'trial.py'), 'utf8');
    const got = brief(chunkPython(trial, 'trial.py'));
    expect(got.map((c) => c.location)).toEqual(golden.trial_py.map((c) => c.location));
    expect(got).toEqual(golden.trial_py);
  });
  it('prose: same pieces at several target sizes', () => {
    for (const t of [300, 800, 1200]) {
      expect(splitProse(golden.prose.text, t)).toEqual((golden.prose.pieces as Record<string, string[]>)[String(t)]);
    }
  });
});

describe('table profiling matches profile_docs.py', () => {
  it("formats numbers like Python's {x:,.4g}", () => {
    for (const [v, want] of golden.fmt as [number, string][]) expect(fmtG(v), `fmtG(${v})`).toBe(want);
  });
  it('csv upload becomes the same schema / column / preview chunks', () => {
    const table = parseCsv(golden.csv.text);
    const dtypes = Object.fromEntries(inferColumns(table).map((c) => [c.name, c.dtype]));
    const pyDtypes = golden.csv.dtypes as Record<string, string>;
    for (const [name, d] of Object.entries(pyDtypes)) expect(dtypes[name]).toBe(d === 'str' ? 'object' : d);

    const got = brief(tabularProfileDocs(table, 'sample.csv', 'sample.csv'));
    const want = golden.csv.chunks.map((c) => ({ ...c, text: c.text.replace(/\((str|string)\)/g, '(object)') }));
    expect(got.map((c) => c.location)).toEqual(want.map((c) => c.location));
    // schema + numeric/text column cards are text-identical; the preview table is formatted by tabulate in Python
    for (let i = 0; i < want.length - 1; i++) expect(got[i].text, want[i].location).toBe(want[i].text);
    expect(got[got.length - 1].text).toContain('| id | amount | name | region |');
  });
  it('parses quoted CSV fields', () => {
    const t = parseCsv('a,b\n"x, y","he said ""hi"""\n2,3\n');
    expect(t.rows).toEqual([['x, y', 'he said "hi"'], ['2', '3']]);
  });
});
