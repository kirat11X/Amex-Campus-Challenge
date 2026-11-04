import { beforeAll, describe, expect, it } from 'vitest';
import { Dataset } from '../src/data/dataset';
import { makeTools, type ToolFn } from '../src/tools/tools';
import { histogram, percentileSorted, sum } from '../src/tools/stats';
import golden from './golden/tools.json';
import { approx, diskBytes } from './helpers';

let tools: Record<string, ToolFn>;
beforeAll(async () => {
  const ds = await Dataset.load(diskBytes);
  tools = makeTools({
    ds,
    kb: { userChunkCount: () => 0, search: async () => [] },
    userCorpusEnabled: () => true,
  });
}, 60_000);

describe('tools reproduce the Python tools (golden outputs from profitlens/tools.py)', () => {
  const cases = golden as { tool: string; args: Record<string, unknown>; result: unknown }[];
  for (const [i, c] of cases.entries()) {
    it(`${i}: ${c.tool}(${JSON.stringify(c.args).slice(0, 70)})`, async () => {
      const got = await tools[c.tool](c.args);
      // sums over 500K doubles may differ from numpy in the last digits; everything else is exact
      const bad = approx(got, c.result, c.tool === 'query_segment' || c.tool === 'get_distribution' ? 1e-3 : 1e-6);
      expect(bad, bad ?? '').toBeNull();
    }, 120_000);
  }
});

describe('error handling matches the Python tools', () => {
  it('unknown customer / column / agg / unsafe filter', async () => {
    expect(await tools.get_customer_profile({ customer_id: 9_999_999 })).toMatchObject({ error: 'Customer 9999999 not found.', hint: 'Valid IDs range from 0 to 499999.' });
    expect(await tools.segment_compare({ feature: 'nope' })).toHaveProperty('error', "Invalid feature 'nope'.");
    expect(await tools.get_distribution({ column: 'nope' })).toHaveProperty('error', "Invalid column 'nope'.");
    expect(await tools.query_segment({ filters: 'f3 >= 1', agg: 'median' })).toHaveProperty('error', "Invalid agg 'median'.");
    expect(await tools.query_segment({ filters: '__import__("os")', agg: 'count' })).toHaveProperty('error', 'Unsafe query pattern detected.');
    expect(await tools.query_segment({ filters: 'zzz > 1', agg: 'count' })).toMatchObject({ error: expect.stringContaining('Query failed') });
  });
  it('coerces LLM-style string arguments', async () => {
    const r = await tools.get_customer_profile({ customer_id: '20' });
    expect(r.customer_id).toBe(20);
    expect((await tools.get_top_customers({ n: '3' })).n).toBe(3);
    expect((await tools.get_top_customers({ n: 999 })).n).toBe(50);
  });
});

describe('knowledge-base tool', () => {
  const hit = { chunk_id: 'x', source: 'trial.py', kind: 'code', location: 'lines 1-5', text: 'W = dict()', score: 0.7 };
  const mk = (enabled: boolean, userChunks: number) => ({
    kb: { userChunkCount: () => userChunks, search: async (_q: string, corpus: 'core' | 'user') => (corpus === 'core' ? [hit] : []) },
    enabled,
  });
  const make = async (enabled: boolean, userChunks: number) => {
    const m = mk(enabled, userChunks);
    return makeTools({ ds: await Dataset.load(diskBytes), kb: m.kb, userCorpusEnabled: () => m.enabled });
  };
  it('reports when user files are switched off or missing', async () => {
    const off = await (await make(false, 3)).search_knowledge_base({ query: 'q' });
    expect(String(off.note)).toContain('NOT consulted');
    const none = await (await make(true, 0)).search_knowledge_base({ query: 'q' });
    expect(none.note).toBe('no user files ingested yet');
    expect((none.results as unknown[]).length).toBe(1);
  });
  it('says nothing relevant when retrieval is empty, and survives embedding failure', async () => {
    const ds = await Dataset.load(diskBytes);
    const empty = makeTools({ ds, kb: { userChunkCount: () => 0, search: async () => [] }, userCorpusEnabled: () => true });
    expect(await empty.search_knowledge_base({ query: 'weather' })).toMatchObject({ results: [], note: expect.stringContaining('nothing relevant found') });
    const broken = makeTools({ ds, kb: { userChunkCount: () => 0, search: async () => { throw new Error('model offline'); } }, userCorpusEnabled: () => true });
    expect(await broken.search_knowledge_base({ query: 'x' })).toMatchObject({ results: [], note: expect.stringContaining('retrieval unavailable') });
  });
});

describe('numpy-compatible stats', () => {
  it('percentile (linear), histogram edges and compensated sum', () => {
    const v = Float64Array.from([1, 2, 3, 4, 10]);
    expect(percentileSorted(v, 50)).toBe(3);
    expect(percentileSorted(v, 90)).toBeCloseTo(7.6, 12);
    const h = histogram(Float64Array.from([0, 1, 2, 3, 4, 4]), 4);
    expect(h.counts).toEqual([1, 1, 1, 3]); // last bin is closed on the right
    expect(h.edges).toEqual([0, 1, 2, 3, 4]);
    expect(sum(Float64Array.from([1e16, 1, -1e16]))).toBe(1);
  });
});
