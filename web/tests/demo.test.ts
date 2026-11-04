import { describe, expect, it } from 'vitest';
import { DemoEngine } from '../src/llm/demo';

describe('DemoEngine', () => {
  const engine = new DemoEngine();

  it('identifies customer 1 breakdown request and returns tool call', async () => {
    const res = await engine.complete([{ role: 'user', content: 'Profit breakdown for customer 1' }]);
    expect(res.toolCalls).toHaveLength(1);
    expect(res.toolCalls[0].name).toBe('get_customer_profile');
    expect(res.toolCalls[0].arguments).toEqual({ customer_id: 1 });
  });

  it('identifies customer 20 breakdown request and returns tool call', async () => {
    const res = await engine.complete([{ role: 'user', content: 'Why is customer 20 in the top 20%?' }]);
    expect(res.toolCalls).toHaveLength(1);
    expect(res.toolCalls[0].name).toBe('get_customer_profile');
    expect(res.toolCalls[0].arguments).toEqual({ customer_id: 20 });
  });

  it('answers formula explanation directly with table', async () => {
    const res = await engine.complete([{ role: 'user', content: 'Explain my profitability formula term by term' }]);
    expect(res.toolCalls).toHaveLength(0);
    expect(res.content).toContain('rev_discount');
    expect(res.content).toContain('cost_collections');
  });

  it('identifies lounge comparison', async () => {
    const res = await engine.complete([{ role: 'user', content: 'Compare lounge usage between top 20% and rest' }]);
    expect(res.toolCalls).toHaveLength(1);
    expect(res.toolCalls[0].name).toBe('segment_compare');
  });

  it('returns narrative after customer 1 tool response', async () => {
    const toolMsg = {
      role: 'tool' as const,
      tool_name: 'get_customer_profile',
      content: JSON.stringify({ customer_id: 1, profit: -683.52, is_top20: false, percentile_rank: 0.152 }),
    };
    const res = await engine.complete([toolMsg]);
    expect(res.content).toContain('Customer 1 scores **-683.52**');
    expect(res.content).toContain('airline credit ($-372.40)');
  });
});
