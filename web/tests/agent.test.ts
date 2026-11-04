import { describe, expect, it } from 'vitest';
import { chatWithLLM, fitToContext, serializeToolResult, ContextOverflowError } from '../src/llm/agent';
import type { ChatMessage, CompleteOptions, CompletionResult, Engine } from '../src/llm/types';
import type { ToolFn } from '../src/tools/tools';

/** An engine that plays back scripted replies and records every prompt it was sent. */
function scripted(replies: CompletionResult[], contextTokens = 8192) {
  const seen: { messages: ChatMessage[]; opts?: CompleteOptions }[] = [];
  const engine: Engine = {
    kind: 'webllm', label: 'fake', contextTokens,
    async complete(messages, opts) {
      seen.push({ messages: messages.map((m) => ({ ...m })), opts });
      const r = replies.shift();
      if (!r) throw new Error('script exhausted');
      opts?.onToken?.(r.content);
      return r;
    },
  };
  return { engine, seen };
}

const kbHit = { source: 'trial.py', location: 'lines 1-5', text: 'W = dict(ic=0.024)' };
const baseTools = (): Record<string, ToolFn> => ({
  search_knowledge_base: async () => ({ results: [kbHit] }),
  get_customer_profile: async (a) => ({ customer_id: a.customer_id, profit: -683.52, terms: { rev_discount: 354.85 } }),
});
const deps = (engine: Engine, tools = baseTools()) => ({ engine, tools, systemPrompt: 'SYSTEM', mode: 'compact' as const });

describe('chatWithLLM', () => {
  it('runs a tool round, then answers from the tool result', async () => {
    const { engine, seen } = scripted([
      { content: '', toolCalls: [{ name: 'get_customer_profile', arguments: { customer_id: 1 } }] },
      { content: 'Customer 1 scores -$683.52.', toolCalls: [] },
    ]);
    const out = await chatWithLLM('Profit breakdown for customer 1', [], deps(engine), { useTools: true });
    expect(out.reply).toBe('Customer 1 scores -$683.52.');
    expect(out.toolsUsed).toEqual(['get_customer_profile']);
    expect(out.traces[0].result).toMatchObject({ customer_id: 1, profit: -683.52 });
    // round 2 sees the assistant's tool call and the tool's JSON
    const second = seen[1].messages;
    expect(second.at(-2)).toMatchObject({ role: 'assistant', tool_calls: [{ name: 'get_customer_profile' }] });
    expect(second.at(-1)).toMatchObject({ role: 'tool', tool_name: 'get_customer_profile' });
    expect(JSON.parse(second.at(-1)!.content).customer_id).toBe(1);
    expect(out.history).toEqual([{ role: 'user', text: 'Profit breakdown for customer 1' }, { role: 'model', text: 'Customer 1 scores -$683.52.' }]);
  });

  it('prefetches knowledge-base excerpts into the user turn (tool mode only)', async () => {
    const withTools = scripted([{ content: 'ok', toolCalls: [] }]);
    await chatWithLLM('how were weights derived', [], deps(withTools.engine), { useTools: true });
    expect(withTools.seen[0].messages.at(-1)!.content).toContain('Retrieved knowledge-base excerpts');
    expect(withTools.seen[0].messages.at(-1)!.content).toContain('[trial.py — lines 1-5]');
    expect(withTools.seen[0].opts?.tools).toHaveLength(8);

    const direct = scripted([{ content: 'ok', toolCalls: [] }]);
    await chatWithLLM('hello', [], deps(direct.engine), { useTools: false });
    expect(direct.seen[0].messages.at(-1)!.content).toBe('hello');
    expect(direct.seen[0].opts?.tools).toBeUndefined();
  });

  it('survives a broken retrieval layer and unknown tools', async () => {
    const tools = baseTools();
    tools.search_knowledge_base = async () => { throw new Error('embedder down'); };
    const { engine, seen } = scripted([
      { content: '', toolCalls: [{ name: 'does_not_exist', arguments: {} }, { name: '__invalid__', arguments: { raw: '{' } }] },
      { content: 'sorry', toolCalls: [] },
    ]);
    const out = await chatWithLLM('x', [], deps(engine, tools), { useTools: true });
    expect(out.reply).toBe('sorry');
    const tool = seen[1].messages.filter((m) => m.role === 'tool');
    expect(JSON.parse(tool[0].content).error).toBe('Unknown tool: does_not_exist');
    expect(JSON.parse(tool[1].content).error).toContain('Could not parse');
  });

  it('gives up after six tool rounds', async () => {
    const loop = () => ({ content: '', toolCalls: [{ name: 'get_customer_profile', arguments: { customer_id: 2 } }] });
    const { engine, seen } = scripted(Array.from({ length: 6 }, loop));
    const out = await chatWithLLM('x', [], deps(engine), { useTools: true });
    expect(out.reply).toBe('Ran out of tool rounds. Try a more specific question.');
    expect(seen).toHaveLength(6);
  });

  it('keeps only the last 12 history messages', async () => {
    const history = Array.from({ length: 20 }, (_, i) => ({ role: (i % 2 ? 'model' : 'user') as 'user' | 'model', text: `m${i}` }));
    const { engine, seen } = scripted([{ content: 'ok', toolCalls: [] }]);
    await chatWithLLM('now', history, deps(engine), { useTools: false });
    const sent = seen[0].messages;
    expect(sent).toHaveLength(1 + 12 + 1);
    expect(sent[1].content).toBe('m8');
    expect(sent[2].role).toBe('assistant');
  });

  it('truncates oversized tool output like llm.py', () => {
    const big = { rows: 'x'.repeat(10_000) };
    const s = serializeToolResult(big, 4000);
    expect(s.length).toBe(4000 + '... [truncated]"}'.length);
    expect(s.endsWith('... [truncated]"}')).toBe(true);
    expect(serializeToolResult({ a: 1 }, 4000)).toBe('{"a":1}');
  });
});

describe('fitToContext', () => {
  const msg = (role: ChatMessage['role'], n: number): ChatMessage => ({ role, content: 'w'.repeat(n) });
  it('drops the oldest history first and never the system prompt or current turn', () => {
    const msgs = [msg('system', 12000), msg('user', 6000), msg('assistant', 6000), msg('user', 6000), msg('assistant', 6000), msg('user', 400)];
    const fitted = fitToContext(msgs, 5, 8192, false);
    expect(fitted[0].role).toBe('system');
    expect(fitted.at(-1)!.content).toHaveLength(400);
    expect(fitted.length).toBeLessThan(msgs.length);
  });
  it('shrinks tool output, then retrieval excerpts, before giving up', () => {
    const msgs = [msg('system', 16000), msg('user', 400), msg('assistant', 10), msg('tool', 9000)];
    const fitted = fitToContext(msgs, 1, 8192, false);
    expect(fitted.at(-1)!.content.length).toBeLessThan(1300);
    const withExtra = [msg('system', 21500), { role: 'user' as const, content: 'q' + '\n\n---\nRetrieved knowledge-base excerpts' + 'e'.repeat(3000) }];
    expect(fitToContext(withExtra, 1, 8192, false)[1].content).toBe('q');
  });
  it('throws a clear error when even that cannot fit', () => {
    expect(() => fitToContext([msg('system', 40000), msg('user', 10)], 1, 8192, true)).toThrow(ContextOverflowError);
  });
});
