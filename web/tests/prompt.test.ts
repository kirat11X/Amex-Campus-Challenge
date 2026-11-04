import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { buildSystemPrompt, compactContext, estimateTokens, FALLBACK_SYSTEM_PROMPT, formatToolCalls, parseToolCalls, RAG_GUIDANCE, TOOLS, toolsBlock } from '../src/llm/prompt';

const md = readFileSync(join(__dirname, '..', 'public', 'data', 'context.md'), 'utf8');
const block = '\n\n## RUNTIME VERIFICATION (auto-generated at startup)\nStatus: ALL MATCH Section 8\n';

describe('system prompt', () => {
  it('full mode is the Streamlit prompt: context file + runtime block + retrieval guidance', () => {
    expect(buildSystemPrompt(md, block, 'full')).toBe(md + block + RAG_GUIDANCE);
  });
  it('falls back to the short prompt when the context file is missing', () => {
    expect(buildSystemPrompt(null, block, 'full')).toBe(FALLBACK_SYSTEM_PROMPT + block + RAG_GUIDANCE);
  });
  it('compact mode keeps the rules, formula and ground-truth facts, and fits a 4B model with room to spare', () => {
    const c = buildSystemPrompt(md, block, 'compact');
    for (const must of ['## 1. WHO YOU ARE', '## 2. HOW TO RESPOND', '## 3. WHAT YOU ANSWER', '### 7.1 Definition', '### 7.2 Term decomposition',
      '## 8. GROUND-TRUTH DATASET FACTS', '$2,012.86', '## 11. HARD "NEVER" LIST', 'RUNTIME VERIFICATION', 'search_knowledge_base']) {
      expect(c, must).toContain(must);
    }
    for (const gone of ['## 4. THE CHALLENGE', '## 5. THE PRODUCT', '### 7.3', '### 7.4', '## 9. GLOSSARY', '## 10. GOLD-STANDARD']) {
      expect(c, gone).not.toContain(gone);
    }
    const full = estimateTokens(md + block + RAG_GUIDANCE);
    const compact = estimateTokens(c);
    expect(compact).toBeLessThan(full * 0.65);
    expect(compact + estimateTokens(JSON.stringify(TOOLS))).toBeLessThan(5200); // leaves ~2K of an 8K window
  });
  it('compactContext is a pure subset (every kept line exists in the original)', () => {
    const orig = new Set(md.split('\n'));
    const extra = compactContext(md).split('\n').filter((l) => l.trim() && !orig.has(l));
    expect(extra.join(' ')).toContain('Not repeated here');
    expect(extra.length).toBeLessThan(6);
  });
  it('declares the same eight tools as llm.py', () => {
    expect(TOOLS.map((t) => t.function.name)).toEqual([
      'get_customer_profile', 'get_top_customers', 'segment_compare', 'term_contribution_summary',
      'get_distribution', 'query_segment', 'explain_formula', 'search_knowledge_base']);
  });
});

describe('prompt-level tool calling (Qwen/Hermes format)', () => {
  it('lists the tools in <tools> tags', () => {
    const b = toolsBlock(TOOLS);
    expect(b).toContain('<tools>');
    expect(b).toContain('"name":"get_customer_profile"');
    expect(b).toContain('<tool_call>');
  });
  it('round-trips tool calls', () => {
    const calls = [{ name: 'get_customer_profile', arguments: { customer_id: 20 } }, { name: 'explain_formula', arguments: {} }];
    const parsed = parseToolCalls('Let me check.\n' + formatToolCalls(calls));
    expect(parsed.calls).toEqual(calls);
    expect(parsed.content).toBe('Let me check.');
  });
  it('accepts string arguments, parameters and an unterminated block', () => {
    expect(parseToolCalls('<tool_call>{"name":"a","arguments":"{\\"x\\":1}"}</tool_call>').calls[0].arguments).toEqual({ x: 1 });
    expect(parseToolCalls('<tool_call>{"name":"a","parameters":{"y":2}}</tool_call>').calls[0].arguments).toEqual({ y: 2 });
    expect(parseToolCalls('<tool_call>{"name":"a","arguments":{"z":3}}').calls[0].arguments).toEqual({ z: 3 });
  });
  it('turns malformed JSON into a retryable __invalid__ call', () => {
    expect(parseToolCalls('<tool_call>{name: broken</tool_call>').calls[0].name).toBe('__invalid__');
  });
  it('strips <think> blocks and returns plain answers untouched', () => {
    expect(parseToolCalls('<think>hmm</think>The answer is 4.')).toEqual({ content: 'The answer is 4.', calls: [] });
  });
});
