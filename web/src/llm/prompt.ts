/**
 * System prompt + tool declarations, ported from profitlens/llm.py.
 *
 * Full mode is exactly the Streamlit prompt: PROFITLENS_CONTEXT.md verbatim + the
 * runtime-verification block + the retrieval guidance (~5.7K tokens).
 * Compact mode keeps the sections that drive behaviour and numbers (1, 2, 3, 7.1-7.2,
 * 8, 11) so a 4B browser model has room left in an 8K window; the dropped sections
 * stay reachable through search_knowledge_base, which indexes the full file.
 */
import type { ToolSpec } from './types';

export const MAX_TOOL_ROUNDS = 6;
export const MAX_HISTORY_MESSAGES = 12;

export const FALLBACK_SYSTEM_PROMPT =
  'You are ProfitLens, an issuer-side profitability analyst for the Amex Premier Card.\n' +
  'Ground every number in tool outputs only — never estimate or invent values.\n' +
  'Use the available tools to answer questions about customer profitability.\n' +
  'The profitability formula and its weights are frozen — explain them, do not change them.\n' +
  'Currency is USD. Missing feature values mean no recorded activity, scored as 0.\n';

export const RAG_GUIDANCE =
  '\n\n## KNOWLEDGE-BASE RETRIEVAL\n' +
  'You MUST call search_knowledge_base BEFORE answering — never say something ' +
  'is unavailable without searching first — for: questions about the code/implementation, the ' +
  'methodology write-up, the challenge rules/deck, feature meanings, and ANY ' +
  'question about user-uploaded files. Do not use it for customer-level numbers ' +
  'or live aggregates — those come from the pandas tools. When you answer from ' +
  "retrieved chunks, cite the source and location (e.g. 'per trial.py lines 40-55' " +
  "or 'from your uploaded report.pdf, p.3'). If retrieval returns nothing " +
  'relevant, say so.\n' +
  'Trust order: live pandas tool result > Section 8 stats > retrieved chunk ' +
  '(cite it) > refuse. For user-uploaded files, retrieved chunks ARE the primary ' +
  'source.\n';

const fn = (name: string, description: string, properties: Record<string, unknown> = {}, required: string[] = []): ToolSpec => ({
  type: 'function',
  function: { name, description, parameters: { type: 'object', properties, required } },
});

export const TOOLS: ToolSpec[] = [
  fn('get_customer_profile', 'Get full profile for a customer: raw features, P&L terms, profit, rank, top-20% flag.',
    { customer_id: { type: 'integer', description: 'Customer ID' } }, ['customer_id']),
  fn('get_top_customers', 'Get top/bottom N customers by profit with key drivers.',
    { n: { type: 'integer', description: 'Count (max 50, default 10)' },
      ascending: { type: 'boolean', description: 'True for bottom N' } }),
  fn('segment_compare', 'Compare a feature between top-20% and rest: mean, median, p25, p75, missing%.',
    { feature: { type: 'string', description: 'f1-f23 or term column name' } }, ['feature']),
  fn('term_contribution_summary', 'Aggregate mean of each P&L term for top-20% vs rest, showing biggest separators.'),
  fn('get_distribution', 'Histogram data for any column. App renders as a chart.',
    { column: { type: 'string', description: 'Column name' },
      bins: { type: 'integer', description: 'Bins (default 40)' },
      clip_percentiles: { type: 'array', items: { type: 'number' }, description: '[lo,hi] percentiles' } }, ['column']),
  fn('query_segment', 'Guarded pandas query with aggregation for ad-hoc questions.',
    { filters: { type: 'string', description: 'pandas query expression' },
      agg: { type: 'string', description: 'count|mean_profit|sum_profit|top20_share' } }, ['filters', 'agg']),
  fn('explain_formula', 'Return weights, equation, term table, and submission methodology.'),
  fn('search_knowledge_base',
    "Semantic search over the ProfitLens knowledge base: the project's own code, methodology write-up, " +
    "challenge deck, feature dictionary and generated data profiles ('core'), plus any files the user " +
    "uploaded ('user'). Returns text chunks with source and location for citation. " +
    'Not a source of customer-level numbers or live aggregates.',
    { query: { type: 'string', description: 'Natural-language search query' },
      corpus: { type: 'string', description: 'core | user | both (default both)' },
      k: { type: 'integer', description: 'Number of chunks (default 5, max 10)' } }, ['query']),
];

export type PromptMode = 'full' | 'compact';

const KEEP_SECTIONS = new Set(['1', '2', '3', '7', '8', '11']);

export function compactContext(md: string): string {
  const parts = md.split(/^(?=## \d+\. )/m);
  const out: string[] = [parts[0]];
  for (const part of parts.slice(1)) {
    const num = /^## (\d+)\./.exec(part)?.[1] ?? '';
    if (!KEEP_SECTIONS.has(num)) continue;
    // Section 7: keep the definition and term table, drop rationale/methodology (7.3+).
    const cut = num === '7' ? part.search(/^### 7\.3/m) : -1;
    out.push(cut > 0 ? part.slice(0, cut) : part);
  }
  out.push(
    '\n(Not repeated here, but searchable with search_knowledge_base: the challenge description, ' +
      'the Premier Card product, the full feature dictionary, coefficient rationale and methodology, ' +
      'glossary, and worked answer examples.)\n',
  );
  return out.join('');
}

export function buildSystemPrompt(contextMd: string | null, runtimeBlock: string, mode: PromptMode): string {
  if (contextMd === null) return FALLBACK_SYSTEM_PROMPT + runtimeBlock + RAG_GUIDANCE;
  const body = mode === 'compact' ? compactContext(contextMd) : contextMd;
  return body + runtimeBlock + RAG_GUIDANCE;
}

/** Rough token estimate — good enough to keep a prompt inside the window. */
export const estimateTokens = (s: string) => Math.ceil(s.length / 3.2);

// ── prompt-level tool calling (Qwen/Hermes format) for engines without native tools ──

export function toolsBlock(tools: ToolSpec[]): string {
  return (
    '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' +
    'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' +
    tools.map((t) => JSON.stringify(t)).join('\n') +
    '\n</tools>\n\nFor each function call, return a json object with function name and arguments ' +
    'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
  );
}

export function formatToolCalls(calls: { name: string; arguments: Record<string, unknown> }[]): string {
  return calls.map((c) => `<tool_call>\n${JSON.stringify({ name: c.name, arguments: c.arguments })}\n</tool_call>`).join('\n');
}

/** Pull `<tool_call>` blocks out of raw model text. Malformed JSON becomes an `__invalid__` call so the model can retry. */
export function parseToolCalls(text: string): { content: string; calls: { name: string; arguments: Record<string, unknown> }[] } {
  const calls: { name: string; arguments: Record<string, unknown> }[] = [];
  const re = /<tool_call>\s*([\s\S]*?)\s*(?:<\/tool_call>|$)/g;
  for (const m of text.matchAll(re)) {
    const raw = m[1].trim();
    if (!raw) continue;
    try {
      const j = JSON.parse(raw);
      const name = String(j.name ?? j.function?.name ?? '');
      let args = j.arguments ?? j.parameters ?? {};
      if (typeof args === 'string') args = JSON.parse(args);
      if (!name) throw new Error('no name');
      calls.push({ name, arguments: args as Record<string, unknown> });
    } catch {
      calls.push({ name: '__invalid__', arguments: { raw: raw.slice(0, 300) } });
    }
  }
  const content = text.replace(/<tool_call>[\s\S]*?(?:<\/tool_call>|$)/g, '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  return { content, calls };
}
