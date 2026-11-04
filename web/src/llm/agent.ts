/**
 * The chat loop, ported from profitlens/llm.py:chat_with_llm — same shape:
 * system prompt + trimmed history + prefetched knowledge-base excerpts, then up to
 * six tool-calling rounds. It talks to an `Engine`, so the in-browser model and a
 * local Ollama behave identically.
 */
import type { ToolResult } from '../data/types';
import type { ToolFn } from '../tools/tools';
import { estimateTokens, MAX_HISTORY_MESSAGES, MAX_TOOL_ROUNDS, TOOLS, type PromptMode } from './prompt';
import type { ChatMessage, Engine } from './types';

export interface HistoryItem {
  role: 'user' | 'model';
  text: string;
}

export interface ToolTrace {
  name: string;
  args: Record<string, unknown>;
  result: ToolResult;
}

export interface ChatDeps {
  engine: Engine;
  tools: Record<string, ToolFn>;
  systemPrompt: string;
  mode: PromptMode;
}

export interface ChatOptions {
  useTools: boolean;
  onStatus?: (status: string) => void;
  onToken?: (textSoFar: string) => void;
  signal?: AbortSignal;
}

export interface ChatOutcome {
  reply: string;
  history: HistoryItem[];
  toolsUsed: string[];
  traces: ToolTrace[];
}

export class ContextOverflowError extends Error {}

const LIMITS = {
  full: { toolChars: 4000, prefetchChunks: 4, prefetchChars: 1200 },
  compact: { toolChars: 3000, prefetchChunks: 3, prefetchChars: 800 },
};
const ANSWER_RESERVE = 700; // tokens kept free for the model's reply
const TOOLS_TOKENS = estimateTokens(JSON.stringify(TOOLS));

/** Python: json.dumps(result)[:MAX] + '... [truncated]"}'. */
export function serializeToolResult(result: unknown, maxChars: number): string {
  const text = JSON.stringify(result);
  return text.length > maxChars ? text.slice(0, maxChars) + '... [truncated]"}' : text;
}

async function executeToolCall(tools: Record<string, ToolFn>, name: string, args: Record<string, unknown>): Promise<ToolResult> {
  if (name === '__invalid__') {
    return { error: 'Could not parse that tool call as JSON.', hint: 'Reply with <tool_call>{"name": ..., "arguments": {...}}</tool_call> containing valid JSON.' };
  }
  const fn = tools[name];
  if (!fn) return { error: `Unknown tool: ${name}` };
  try {
    return await fn(args ?? {});
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}

/** Always-on retrieval for tool mode: a small model won't reliably decide to call search_knowledge_base. */
async function prefetchContext(tools: Record<string, ToolFn>, query: string, limits: (typeof LIMITS)['full']): Promise<string> {
  try {
    const out = await tools.search_knowledge_base({ query, corpus: 'both', k: limits.prefetchChunks });
    const hits = (out.results ?? []) as { source: string; location: string; text: string }[];
    if (!hits.length) return '';
    const blocks = hits.map((h) => `[${h.source} — ${h.location}]\n${h.text.slice(0, limits.prefetchChars)}`);
    return '\n\n---\nRetrieved knowledge-base excerpts (cite source + location when you use them):\n\n' + blocks.join('\n\n');
  } catch (e) {
    console.warn('Prefetch retrieval failed:', e);
    return '';
  }
}

const tokensOf = (messages: ChatMessage[]) =>
  messages.reduce((n, m) => n + estimateTokens(m.content) + (m.tool_calls ? estimateTokens(JSON.stringify(m.tool_calls)) : 0) + 4, 0);

/** Keep the prompt inside the context window: drop old history, then shrink tool output, then retrieval excerpts. */
export function fitToContext(messages: ChatMessage[], turnStart: number, contextTokens: number, withTools: boolean): ChatMessage[] {
  const budget = contextTokens - ANSWER_RESERVE - (withTools ? TOOLS_TOKENS : 0);
  const msgs = [...messages];
  let start = turnStart;
  while (tokensOf(msgs) > budget && start > 1) {
    msgs.splice(1, 1);
    start--;
  }
  if (tokensOf(msgs) > budget) {
    for (let i = start + 1; i < msgs.length && tokensOf(msgs) > budget; i++) {
      if (msgs[i].role === 'tool' && msgs[i].content.length > 1200) {
        msgs[i] = { ...msgs[i], content: msgs[i].content.slice(0, 1200) + '... [truncated]"}' };
      }
    }
  }
  if (tokensOf(msgs) > budget) {
    const u = msgs[start];
    const cut = u.content.indexOf('\n\n---\nRetrieved knowledge-base excerpts');
    if (cut > 0) msgs[start] = { ...u, content: u.content.slice(0, cut) };
  }
  if (tokensOf(msgs) > budget) {
    throw new ContextOverflowError(
      `This conversation no longer fits the model's ${contextTokens}-token window. Reset the chat, or use a model/engine with a larger context.`,
    );
  }
  return msgs;
}

export async function chatWithLLM(
  userMessage: string,
  history: HistoryItem[],
  deps: ChatDeps,
  opts: ChatOptions,
): Promise<ChatOutcome> {
  const { engine, tools, systemPrompt, mode } = deps;
  const limits = LIMITS[mode];

  const messages: ChatMessage[] = [{ role: 'system', content: systemPrompt }];
  for (const h of history.slice(-MAX_HISTORY_MESSAGES)) {
    messages.push({ role: h.role === 'model' ? 'assistant' : 'user', content: h.text });
  }
  const turnStart = messages.length;
  if (opts.useTools) opts.onStatus?.('Searching the knowledge base…');
  const extra = opts.useTools ? await prefetchContext(tools, userMessage, limits) : '';
  messages.push({ role: 'user', content: userMessage + extra });

  const toolsUsed: string[] = [];
  const traces: ToolTrace[] = [];
  let text: string | null = null;

  for (let round = 0; round < (opts.useTools ? MAX_TOOL_ROUNDS : 1); round++) {
    opts.onStatus?.(round === 0 ? 'Thinking…' : 'Reading the tool results…');
    const fitted = fitToContext(messages, turnStart, engine.contextTokens, opts.useTools);
    const reply = await engine.complete(fitted, {
      tools: opts.useTools ? TOOLS : undefined,
      signal: opts.signal,
      onToken: opts.onToken,
    });
    if (!reply.toolCalls.length) {
      text = reply.content.trim() || '(No response)';
      break;
    }
    messages.push({ role: 'assistant', content: reply.content, tool_calls: reply.toolCalls });
    for (const call of reply.toolCalls) {
      toolsUsed.push(call.name);
      opts.onStatus?.(`Running ${call.name}(${JSON.stringify(call.arguments).slice(0, 80)})…`);
      const result = await executeToolCall(tools, call.name, call.arguments);
      traces.push({ name: call.name, args: call.arguments, result });
      messages.push({ role: 'tool', tool_name: call.name, content: serializeToolResult(result, limits.toolChars) });
    }
  }
  if (text === null) text = 'Ran out of tool rounds. Try a more specific question.';

  return {
    reply: text,
    history: [...history, { role: 'user', text: userMessage }, { role: 'model', text }],
    toolsUsed,
    traces,
  };
}
