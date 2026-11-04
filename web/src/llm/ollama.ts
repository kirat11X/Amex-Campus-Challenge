import type { ChatMessage, CompleteOptions, CompletionResult, Engine, ToolCall } from './types';

export const DEFAULT_OLLAMA_URL = 'http://localhost:11434';
export const DEFAULT_OLLAMA_MODEL = 'qwen3:4b-instruct-2507-q4_K_M';
export const OLLAMA_NUM_CTX = 8192; // same window as the Streamlit app

const trimUrl = (u: string) => u.trim().replace(/\/+$/, '');

export interface OllamaCheck {
  ok: boolean;
  models: string[];
  message: string;
}

/** Mirror of llm.check_ollama(): is the server reachable (from this browser) and the model pulled? */
export async function checkOllama(baseUrl: string, model: string): Promise<OllamaCheck> {
  const url = trimUrl(baseUrl);
  try {
    const r = await fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const models: string[] = ((await r.json()).models ?? []).map((m: { name: string }) => m.name);
    if (!models.includes(model) && !models.includes(`${model}:latest`)) {
      return { ok: false, models, message: `Model \`${model}\` is not installed. Run \`ollama pull ${model}\`.` };
    }
    return { ok: true, models, message: `Connected — ${model} is available.` };
  } catch (e) {
    return {
      ok: false,
      models: [],
      message:
        `Could not reach Ollama at ${url} from this page (${e instanceof Error ? e.message : e}). ` +
        'Is it running (`ollama serve`)? If this site is not served from localhost, start Ollama with ' +
        `OLLAMA_ORIGINS=${location.origin} so the browser is allowed to call it.`,
    };
  }
}

export class OllamaEngine implements Engine {
  readonly kind = 'ollama' as const;
  readonly contextTokens = OLLAMA_NUM_CTX;
  readonly label: string;

  constructor(private baseUrl: string, private model: string) {
    this.baseUrl = trimUrl(baseUrl);
    this.label = `${model} (your Ollama)`;
  }

  async complete(messages: ChatMessage[], opts: CompleteOptions = {}): Promise<CompletionResult> {
    const body: Record<string, unknown> = {
      model: this.model,
      stream: true,
      options: { temperature: 0.2, num_ctx: OLLAMA_NUM_CTX },
      messages: messages.map((m) => {
        if (m.role === 'tool') return { role: 'tool', tool_name: m.tool_name, content: m.content };
        if (m.role === 'assistant' && m.tool_calls?.length) {
          return { role: 'assistant', content: m.content, tool_calls: m.tool_calls.map((c) => ({ function: { name: c.name, arguments: c.arguments } })) };
        }
        return { role: m.role, content: m.content };
      }),
    };
    if (opts.tools?.length) body.tools = opts.tools;

    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) throw new Error(`Ollama request failed: HTTP ${res.status} ${(await res.text().catch(() => '')).slice(0, 200)}`);

    let content = '';
    const toolCalls: ToolCall[] = [];
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    const handle = (line: string) => {
      if (!line.trim()) return;
      const j = JSON.parse(line);
      if (j.error) throw new Error(`Ollama: ${j.error}`);
      const msg = j.message ?? {};
      if (msg.content) {
        content += msg.content;
        opts.onToken?.(content);
      }
      for (const c of msg.tool_calls ?? []) {
        let args = c.function?.arguments ?? {};
        if (typeof args === 'string') {
          try { args = JSON.parse(args); } catch { args = {}; }
        }
        toolCalls.push({ name: c.function?.name ?? '', arguments: args });
      }
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        handle(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    }
    handle(buf);
    return { content: content.replace(/<think>[\s\S]*?<\/think>/g, '').trim(), toolCalls };
  }
}
