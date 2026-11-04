import { CreateWebWorkerMLCEngine, hasModelInCache, type InitProgressReport, type WebWorkerMLCEngine } from '@mlc-ai/web-llm';
import { WEBLLM_CONTEXT } from './gpu';
import { formatToolCalls, parseToolCalls, toolsBlock } from './prompt';
import type { ChatMessage, CompleteOptions, CompletionResult, Engine } from './types';

export const isCached = (id: string) => hasModelInCache(id).catch(() => false);

export class WebLLMEngine implements Engine {
  readonly kind = 'webllm' as const;
  readonly contextTokens = WEBLLM_CONTEXT;
  readonly label: string;

  private constructor(private engine: WebWorkerMLCEngine, readonly id: string) {
    this.label = `${id.replace(/-MLC$/, '')} (in your browser)`;
  }

  static async create(id: string, onProgress: (p: InitProgressReport) => void): Promise<WebLLMEngine> {
    const worker = new Worker(new URL('./webllm.worker.ts', import.meta.url), { type: 'module' });
    const engine = await CreateWebWorkerMLCEngine(
      worker,
      id,
      { initProgressCallback: onProgress },
      { context_window_size: WEBLLM_CONTEXT },
    );
    return new WebLLMEngine(engine, id);
  }

  async unload(): Promise<void> {
    await this.engine.unload();
  }

  async complete(messages: ChatMessage[], opts: CompleteOptions = {}): Promise<CompletionResult> {
    // Qwen's chat template has no tool role: tool results are user turns wrapped in <tool_response>,
    // and tool calls are re-serialised as <tool_call> text in the assistant turn.
    const wire: { role: 'system' | 'user' | 'assistant'; content: string }[] = [];
    for (const m of messages) {
      if (m.role === 'system') {
        wire.push({ role: 'system', content: m.content + (opts.tools?.length ? toolsBlock(opts.tools) : '') });
      } else if (m.role === 'tool') {
        const block = `<tool_response>\n${m.content}\n</tool_response>`;
        const last = wire[wire.length - 1];
        if (last?.role === 'user' && last.content.startsWith('<tool_response>')) last.content += `\n${block}`;
        else wire.push({ role: 'user', content: block });
      } else if (m.role === 'assistant' && m.tool_calls?.length) {
        wire.push({ role: 'assistant', content: [m.content, formatToolCalls(m.tool_calls)].filter(Boolean).join('\n') });
      } else {
        wire.push({ role: m.role, content: m.content });
      }
    }

    const onAbort = () => this.engine.interruptGenerate();
    opts.signal?.addEventListener('abort', onAbort);
    try {
      const stream = await this.engine.chat.completions.create({
        messages: wire,
        stream: true,
        temperature: 0.2,
        max_tokens: 1024,
        // Qwen3 is a hybrid thinking model; the Ollama build (Instruct-2507) does not think.
        extra_body: { enable_thinking: false },
      });
      let text = '';
      for await (const chunk of stream) {
        text += chunk.choices[0]?.delta?.content ?? '';
        if (opts.onToken) opts.onToken(text.split('<tool_call>')[0].replace(/<think>[\s\S]*?(<\/think>|$)/, '').trimStart());
      }
      if (opts.signal?.aborted) throw new DOMException('Stopped', 'AbortError');
      const { content, calls } = parseToolCalls(text);
      return { content, toolCalls: opts.tools?.length ? calls : [] };
    } finally {
      opts.signal?.removeEventListener('abort', onAbort);
    }
  }
}
