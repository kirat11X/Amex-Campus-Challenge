import { useEffect, useRef, useState } from 'react';
import { chatWithLLM, ContextOverflowError, type HistoryItem } from '../llm/agent';
import type { PromptMode } from '../llm/prompt';
import type { Engine } from '../llm/types';
import type { Services } from '../services';
import { Chart, chartsFromTraces, type ChartSpec } from './Charts';
import { Markdown } from './Markdown';

interface UiMessage {
  role: 'user' | 'assistant';
  content: string;
  charts?: ChartSpec[];
  tools?: string[];
  error?: boolean;
  pending?: boolean;
}

interface Props {
  services: Services;
  engine: Engine | null;
  systemPrompt: string;
  mode: PromptMode;
  useTools: boolean;
  queued: string | null;
  onQueuedConsumed: () => void;
}

export function Chat({ services, engine, systemPrompt, mode, useTools, queued, onQueuedConsumed }: Props) {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' }); // braces: newer Chrome returns a Promise, which React forbids as a cleanup
  }, [messages, status]);

  async function send(text: string) {
    text = text.trim();
    if (!text || !engine || busy) return;
    setInput('');
    setBusy(true);
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '', pending: true }]);
    const ac = new AbortController();
    abort.current = ac;
    services.progress.handler = setStatus;
    const patchLast = (patch: Partial<UiMessage>) =>
      setMessages((m) => m.map((x, i) => (i === m.length - 1 ? { ...x, ...patch } : x)));
    try {
      const out = await chatWithLLM(text, history, { engine, tools: services.tools, systemPrompt, mode }, {
        useTools,
        signal: ac.signal,
        onStatus: setStatus,
        onToken: (t) => patchLast({ content: t }),
      });
      setHistory(out.history);
      patchLast({
        content: out.reply,
        pending: false,
        tools: [...new Set(out.toolsUsed)],
        charts: chartsFromTraces(out.traces, services.ds.meta.term_meanings),
      });
    } catch (e) {
      const stopped = e instanceof DOMException && e.name === 'AbortError';
      const msg = stopped ? '*(stopped)*' : e instanceof ContextOverflowError ? e.message : `Error: ${e instanceof Error ? e.message : e}`;
      patchLast({ content: msg, pending: false, error: !stopped });
    } finally {
      services.progress.handler = null;
      abort.current = null;
      setStatus('');
      setBusy(false);
    }
  }

  useEffect(() => {
    if (queued && engine && !busy) {
      onQueuedConsumed();
      void send(queued);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queued, engine, busy]);

  return (
    <div className="chat">
      <div className="messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            {engine
              ? 'Ask about a customer, a segment, the formula or the challenge — or pick an example on the left.'
              : 'Load a model in the sidebar to start. It runs on your device; the first download is a few GB and is cached.'}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.error ? ' error' : ''}`}>
            <div className={`avatar ${m.role}`}>
              {m.role === 'user' ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A1.5 1.5 0 0 0 6 14.5a1.5 1.5 0 0 0 3 0A1.5 1.5 0 0 0 7.5 13m9 0a1.5 1.5 0 0 0-1.5 1.5 1.5 1.5 0 0 0 3 0 1.5 1.5 0 0 0-1.5-1.5"/>
                </svg>
              )}
            </div>
            <div className="bubble">
              {m.content ? (m.role === 'assistant' ? <Markdown text={m.content} /> : <p>{m.content}</p>) : null}
              {m.pending && <div className="status">{status || 'Thinking…'}</div>}
              {m.charts?.map((c, j) => <Chart key={j} spec={c} />)}
              {m.tools && m.tools.length > 0 && <div className="tools">🛠 {m.tools.join(' · ')}</div>}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <form className="composer" onSubmit={(e) => { e.preventDefault(); void send(input); }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about customer profitability..."
          disabled={!engine}
          aria-label="Message"
        />
        {busy ? (
          <button type="button" className="secondary" onClick={() => abort.current?.abort()}>Stop</button>
        ) : (
          <button type="submit" className="primary circle-btn" disabled={!engine || !input.trim()} title="Send message">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        )}
      </form>
    </div>
  );
}
