import { useEffect, useMemo, useState } from 'react';
import { buildSystemPrompt, type PromptMode } from './llm/prompt';
import { bootOnce, type Services } from './services';
import { Chat } from './ui/Chat';
import { Digest } from './ui/Digest';
import { Sidebar } from './ui/Sidebar';
import { loadPrefs, savePrefs, useEngine, type Prefs } from './ui/useEngine';

export function App() {
  const [services, setServices] = useState<Services | null>(null);
  const [bootError, setBootError] = useState('');
  useEffect(() => {
    bootOnce().then(setServices, (e) => setBootError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (bootError) return <div className="boot error">Failed to load the app data: {bootError}</div>;
  if (!services) return <div className="boot">Loading the customer index and knowledge base…</div>;
  return <Main services={services} />;
}

function Main({ services }: { services: Services }) {
  const [prefs, setPrefsState] = useState<Prefs>(loadPrefs);
  const setPrefs = (p: Prefs) => { setPrefsState(p); savePrefs(p); };
  const hook = useEngine(prefs);
  const engine = hook.state.engine;

  const [tab, setTab] = useState<'chat' | 'digest'>('chat');
  const [includeUser, setIncludeUserState] = useState(services.prefs.includeUser);
  const setIncludeUser = (v: boolean) => { services.prefs.includeUser = v; setIncludeUserState(v); };
  useEffect(() => services.kb.subscribe(() => setIncludeUserState(services.prefs.includeUser)), [services]);

  const [promptMode, setPromptMode] = useState<'auto' | 'compact' | 'full'>('auto');
  const mode: PromptMode = promptMode === 'auto' ? (engine?.kind === 'ollama' ? 'full' : 'compact') : promptMode;
  const systemPrompt = useMemo(
    () => buildSystemPrompt(services.contextMd, services.ds.meta.runtime.block, mode),
    [services, mode],
  );

  const [queued, setQueued] = useState<string | null>(null);
  const [chatKey, setChatKey] = useState(0);

  return (
    <div className="app">
      <Sidebar
        services={services} hook={hook} prefs={prefs} setPrefs={setPrefs}
        includeUser={includeUser} setIncludeUser={setIncludeUser}
        canAsk={engine !== null}
        onExample={(t) => { setTab('chat'); setQueued(t); }}
        onReset={() => setChatKey((k) => k + 1)}
        promptMode={promptMode} setPromptMode={setPromptMode}
      />
      <main>
        <h1>ProfitLens</h1>
        <p className="muted">Premier Card Profitability Chatbot — powered by a local LLM (Ollama)</p>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'chat'} onClick={() => setTab('chat')}>💬 Chat</button>
          <button role="tab" aria-selected={tab === 'digest'} onClick={() => setTab('digest')}>📂 Digest your own files?</button>
        </div>
        <div hidden={tab !== 'chat'}>
          <Chat key={chatKey} services={services} engine={engine} systemPrompt={systemPrompt} mode={mode}
            useTools={prefs.useTools} queued={queued} onQueuedConsumed={() => setQueued(null)} />
        </div>
        <div hidden={tab !== 'digest'}><Digest services={services} /></div>
      </main>
    </div>
  );
}
