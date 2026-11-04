import { useState } from 'react';
import { WEBLLM_MODELS } from '../llm/gpu';
import type { useEngine, Prefs } from './useEngine';

type Hook = ReturnType<typeof useEngine>;

export function ModelPanel({ hook, prefs, setPrefs }: { hook: Hook; prefs: Prefs; setPrefs: (p: Prefs) => void }) {
  const { state, gpu, cached, loadDemo, loadWebLLM, connectOllama, disconnect } = hook;
  const [helpOpen, setHelpOpen] = useState(false);
  const loading = state.phase === 'loading';
  const set = (patch: Partial<Prefs>) => setPrefs({ ...prefs, ...patch });
  const gpuMissing = gpu !== null && !gpu.available;
  const tab = prefs.engine;

  return (
    <section className="panel" aria-label="Model">
      <div className="seg" role="tablist" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
        <button role="tab" aria-selected={tab === 'demo'} onClick={() => { set({ engine: 'demo' }); loadDemo(); }}>Demo</button>
        <button role="tab" aria-selected={tab === 'webllm'} disabled={gpuMissing} onClick={() => set({ engine: 'webllm' })}>Browser</button>
        <button role="tab" aria-selected={tab === 'ollama'} onClick={() => set({ engine: 'ollama' })}>Ollama</button>
      </div>

      {gpuMissing && tab === 'webllm' && state.phase !== 'ready' && (
        <p className="warn">{gpu?.reason} The in-browser model needs WebGPU; you can use Demo mode or your own Ollama instead.</p>
      )}

      {state.phase === 'ready' && state.engine ? (
        <div className="ready">
          <div className="badge-ok">● {state.engine.label}</div>
          {state.engine.kind !== 'demo' && (
            <button className="link" onClick={disconnect}>{state.engine.kind === 'webllm' ? 'Unload model' : 'Disconnect'}</button>
          )}
        </div>
      ) : tab === 'demo' ? (
        <>
          <p className="muted small">Interactive instant demo for Vercel: explores customer profiles, P&amp;L waterfall charts, formula, and distributions with zero download.</p>
          <button className="primary" onClick={loadDemo}>Activate Demo Mode</button>
        </>
      ) : tab === 'webllm' ? (
        <>
          {gpu === null && <p className="muted">Checking WebGPU…</p>}
          {gpu?.available && (
            <p className="muted">
              ✓ WebGPU: {gpu.description}. The model runs on <b>your</b> GPU; nothing is sent to a server.
              {!gpu.f16 && ' (No shader-f16 support — using the larger f32 build.)'}
            </p>
          )}
          {gpu?.available && (
            <>
              <div className="models">
                {WEBLLM_MODELS.map((m) => (
                  <label key={m.base} className={prefs.base === m.base ? 'model on' : 'model'}>
                    <input type="radio" name="model" checked={prefs.base === m.base} disabled={loading} onChange={() => set({ base: m.base })} />
                    <span>
                      <b>{m.label}</b> <small>{cached[m.base] ? '✓ downloaded' : `${m.download} download`} · {m.vram} GPU</small>
                      <small className="blurb">{m.blurb}</small>
                    </span>
                  </label>
                ))}
              </div>
              <button className="primary" disabled={loading} onClick={() => loadWebLLM(prefs.base)}>
                {loading ? 'Loading…' : cached[prefs.base] ? 'Load model' : 'Download & load model'}
              </button>
            </>
          )}
        </>
      ) : (
        <>
          <label className="field">Ollama URL
            <input value={prefs.ollamaUrl} onChange={(e) => set({ ollamaUrl: e.target.value })} spellCheck={false} />
          </label>
          <label className="field">Model
            <input value={prefs.ollamaModel} onChange={(e) => set({ ollamaModel: e.target.value })} spellCheck={false} />
          </label>
          <button className="primary" disabled={loading} onClick={() => connectOllama(prefs.ollamaUrl, prefs.ollamaModel)}>
            {loading ? 'Connecting…' : 'Connect'}
          </button>
          <button className="link" onClick={() => setHelpOpen(!helpOpen)}>{helpOpen ? 'Hide' : 'Show'} setup help</button>
          {helpOpen && (
            <div className="help">
              <p>Your browser talks straight to Ollama on your machine — the inference stays local.</p>
              <ol>
                <li>Install Ollama and run <code>ollama pull {prefs.ollamaModel}</code>.</li>
                <li>Allow this page to call it, then start the server:
                  <pre>{`OLLAMA_ORIGINS="${location.origin}" ollama serve`}</pre>
                  Pages on <code>localhost</code> are allowed by default.</li>
                <li>Chrome may ask to “access devices on your local network” — allow it.</li>
              </ol>
            </div>
          )}
        </>
      )}

      {loading && (
        <div className="progress" role="status">
          <div className="bar"><div style={{ width: `${Math.round(state.progress * 100)}%` }} /></div>
          <small>{state.text}</small>
        </div>
      )}
      {state.phase === 'error' && <p className="warn" role="alert">{state.error}</p>}
    </section>
  );
}
