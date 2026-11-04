import { useCallback, useEffect, useRef, useState } from 'react';
import type { Engine } from '../llm/types';
import { checkOllama, DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaEngine } from '../llm/ollama';
import { detectGpu, modelId, type GpuInfo } from '../llm/gpu';
import type { WebLLMEngine } from '../llm/webllm';
import { DemoEngine } from '../llm/demo';

// The WebLLM runtime is ~2 MB gzipped: fetched only on devices that can use it.
const webllmModule = () => import('../llm/webllm');
const isCached = async (id: string) => (await webllmModule()).isCached(id);

export type EngineKind = 'demo' | 'webllm' | 'ollama';

export interface Prefs {
  engine: EngineKind;
  base: string; // WebLLM model family, e.g. "Qwen3-4B"
  ollamaUrl: string;
  ollamaModel: string;
  useTools: boolean;
}

const KEY = 'profitlens.prefs.v2';
const DEFAULTS: Prefs = { engine: 'demo', base: 'Qwen3-4B', ollamaUrl: DEFAULT_OLLAMA_URL, ollamaModel: DEFAULT_OLLAMA_MODEL, useTools: true };

export function loadPrefs(): Prefs {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') };
  } catch {
    return DEFAULTS;
  }
}

export function savePrefs(p: Prefs) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch { /* storage blocked — preferences just won't persist */ }
}

export interface EngineState {
  phase: 'idle' | 'loading' | 'ready' | 'error';
  engine: Engine | null;
  progress: number;
  text: string;
  error: string;
}

const IDLE: EngineState = { phase: 'idle', engine: null, progress: 0, text: '', error: '' };

export function useEngine(prefs: Prefs) {
  const [state, setState] = useState<EngineState>(() =>
    prefs.engine === 'demo' ? { ...IDLE, phase: 'ready', engine: new DemoEngine() } : IDLE,
  );
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [cached, setCached] = useState<Record<string, boolean>>({});
  const webllm = useRef<WebLLMEngine | null>(null);

  useEffect(() => {
    detectGpu().then(setGpu);
  }, []);

  // Which models are already in the browser cache (so "Load" means seconds, not gigabytes).
  useEffect(() => {
    if (!gpu?.available) return;
    const bases = ['Qwen3-4B', 'Qwen3-1.7B', 'Qwen3-0.6B'];
    Promise.all(bases.map(async (b) => [b, await isCached(modelId(b, gpu.f16))] as const)).then((e) => setCached(Object.fromEntries(e)));
  }, [gpu, state.phase]);

  const loadDemo = useCallback(() => {
    void webllm.current?.unload().catch(() => undefined);
    webllm.current = null;
    setState({ ...IDLE, phase: 'ready', engine: new DemoEngine() });
  }, []);

  const loadWebLLM = useCallback(async (base: string) => {
    if (!gpu?.available) return;
    await webllm.current?.unload().catch(() => undefined);
    webllm.current = null;
    setState({ ...IDLE, phase: 'loading', text: 'Starting…' });
    try {
      const engine = await (await webllmModule()).WebLLMEngine.create(modelId(base, gpu.f16), (p) =>
        setState((s) => ({ ...s, phase: 'loading', progress: p.progress, text: p.text })));
      webllm.current = engine;
      setState({ ...IDLE, phase: 'ready', engine });
    } catch (e) {
      setState({ ...IDLE, phase: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  }, [gpu]);

  const connectOllama = useCallback(async (url: string, model: string) => {
    setState({ ...IDLE, phase: 'loading', text: `Contacting ${url}…` });
    const check = await checkOllama(url, model);
    if (!check.ok) {
      setState({ ...IDLE, phase: 'error', error: check.message });
      return;
    }
    setState({ ...IDLE, phase: 'ready', engine: new OllamaEngine(url, model), text: check.message });
  }, []);

  const disconnect = useCallback(async () => {
    await webllm.current?.unload().catch(() => undefined);
    webllm.current = null;
    setState(IDLE);
  }, []);

  // Auto-resume: a model that is already cached loads without a download; Ollama just reconnects.
  const autoStarted = useRef(false);
  useEffect(() => {
    if (autoStarted.current) return;
    autoStarted.current = true;
    if (prefs.engine === 'demo') {
      loadDemo();
    } else if (prefs.engine === 'ollama') {
      void connectOllama(prefs.ollamaUrl, prefs.ollamaModel);
    } else if (gpu?.available) {
      void isCached(modelId(prefs.base, gpu.f16)).then((hit) => { if (hit) void loadWebLLM(prefs.base); });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpu]);

  return { state, gpu, cached, loadDemo, loadWebLLM, connectOllama, disconnect };
}
