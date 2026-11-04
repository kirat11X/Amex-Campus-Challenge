/** Light-weight WebGPU / model-catalogue helpers (kept apart from webllm.ts so the 6 MB WebLLM chunk loads lazily). */
/** Context window requested from WebLLM. The prebuilt libs default to 4096; the paged KV cache allows more. */
export const WEBLLM_CONTEXT = 8192;

export interface WebLLMModel {
  /** model ids without the quantisation suffix, e.g. "Qwen3-4B" */
  base: string;
  label: string;
  /** approximate one-time download */
  download: string;
  /** approximate GPU memory at the 8K context */
  vram: string;
  blurb: string;
}

// Qwen3 is the family the Streamlit app runs (via Ollama); these are its prebuilt MLC builds.
export const WEBLLM_MODELS: WebLLMModel[] = [
  { base: 'Qwen3-4B', label: 'Qwen3 4B', download: '≈2.3 GB', vram: '≈4.2 GB',
    blurb: 'Closest to the Ollama model. Best answers and tool use. Needs a mid-range GPU.' },
  { base: 'Qwen3-1.7B', label: 'Qwen3 1.7B', download: '≈1.1 GB', vram: '≈2.3 GB',
    blurb: 'Lighter GPUs and integrated graphics. Fine for lookups, weaker at multi-step questions.' },
  { base: 'Qwen3-0.6B', label: 'Qwen3 0.6B', download: '≈0.4 GB', vram: '≈1.5 GB',
    blurb: 'Smoke-test only: loads anywhere WebGPU runs, but answers are unreliable.' },
];

export interface GpuInfo {
  available: boolean;
  f16: boolean;
  description: string;
  reason?: string;
}

export async function detectGpu(): Promise<GpuInfo> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<GPUAdapterLike | null> } }).gpu;
  if (!gpu) return { available: false, f16: false, description: '', reason: 'This browser has no WebGPU (navigator.gpu is undefined).' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { available: false, f16: false, description: '', reason: 'WebGPU is present but no GPU adapter was found.' };
    const info = adapter.info ?? {};
    const description = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'GPU';
    return { available: true, f16: adapter.features.has('shader-f16'), description };
  } catch (e) {
    return { available: false, f16: false, description: '', reason: `WebGPU adapter request failed: ${e instanceof Error ? e.message : e}` };
  }
}

interface GPUAdapterLike {
  features: { has(f: string): boolean };
  info?: { vendor?: string; architecture?: string; description?: string };
}

export const modelId = (base: string, f16: boolean) => `${base}-${f16 ? 'q4f16_1' : 'q4f32_1'}-MLC`;
