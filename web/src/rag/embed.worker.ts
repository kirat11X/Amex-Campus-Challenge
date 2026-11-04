/// <reference lib="webworker" />
// Sentence embeddings with transformers.js (ONNX Runtime, wasm), off the main thread.
import { env, pipeline } from '@huggingface/transformers';

type Req =
  | { id: number; type: 'init'; model: string; dtype: string; ortBase: string }
  | { id: number; type: 'embed'; texts: string[]; pooling: 'cls' | 'mean' };

let extractor: ((t: string[], o: Record<string, unknown>) => Promise<{ data: Float32Array; dims: number[] }>) | null = null;

self.onmessage = async (ev: MessageEvent<Req>) => {
  const req = ev.data;
  try {
    if (req.type === 'init') {
      env.allowLocalModels = false;
      // Self-hosted runtime (scripts/copy-ort.mjs); only the model files come from Hugging Face.
      env.backends.onnx.wasm!.wasmPaths = {
        mjs: `${req.ortBase}ort-wasm-simd-threaded.mjs`,
        wasm: `${req.ortBase}ort-wasm-simd-threaded.wasm`,
      };
      extractor = (await pipeline('feature-extraction', req.model, {
        dtype: req.dtype as 'q8',
        device: 'wasm',
        progress_callback: (p: unknown) => self.postMessage({ type: 'progress', progress: p }),
      })) as unknown as typeof extractor;
      self.postMessage({ id: req.id, ok: true });
    } else {
      if (!extractor) throw new Error('embedder not initialised');
      const out = await extractor(req.texts, { pooling: req.pooling, normalize: true });
      const dim = out.dims[out.dims.length - 1];
      const vectors = req.texts.map((_, i) => out.data.slice(i * dim, (i + 1) * dim));
      self.postMessage({ id: req.id, ok: true, vectors }, vectors.map((v) => v.buffer));
    }
  } catch (e) {
    self.postMessage({ id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
};
