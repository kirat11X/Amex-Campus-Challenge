import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [
    react(),
    {
      // transformers.js references its bundled ONNX wasm by URL, so Vite emits a 27 MB copy that this app
      // never loads (embed.worker.ts points the runtime at the self-hosted public/ort files instead).
      name: 'drop-unused-ort-wasm',
      generateBundle(_, bundle) {
        for (const name of Object.keys(bundle)) if (/assets\/ort-wasm-.*\.wasm$/.test(name)) delete bundle[name];
      },
    },
  ],
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 6500 },
  test: { environment: 'node', include: ['tests/**/*.test.ts'] },
});
