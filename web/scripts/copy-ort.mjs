// Self-host the ONNX Runtime wasm that transformers.js needs for embeddings, so the
// deployed site makes no CDN requests besides the Hugging Face model files.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dst = join(root, 'public', 'ort');
mkdirSync(dst, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  if (!existsSync(join(src, f))) throw new Error(`missing ${f} in onnxruntime-web/dist (run npm install)`);
  copyFileSync(join(src, f), join(dst, f));
}
console.log('copied ONNX runtime wasm -> public/ort');
