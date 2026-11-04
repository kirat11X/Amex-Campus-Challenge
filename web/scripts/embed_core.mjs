// Pre-embeds the core knowledge-base chunks with the *same* ONNX model the browser
// loads at runtime, so only user questions/uploads are embedded client-side.
//
//   npm run embed-core        (after scripts/export_data.py wrote core_chunks.json)
//
// Writes public/data/core_vectors.bin (float32, N x dim, L2-normalised) and
// public/data/embed.json (the model settings the browser reads).
import { pipeline } from '@huggingface/transformers';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data');

// bge-small-en-v1.5: 384-d, ~34 MB quantised. CLS pooling; queries (not documents)
// take the instruction prefix. Scores on this corpus: relevant 0.62-0.80, off-topic <= 0.52.
export const EMBED = {
  model: 'Xenova/bge-small-en-v1.5',
  dtype: 'q8',
  pooling: 'cls',
  query_prefix: 'Represent this sentence for searching relevant passages: ',
  sim_floor: 0.58,
};

const chunksRaw = readFileSync(join(DATA, 'core_chunks.json'), 'utf8');
const chunks = JSON.parse(chunksRaw);

const extractor = await pipeline('feature-extraction', EMBED.model, { dtype: EMBED.dtype });
// One text per call: the q8 model quantises activations per batch, so batching (padding) nudges the
// vectors by ~1%. Single calls match how queries are embedded at runtime.
const vectors = [];
for (const c of chunks) {
  const out = await extractor([c.text], { pooling: EMBED.pooling, normalize: true });
  vectors.push(...out.tolist());
}
const dim = vectors[0].length;
const flat = new Float32Array(vectors.length * dim);
vectors.forEach((v, i) => flat.set(v, i * dim));

writeFileSync(join(DATA, 'core_vectors.bin'), Buffer.from(flat.buffer));
writeFileSync(join(DATA, 'embed.json'), JSON.stringify({
  ...EMBED, dim, count: vectors.length,
  chunks_sha1: createHash('sha1').update(chunksRaw).digest('hex'),
}, null, 1));
console.log(`embedded ${vectors.length} chunks x ${dim} with ${EMBED.model} (${EMBED.dtype})`);
