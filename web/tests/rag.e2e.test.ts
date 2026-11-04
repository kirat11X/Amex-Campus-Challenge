/**
 * Retrieval quality with the real embedding model (Node's onnxruntime — the same quantised ONNX the
 * browser runs under wasm). Needs the model files (downloaded once from Hugging Face), so it is opt-in:
 *
 *   RAG_E2E=1 npx vitest run tests/rag.e2e.test.ts
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { pipeline } from '@huggingface/transformers';
import { VectorStore } from '../src/rag/store';
import type { Chunk } from '../src/rag/types';
import { diskBytes } from './helpers';

const run = process.env.RAG_E2E ? describe : describe.skip;

run('retrieval over the core corpus', () => {
  let embed: (t: string[]) => Promise<Float32Array[]>;
  let store: VectorStore;
  let chunks: Chunk[];
  let cfg: { sim_floor: number; query_prefix: string; model: string; dtype: string; pooling: string; dim: number };

  beforeAll(async () => {
    cfg = JSON.parse(new TextDecoder().decode(await diskBytes('data/embed.json')));
    chunks = JSON.parse(new TextDecoder().decode(await diskBytes('data/core_chunks.json')));
    const flat = new Float32Array(await diskBytes('data/core_vectors.bin'));
    store = new VectorStore(cfg.dim);
    store.add(chunks.map((_, i) => flat.slice(i * cfg.dim, (i + 1) * cfg.dim)), chunks);
    const ex = await pipeline('feature-extraction', cfg.model, { dtype: cfg.dtype as 'q8' });
    embed = async (texts) => {
      const out = await ex(texts, { pooling: cfg.pooling as 'cls', normalize: true });
      return texts.map((_, i) => out.data.slice(i * cfg.dim, (i + 1) * cfg.dim) as Float32Array);
    };
  }, 120_000);

  const ask = async (q: string, k = 3) => store.search((await embed([cfg.query_prefix + q]))[0], k, cfg.sim_floor);

  it('the shipped vectors are what the model produces for the shipped chunks', async () => {
    // one text per call, exactly as scripts/embed_core.mjs produced them
    const shipped = store.all().vectors;
    for (const i of [0, 1, 2, 3, 4]) {
      const [v] = await embed([chunks[i].text]);
      const cos = v.reduce((s, x, j) => s + x * shipped[i][j], 0);
      expect(cos).toBeGreaterThan(0.9999);
    }
  });

  const cases: [string, string][] = [
    ['what is the profit formula weight for lounge', 'cost_lounge'],
    ['how were the weights derived methodology', 'weights were derived'],
    ['what was the evaluation criterion of the challenge', 'CHALLENGE'],
    ['how many submissions were allowed per team', 'campus_challenge26'],
    ['how does the code handle missing values', 'missing-value'],
    ['profit distribution percentiles', 'profit distribution'],
    ['why is the collections coefficient 8500', 'collections'],
  ];
  for (const [q, needle] of cases) {
    it(`finds "${q}"`, async () => {
      const hits = await ask(q, 3);
      expect(hits.length).toBeGreaterThan(0);
      expect(hits.some((h) => `${h.source} ${h.location} ${h.text}`.toLowerCase().includes(needle.toLowerCase()))).toBe(true);
    });
  }

  it.each(['what is the weather in paris', 'recipe for chocolate cake', 'tell me a joke about football'])(
    'returns nothing above the floor for off-topic "%s"', async (q) => {
      expect(await ask(q, 5)).toEqual([]);
    });
});
