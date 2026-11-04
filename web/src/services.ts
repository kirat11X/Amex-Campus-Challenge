import { Dataset, type ByteSource } from './data/dataset';
import type { EmbedConfig } from './data/types';
import { Embedder } from './rag/embedder';
import { Knowledge } from './rag/kb';
import { makeTools, type ToolFn } from './tools/tools';

export interface Services {
  ds: Dataset;
  kb: Knowledge;
  embedder: Embedder;
  contextMd: string | null;
  tools: Record<string, ToolFn>;
  /** The sidebar toggle "Include my uploaded files in answers". */
  prefs: { includeUser: boolean };
  /** Chat subscribes here to show long-running tool progress. */
  progress: { handler: ((msg: string) => void) | null };
}

export const fetchBytes: ByteSource = async (path) => {
  const url = import.meta.env.BASE_URL + path;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (HTTP ${r.status})`);
  return r.arrayBuffer();
};

export async function bootServices(bytes: ByteSource = fetchBytes): Promise<Services> {
  const embedCfg: EmbedConfig = JSON.parse(new TextDecoder().decode(await bytes('data/embed.json')));
  const embedder = new Embedder(embedCfg);
  const [ds, kb, contextMd] = await Promise.all([
    Dataset.load(bytes),
    Knowledge.load(bytes, embedder, embedCfg),
    bytes('data/context.md').then((b) => new TextDecoder().decode(b), () => null),
  ]);
  const prefs = { includeUser: kb.userChunkCount() > 0 };
  const progress: Services['progress'] = { handler: null };
  const tools = makeTools({
    ds,
    kb,
    userCorpusEnabled: () => prefs.includeUser,
    onFullTableProgress: (done, total) => progress.handler?.(`Downloading the customer table for this query… ${done}/${total}`),
  });
  // Warm the (small) embedding model in the background so the first question isn't slowed by it.
  embedder.init().catch(() => undefined);
  return { ds, kb, embedder, contextMd, tools, prefs, progress };
}

let booted: Promise<Services> | null = null;
/** One boot per page load (React StrictMode runs effects twice in dev). */
export const bootOnce = () => (booted ??= bootServices());
