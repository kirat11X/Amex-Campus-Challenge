/**
 * The eight LLM tools, ported from profitlens/tools.py.
 *
 * Output shapes are identical to the Python tools. Most answers are precomputed
 * exactly (by scripts/export_data.py) over all 500K customers; customer lookups
 * read one shard; ad-hoc filters and custom histograms download the full table
 * once (~23 MB, cached) and run here.
 */
import type { Dataset, FullTable } from '../data/dataset';
import { FEATURES, round } from '../data/dataset';
import type { ToolResult } from '../data/types';
import type { KnowledgeBase } from '../rag/types';
import { compileQuery, QueryError } from './query';
import { dropNaN, histogram, mean, percentileSorted, sum } from './stats';

const UNSAFE = /(@|__|\bimport\b|\bexec\b|\beval\b|\bos\b|\bsys\b)/;
const AGGS = ['count', 'mean_profit', 'sum_profit', 'top20_share'];

export interface ToolContext {
  ds: Dataset;
  kb: KnowledgeBase;
  /** The sidebar toggle "Include my uploaded files in answers". */
  userCorpusEnabled: () => boolean;
  /** Progress of the one-time full-table download. */
  onFullTableProgress?: (done: number, total: number) => void;
}

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));
const num = (v: unknown, dflt: number) => {
  const n = typeof v === 'string' ? Number(v) : (v as number);
  return Number.isFinite(n) ? n : dflt;
};

export type ToolFn = (args: Record<string, unknown>) => Promise<ToolResult>;

export function makeTools(ctx: ToolContext): Record<string, ToolFn> {
  const { ds } = ctx;
  const meta = ds.meta;
  const validColumns = meta.valid_columns;
  const describe = (c: string) => meta.feature_dict[c] ?? meta.term_meanings[c] ?? c;
  const table = (): Promise<FullTable> => ds.fullTable(ctx.onFullTableProgress);

  return {
    async get_customer_profile(args) {
      const id = Math.trunc(num(args.customer_id, NaN));
      const profile = Number.isNaN(id) ? null : await ds.customer(id);
      if (!profile) {
        return {
          error: `Customer ${args.customer_id} not found.`,
          hint: `Valid IDs range from ${meta.id_min} to ${meta.id_max}.`,
        };
      }
      return profile as unknown as ToolResult;
    },

    async get_top_customers(args) {
      const n = Math.max(1, Math.min(Math.trunc(num(args.n, 10)), 50));
      const ascending = args.ascending === true || args.ascending === 'true';
      const src = ds.staticTools.top_customers[ascending ? 'bottom' : 'top'].customers;
      return { customers: clone(src.slice(0, n)), n, ascending };
    },

    async segment_compare(args) {
      const feature = String(args.feature);
      const hit = ds.staticTools.segment_compare[feature];
      if (!validColumns.includes(feature) || !hit) {
        return { error: `Invalid feature '${feature}'.`, hint: `Valid names: ${JSON.stringify(validColumns)}` };
      }
      return clone(hit);
    },

    async term_contribution_summary() {
      return clone(ds.staticTools.term_contribution_summary);
    },

    async get_distribution(args) {
      const column = String(args.column);
      if (!validColumns.includes(column)) {
        return { error: `Invalid column '${column}'.`, hint: `Valid names: ${JSON.stringify(validColumns)}` };
      }
      const bins = Math.trunc(num(args.bins, 40));
      const clipArg = Array.isArray(args.clip_percentiles) ? (args.clip_percentiles as unknown[]).map((x) => num(x, NaN)) : null;
      const clip = clipArg && clipArg.length === 2 && clipArg.every(Number.isFinite) ? clipArg : [1, 99];
      if (bins === 40 && clip[0] === 1 && clip[1] === 99) return clone(ds.staticTools.distributions[column]);
      if (bins < 1 || bins > 500) return { error: `Invalid bins ${bins}.`, hint: 'Use 1-500 bins.' };

      const raw = (await table()).col(column);
      const s = dropNaN(raw);
      const sorted = Float64Array.from(s).sort();
      const lo = percentileSorted(sorted, clip[0]);
      const hi = percentileSorted(sorted, clip[1]);
      const clipped = s.map((v) => Math.min(Math.max(v, lo), hi));
      const { counts, edges } = histogram(clipped, bins);
      return {
        column,
        description: describe(column),
        bin_edges: edges.map((e) => round(e, 4)),
        counts,
        clip: [lo, hi],
        total_values: s.length,
        nan_count: raw.length - s.length,
      };
    },

    async query_segment(args) {
      const filters = String(args.filters ?? '');
      const agg = String(args.agg ?? '');
      if (!AGGS.includes(agg)) return { error: `Invalid agg '${agg}'.`, hint: `Choose from ${AGGS.join(', ')}` };
      if (UNSAFE.test(filters)) {
        return { error: 'Unsafe query pattern detected.', hint: "Only use column comparisons, e.g. 'f3 >= 1 and profit > 0'." };
      }
      const t = await table();
      let keep: Uint8Array;
      try {
        const pred = compileQuery(filters, (name) => {
          if (!t.has(name)) throw new Error(name);
          return t.col(name);
        });
        keep = new Uint8Array(t.n);
        for (let i = 0; i < t.n; i++) keep[i] = pred(i) !== 0 ? 1 : 0;
      } catch (e) {
        const msg = e instanceof QueryError ? e.message : String(e);
        return { error: `Query failed: ${msg}`, hint: 'Use pandas query syntax over columns: ' + validColumns.join(', ') };
      }

      const profit = t.col('profit');
      const top = t.col('is_top20');
      let count = 0;
      let top20 = 0;
      const picked: number[] = [];
      for (let i = 0; i < t.n; i++) {
        if (!keep[i]) continue;
        count++;
        top20 += top[i];
        picked.push(profit[i]);
      }
      let result: Record<string, unknown>;
      if (agg === 'count') result = { count };
      else if (agg === 'mean_profit') result = { mean_profit: count ? round(mean(picked), 4) : null, count };
      else if (agg === 'sum_profit') result = { sum_profit: round(sum(picked), 4), count };
      else result = { top20_count: top20, total: count, top20_share: count ? round((top20 / count) * 100, 2) : null };
      return { filters, agg, ...result };
    },

    async explain_formula() {
      return clone(ds.staticTools.explain_formula);
    },

    async search_knowledge_base(args) {
      let corpus = String(args.corpus ?? 'both').toLowerCase();
      if (!['core', 'user', 'both'].includes(corpus)) corpus = 'both';
      const k = Math.max(1, Math.min(Math.trunc(num(args.k, 5)), 10));
      const query = String(args.query ?? '');

      const userRequested = corpus === 'user' || corpus === 'both';
      const enabled = ctx.userCorpusEnabled();
      if (!enabled) corpus = 'core';

      const notes: string[] = [];
      if (userRequested && !enabled) {
        notes.push("your uploaded files were NOT consulted — the sidebar toggle 'Include my uploaded files in answers' is off; " +
          'tell the user this and that turning it on would let you search them');
      } else if (userRequested && ctx.kb.userChunkCount() === 0) {
        notes.push('no user files ingested yet');
      }

      let hits: Awaited<ReturnType<KnowledgeBase['search']>> = [];
      try {
        if (corpus === 'core' || corpus === 'both') hits = hits.concat(await ctx.kb.search(query, 'core', k));
        if (corpus === 'user' || corpus === 'both') hits = hits.concat(await ctx.kb.search(query, 'user', k));
      } catch (e) {
        return { results: [], note: `retrieval unavailable (embedding failed: ${e instanceof Error ? e.message : e})` };
      }
      hits.sort((a, b) => b.score - a.score);
      hits = hits.slice(0, k);

      if (hits.length === 0) {
        notes.unshift('nothing relevant found');
        return { results: [], note: notes.join('; ') };
      }
      const out: ToolResult = {
        results: hits.map((h) => ({ source: h.source, location: h.location, kind: h.kind, score: h.score, text: h.text })),
      };
      if (notes.length) out.note = notes.join('; ');
      return out;
    },
  };
}

export const CHART_TOOLS = ['get_customer_profile', 'get_distribution', 'term_contribution_summary'] as const;
export { FEATURES };
