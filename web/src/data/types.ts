export interface ColumnEnc {
  name: string;
  kind: 'i8' | 'i16' | 'i32' | 'f64';
  scale: number;
}

export interface Verification {
  ok: boolean;
  message: string;
  match_count: number;
  total: number;
  mismatches: number;
}

export interface StatsPack {
  row_count: number;
  top20_cutoff: number;
  top20_count: number;
  profit_stats: { top20: { mean: number; median: number }; rest: { mean: number; median: number } };
  term_means: Record<string, { top20_mean: number; rest_mean: number }>;
  missing_pct: Record<string, number>;
}

export interface Meta {
  generated_at: string;
  weights: Record<string, number>;
  equation: string;
  term_columns: string[];
  term_meanings: Record<string, string>;
  feature_dict: Record<string, string>;
  valid_columns: string[];
  id_min: number;
  id_max: number;
  stats: StatsPack;
  verification: Verification;
  runtime: { live: Record<string, number>; mismatches: string[]; all_match: boolean; block: string };
  customers: { shard_rows: number; n_shards: number; rows: number; columns: ColumnEnc[]; bytes: number };
}

export interface EmbedConfig {
  model: string;
  dtype: string;
  pooling: 'cls' | 'mean';
  query_prefix: string;
  sim_floor: number;
  dim: number;
  count: number;
  /** sha1 of core_chunks.json the vectors were computed from */
  chunks_sha1: string;
}

/** Output shape of the Python tools — kept identical so prompts and charts are unchanged. */
export type ToolResult = Record<string, unknown>;

export interface CustomerProfile {
  customer_id: number;
  raw_features: Record<string, number | null>;
  terms: Record<string, number>;
  profit: number;
  percentile_rank: number;
  rank_out_of_500k: number;
  is_top20: boolean;
}

export interface ToolsStatic {
  term_contribution_summary: ToolResult;
  explain_formula: ToolResult;
  top_customers: { top: { customers: unknown[] }; bottom: { customers: unknown[] } };
  segment_compare: Record<string, ToolResult>;
  distributions: Record<string, ToolResult>;
}
