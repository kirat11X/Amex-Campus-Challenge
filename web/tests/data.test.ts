import { beforeAll, describe, expect, it } from 'vitest';
import { Dataset, decodeShard, gunzip, round } from '../src/data/dataset';
import profiles from './golden/profiles.json';
import { approx, diskBytes } from './helpers';

let ds: Dataset;
beforeAll(async () => {
  ds = await Dataset.load(diskBytes);
});

describe('customer shards', () => {
  it('reproduces get_customer_profile from the Python tool for every golden customer', async () => {
    const ids = Object.keys(profiles).map(Number);
    expect(ids.length).toBeGreaterThan(150);
    let exact = 0;
    for (const id of ids) {
      const got = await ds.customer(id);
      const want = (profiles as Record<string, unknown>)[id];
      // raw features, rank, percentile and the top-20% flag are bit-exact; money may differ in the 4th decimal
      expect(got, `customer ${id}`).not.toBeNull();
      const bad = approx(got, want, 2e-4);
      expect(bad, `customer ${id}: ${bad}`).toBeNull();
      expect(got!.is_top20).toBe((want as { is_top20: boolean }).is_top20);
      expect(got!.rank_out_of_500k).toBe((want as { rank_out_of_500k: number }).rank_out_of_500k);
      expect(got!.percentile_rank).toBe((want as { percentile_rank: number }).percentile_rank);
      expect(got!.raw_features).toEqual((want as { raw_features: unknown }).raw_features);
      if (JSON.stringify(got) === JSON.stringify(want)) exact++;
    }
    // Same formula, same doubles: essentially every profile should be identical to the last digit.
    expect(exact / ids.length).toBeGreaterThan(0.98);
  });

  it('rejects ids outside the dataset', async () => {
    expect(await ds.customer(-1)).toBeNull();
    expect(await ds.customer(500000)).toBeNull();
    expect(await ds.customer(1.5)).toBeNull();
  });

  it('decodes the last shard with the right row count', async () => {
    const c = ds.meta.customers;
    const raw = await gunzip(await diskBytes(`data/customers/${String(c.n_shards - 1).padStart(3, '0')}.bin`));
    const sh = decodeShard(raw, c.columns, c.rows - (c.n_shards - 1) * c.shard_rows);
    expect(sh.features).toHaveLength(23);
    expect(sh.less).toHaveLength(sh.rows);
  });

  it('rounds like Python for ordinary values', () => {
    expect(round(2.675, 2)).toBe(2.67); // the double is just below the tie, as in Python
    expect(round(1.005, 2)).toBe(1);
    expect(round(-683.5187, 4)).toBe(-683.5187);
    // exact binary ties go to the even digit (Python), not away from zero (JS toFixed)
    expect(round(0.125, 2)).toBe(0.12);
    expect(round(0.375, 2)).toBe(0.38);
    expect(round(2.5, 0)).toBe(2);
    expect(round(3.5, 0)).toBe(4);
    expect(round(-0.125, 2)).toBe(-0.12);
  });
});

describe('dataset-level facts', () => {
  it('matches the Streamlit runtime verification', () => {
    expect(ds.meta.runtime.all_match).toBe(true);
    expect(ds.meta.stats.row_count).toBe(500_000);
    expect(ds.meta.verification.ok).toBe(true);
  });
});
