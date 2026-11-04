import { describe, expect, it } from 'vitest';
import { VectorStore } from '../src/rag/store';
import { makeChunk } from '../src/rag/chunker';

const unit = (...v: number[]) => {
  const n = Math.hypot(...v) || 1;
  return Float32Array.from(v.map((x) => x / n));
};

describe('VectorStore', () => {
  const chunks = [makeChunk('a', 's1', 'user', 'p.1'), makeChunk('b', 's1', 'user', 'p.2'), makeChunk('c', 's2', 'user', 'p.1')];
  const vecs = [unit(1, 0, 0), unit(1, 1, 0), unit(0, 0, 1)];
  const make = () => {
    const s = new VectorStore(3);
    s.add(vecs, chunks);
    return s;
  };

  it('ranks by cosine similarity and applies the floor', () => {
    const hits = make().search(unit(1, 0, 0), 3, 0.5);
    expect(hits.map((h) => h.location)).toEqual(['p.1', 'p.2']);
    expect(hits[0].score).toBe(1);
    expect(hits[1].score).toBeCloseTo(0.7071, 4);
    expect(make().search(unit(0, 1, 0), 5, 0.99)).toEqual([]);
  });
  it('skips chunk ids it already has', () => {
    const s = make();
    expect(s.add(vecs, chunks)).toBe(0);
    expect(s.size).toBe(3);
  });
  it('deletes by source and lists sources', () => {
    const s = make();
    expect(s.sources).toEqual(['s1', 's2']);
    expect(s.deleteBySource('s1')).toBe(2);
    expect(s.size).toBe(1);
    expect(s.search(unit(0, 0, 1), 5)[0].source).toBe('s2');
    s.clear();
    expect(s.size).toBe(0);
  });
  it('refuses vectors of the wrong dimension', () => {
    expect(() => new VectorStore(4).add([unit(1, 0, 0)], [chunks[0]])).toThrow(/4-d/);
  });
});
