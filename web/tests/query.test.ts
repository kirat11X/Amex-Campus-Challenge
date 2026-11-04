import { describe, expect, it } from 'vitest';
import { compileQuery, QueryError } from '../src/tools/query';

const cols: Record<string, Float64Array> = {
  a: Float64Array.from([0, 1, 2, 3, 4, 5]),
  b: Float64Array.from([5, 4, 3, 2, 1, 0]),
  n: Float64Array.from([NaN, 1, NaN, 3, NaN, 5]),
  flag: Float64Array.from([1, 0, 1, 0, 1, 0]),
};
const col = (name: string) => {
  if (!(name in cols)) throw new Error(name);
  return cols[name];
};
const run = (q: string) => {
  const f = compileQuery(q, col);
  return [0, 1, 2, 3, 4, 5].filter((i) => f(i) !== 0);
};

describe('query evaluator (pandas DataFrame.query subset)', () => {
  it('comparisons and boolean words', () => {
    expect(run('a >= 3')).toEqual([3, 4, 5]);
    expect(run('a >= 1 and b >= 2')).toEqual([1, 2, 3]);
    expect(run('a == 0 or b == 0')).toEqual([0, 5]);
    expect(run('not (a >= 2)')).toEqual([0, 1]);
  });
  it('& and | bind like and/or, looser than comparisons', () => {
    expect(run('a > 1 & b > 1')).toEqual([2, 3]);
    expect(run('a < 1 | b < 1')).toEqual([0, 5]);
    expect(run('~(a > 1)')).toEqual([0, 1]);
  });
  it('chained comparisons and arithmetic', () => {
    expect(run('1 <= a < 3')).toEqual([1, 2]);
    expect(run('(a + b) * 2 == 10')).toEqual([0, 1, 2, 3, 4, 5]);
    expect(run('a ** 2 > 10')).toEqual([4, 5]);
    expect(run('-a > -2')).toEqual([0, 1]);
    expect(run('a % 2 == 0')).toEqual([0, 2, 4]);
    expect(run('a / 2 >= 2')).toEqual([4, 5]);
  });
  it('NaN semantics: comparisons are false, != is true', () => {
    expect(run('n > 0')).toEqual([1, 3, 5]);
    expect(run('n != n')).toEqual([0, 2, 4]);
    expect(run('n.isna()')).toEqual([0, 2, 4]);
    expect(run('n.notna()')).toEqual([1, 3, 5]);
  });
  it('booleans, scientific notation and membership', () => {
    expect(run('flag == True')).toEqual([0, 2, 4]);
    expect(run('flag')).toEqual([0, 2, 4]);
    expect(run('a > 1e0 and a < 2.5e0')).toEqual([2]);
    expect(run('a in [1, 3, 9]')).toEqual([1, 3]);
    expect(run('a not in [1, 3]')).toEqual([0, 2, 4, 5]);
  });
  it('rejects bad input without executing anything', () => {
    expect(() => compileQuery('zzz > 1', col)).toThrow(/name 'zzz' is not defined/);
    expect(() => compileQuery('a >', col)).toThrow(QueryError);
    expect(() => compileQuery('a > 1 )', col)).toThrow(QueryError);
    expect(() => compileQuery('a $ 1', col)).toThrow(/unexpected character/);
    expect(() => compileQuery('a.sum()', col)).toThrow(/unsupported method/);
  });
});
