/**
 * A small, safe evaluator for the pandas `DataFrame.query` subset the LLM uses:
 *
 *   f3 >= 1 and profit > 0        1 <= f19 < 3        not (f2 >= 1)       ~(f12 > 20)
 *   f1 > 0 & f11 < 0.05           (f6 + f7) * 2 > 1e5  f4 != f4            f4.isna()
 *   f13 in [1, 2]                 is_top20 == True
 *
 * It never executes the text: it is tokenised, parsed to an AST and compiled to
 * closures over typed-array columns. As in pandas, `&`/`|` bind like and/or
 * (looser than comparisons), and comparisons against NaN are false.
 */

export class QueryError extends Error {}

type Tok = { t: 'num' | 'id' | 'op'; v: string };
type Fn = (i: number) => number;

const OPS = ['**', '//', '<=', '>=', '==', '!=', '<', '>', '+', '-', '*', '/', '%', '(', ')', '[', ']', ',', '&', '|', '~', '.'];

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    const num = /^(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)/.exec(src.slice(i));
    if (num) {
      out.push({ t: 'num', v: num[0] });
      i += num[0].length;
      continue;
    }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (id) {
      out.push({ t: 'id', v: id[0] });
      i += id[0].length;
      continue;
    }
    const op = OPS.find((o) => src.startsWith(o, i));
    if (!op) throw new QueryError(`unexpected character '${ch}'`);
    out.push({ t: 'op', v: op });
    i += op.length;
  }
  return out;
}

export function compileQuery(src: string, col: (name: string) => Float64Array): Fn {
  const toks = tokenize(src);
  let p = 0;
  const peek = () => toks[p];
  const isOp = (v: string) => toks[p]?.t === 'op' && toks[p].v === v;
  const isKw = (v: string) => toks[p]?.t === 'id' && toks[p].v === v;
  const eat = (v: string) => {
    if (!isOp(v)) throw new QueryError(`expected '${v}'`);
    p++;
  };

  const lit = (x: number): Fn => () => x;

  function orExpr(): Fn {
    let l = andExpr();
    while (isKw('or') || isOp('|')) {
      p++;
      const a = l;
      const b = andExpr();
      l = (i) => (a(i) !== 0 || b(i) !== 0 ? 1 : 0);
    }
    return l;
  }

  function andExpr(): Fn {
    let l = notExpr();
    while (isKw('and') || isOp('&')) {
      p++;
      const a = l;
      const b = notExpr();
      l = (i) => (a(i) !== 0 && b(i) !== 0 ? 1 : 0);
    }
    return l;
  }

  function notExpr(): Fn {
    if (isKw('not') || isOp('~')) {
      p++;
      const a = notExpr();
      return (i) => (a(i) === 0 ? 1 : 0);
    }
    return cmpExpr();
  }

  const CMP: Record<string, (a: number, b: number) => boolean> = {
    '<': (a, b) => a < b, '<=': (a, b) => a <= b, '>': (a, b) => a > b,
    '>=': (a, b) => a >= b, '==': (a, b) => a === b, '!=': (a, b) => a !== b,
  };

  /** a < b < c is "a < b and b < c", as in Python. */
  function cmpExpr(): Fn {
    const first = arith();
    const links: Fn[] = [];
    let lhs = first;
    for (;;) {
      const t = peek();
      if (t?.t === 'op' && CMP[t.v]) {
        p++;
        const rhs = arith();
        const op = CMP[t.v];
        const l = lhs;
        links.push((i) => (op(l(i), rhs(i)) ? 1 : 0));
        lhs = rhs;
      } else if (isKw('in') || (isKw('not') && toks[p + 1]?.v === 'in')) {
        const negate = isKw('not');
        p += negate ? 2 : 1;
        eat('[');
        const items: Fn[] = [];
        while (!isOp(']')) {
          items.push(arith());
          if (isOp(',')) p++;
          else if (!isOp(']')) throw new QueryError("expected ',' or ']'");
        }
        eat(']');
        const l = lhs;
        links.push((i) => {
          const v = l(i);
          const hit = items.some((it) => it(i) === v);
          return hit !== negate ? 1 : 0;
        });
        lhs = lit(NaN);
      } else break;
    }
    if (links.length === 0) return first;
    if (links.length === 1) return links[0];
    return (i) => (links.every((l) => l(i) !== 0) ? 1 : 0);
  }

  function arith(): Fn {
    let l = term();
    while (isOp('+') || isOp('-')) {
      const plus = toks[p++].v === '+';
      const a = l;
      const b = term();
      l = plus ? (i) => a(i) + b(i) : (i) => a(i) - b(i);
    }
    return l;
  }

  function term(): Fn {
    let l = unary();
    while (isOp('*') || isOp('/') || isOp('%') || isOp('//')) {
      const op = toks[p++].v;
      const a = l;
      const b = unary();
      if (op === '*') l = (i) => a(i) * b(i);
      else if (op === '/') l = (i) => a(i) / b(i);
      else if (op === '//') l = (i) => Math.floor(a(i) / b(i));
      else l = (i) => { const x = a(i), y = b(i); return x - Math.floor(x / y) * y; };
    }
    return l;
  }

  function unary(): Fn {
    if (isOp('-')) {
      p++;
      const a = unary();
      return (i) => -a(i);
    }
    if (isOp('+')) {
      p++;
      return unary();
    }
    return power();
  }

  function power(): Fn {
    const base = atom();
    if (isOp('**')) {
      p++;
      const e = unary();
      return (i) => Math.pow(base(i), e(i));
    }
    return base;
  }

  function atom(): Fn {
    const t = toks[p];
    if (!t) throw new QueryError('unexpected end of expression');
    if (t.t === 'num') {
      p++;
      return lit(Number(t.v));
    }
    if (isOp('(')) {
      p++;
      const e = orExpr();
      eat(')');
      return e;
    }
    if (t.t === 'id') {
      p++;
      if (t.v === 'True') return lit(1);
      if (t.v === 'False') return lit(0);
      if (['and', 'or', 'not', 'in'].includes(t.v)) throw new QueryError(`unexpected '${t.v}'`);
      let c: Float64Array;
      try {
        c = col(t.v);
      } catch {
        throw new QueryError(`name '${t.v}' is not defined`);
      }
      if (isOp('.')) {
        p++;
        const m = toks[p++]?.v;
        if (!['isna', 'notna', 'isnull', 'notnull'].includes(m)) throw new QueryError(`unsupported method '.${m}()'`);
        eat('(');
        eat(')');
        const want = m === 'isna' || m === 'isnull';
        return (i) => (Number.isNaN(c[i]) === want ? 1 : 0);
      }
      return (i) => c[i];
    }
    throw new QueryError(`unexpected '${t.v}'`);
  }

  const fn = orExpr();
  if (p < toks.length) throw new QueryError(`unexpected '${toks[p].v}'`);
  return fn;
}
