/**
 * Python-compatible number rounding/formatting.
 *
 * JS rounds exact binary ties (0.125, 2.5, -1234.5 …) half away from zero; Python rounds them
 * half to even. Both are exact for every other value, so only ties need handling, and they are
 * detected from the exact decimal expansion of the double.
 */

/** round(x, d) as Python does it. */
export function round(x: number, d: number): number {
  if (!Number.isFinite(x)) return x;
  const exact = Math.abs(x).toFixed(100); // exact expansion (any double ≥ ~1e-30 has fewer digits)
  const dot = exact.indexOf('.');
  const rest = exact.slice(dot + 1 + d);
  if (/^50*$/.test(rest)) {
    const kept = exact.slice(0, dot + 1 + d).replace(/\.$/, '');
    const lastDigit = Number(kept[kept.length - 1]);
    if (lastDigit % 2 === 0) return Number(x < 0 ? `-${kept}` : kept); // tie, previous digit even: round down
  }
  return Number(x.toFixed(d));
}

/** `p` significant digits of |x|, rounded half-even on the exact value. */
function significant(x: number, p: number): { digits: string; exp: number } {
  const [m, e] = Math.abs(x).toExponential(100).split('e');
  const all = m.replace('.', '');
  let exp = Number(e);
  const head = all.slice(0, p);
  const rest = all.slice(p);
  const up = rest[0] > '5' || (rest[0] === '5' && (/[1-9]/.test(rest.slice(1)) || Number(head[p - 1]) % 2 === 1));
  let digits = (BigInt(head) + (up ? 1n : 0n)).toString();
  if (digits.length > p) {
    digits = digits.slice(0, p);
    exp += 1;
  }
  return { digits, exp };
}

/** Python's f"{x:,.{p}g}". */
export function fmtG(x: number | null, p = 4): string {
  if (x === null || !Number.isFinite(x)) return 'n/a';
  if (x === 0) return '0';
  const sign = x < 0 ? '-' : '';
  const { digits, exp } = significant(x, p);
  if (exp < -4 || exp >= p) {
    const frac = digits.slice(1).replace(/0+$/, '');
    return `${sign}${digits[0]}${frac ? '.' + frac : ''}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  const group = (s: string) => s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  if (exp >= 0) {
    const int = digits.slice(0, exp + 1).padEnd(exp + 1, '0');
    const frac = digits.slice(exp + 1).replace(/0+$/, '');
    return `${sign}${group(int)}${frac ? '.' + frac : ''}`;
  }
  const frac = ('0'.repeat(-exp - 1) + digits).replace(/0+$/, '');
  return `${sign}0.${frac}`;
}
