import type { ToolResult } from '../data/types';

export type ChartSpec =
  | { type: 'waterfall'; data: ToolResult; meanings: Record<string, string> }
  | { type: 'histogram'; data: ToolResult }
  | { type: 'term_comparison'; data: ToolResult };

const GREEN = '#10b981';
const RED = '#f87171';
const BLUE = '#58a6ff';
const usd = (v: number) => `${v < 0 ? '$-' : '$'}${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const compact = (v: number) => Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(v);

/** "nice" tick step covering [lo, hi] in about n steps */
function ticks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-9 ? 0 : t);
  return out;
}

function Frame({ title, children, height, width = 840 }: { title: string; children: React.ReactNode; height: number; width?: number }) {
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {children}
      </svg>
    </figure>
  );
}

export function Waterfall({ data, meanings }: { data: ToolResult; meanings: Record<string, string> }) {
  const terms = (data.terms ?? {}) as Record<string, number>;
  const labels = Object.keys(terms);
  const values = labels.map((l) => terms[l]);
  if (!labels.length) return null;
  const L = 300, R = 90, W = 840, row = 32, top = 14, bottom = 40;
  const H = top + labels.length * row + bottom;
  const lo = Math.min(-400, ...values), hi = Math.max(300, ...values);
  const x = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R);
  const ts = ticks(lo, hi, 7);
  return (
    <Frame title={`Customer ${data.customer_id} — P&L Breakdown (Profit: ${usd(Number(data.profit))})`} height={H} width={W}>
      {ts.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - bottom} className="grid" />
          <text x={x(t)} y={H - bottom + 16} textAnchor="middle" className="tick">{t}</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={top} y2={H - bottom} className="axis" />
      {labels.map((l, i) => {
        const v = values[i];
        const y = top + i * row;
        const x0 = x(Math.min(0, v)), w = Math.max(1, Math.abs(x(v) - x(0)));
        return (
          <g key={l}>
            <text x={L - 10} y={y + row / 2 + 4} textAnchor="end" className="label">{meanings[l] ?? l}</text>
            <rect x={x0} y={y + 5} width={w} height={row - 10} fill={v >= 0 ? GREEN : RED} rx={2}>
              <title>{`${meanings[l] ?? l}: ${usd(v)}`}</title>
            </rect>
            <text x={v >= 0 ? x0 + w + 6 : x0 - 6} y={y + row / 2 + 4} textAnchor={v >= 0 ? 'start' : 'end'} className="value">{usd(v)}</text>
          </g>
        );
      })}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="tick">USD ($)</text>
    </Frame>
  );
}

export function Histogram({ data }: { data: ToolResult }) {
  const edges = (data.bin_edges ?? []) as number[];
  const counts = (data.counts ?? []) as number[];
  if (!edges.length || !counts.length) return null;
  const W = 760, H = 340, L = 64, R = 16, T = 12, B = 54;
  const maxC = Math.max(...counts, 1);
  const x = (v: number) => L + ((v - edges[0]) / (edges[edges.length - 1] - edges[0] || 1)) * (W - L - R);
  const y = (c: number) => T + (1 - c / maxC) * (H - T - B);
  const col = String(data.column ?? '');
  return (
    <Frame title={`Distribution of ${data.description ?? col} (${col})`} height={H}>
      {ticks(0, maxC, 4).map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
          <text x={L - 6} y={y(t) + 4} textAnchor="end" className="tick">{compact(t)}</text>
        </g>
      ))}
      {counts.map((c, i) => (
        <rect key={i} x={x(edges[i]) + 0.5} y={y(c)} width={Math.max(1, x(edges[i + 1]) - x(edges[i]) - 1)} height={H - B - y(c)} fill={BLUE} fillOpacity={0.75} stroke={BLUE} strokeWidth={0.5}>
          <title>{`${edges[i].toLocaleString()} – ${edges[i + 1].toLocaleString()}: ${c.toLocaleString()} customers`}</title>
        </rect>
      ))}
      {ticks(edges[0], edges[edges.length - 1], 6).map((t) => (
        <text key={t} x={x(t)} y={H - B + 16} textAnchor="middle" className="tick">{compact(t)}</text>
      ))}
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" className="tick">{col}</text>
      <text transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" className="tick">Count</text>
    </Frame>
  );
}

export function TermComparison({ data }: { data: ToolResult }) {
  const terms = (data.terms ?? []) as { meaning: string; top20_mean: number; rest_mean: number }[];
  if (!terms.length) return null;
  const L = 250, R = 90, W = 760, row = 44, top = 30, bottom = 40;
  const H = top + terms.length * row + bottom;
  const vals = terms.flatMap((t) => [t.top20_mean, t.rest_mean]);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const x = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R);
  return (
    <Frame title="Mean P&L term: top 20% vs rest" height={H}>
      <rect x={L} y={8} width={12} height={12} fill={GREEN} rx={2} /><text x={L + 18} y={19} className="tick">Top 20%</text>
      <rect x={L + 100} y={8} width={12} height={12} fill={RED} rx={2} /><text x={L + 118} y={19} className="tick">Bottom 80%</text>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - bottom} className="grid" />
          <text x={x(t)} y={H - bottom + 16} textAnchor="middle" className="tick">{compact(t)}</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={top} y2={H - bottom} className="axis" />
      {terms.map((t, i) => {
        const y = top + i * row;
        const bar = (v: number, dy: number, color: string) => (
          <g>
            <rect x={x(Math.min(0, v))} y={y + dy} width={Math.max(1, Math.abs(x(v) - x(0)))} height={14} fill={color} rx={2}>
              <title>{usd(v)}</title>
            </rect>
            <text x={v >= 0 ? x(v) + 5 : x(v) - 5} y={y + dy + 11} textAnchor={v >= 0 ? 'start' : 'end'} className="value">{usd(v)}</text>
          </g>
        );
        return (
          <g key={t.meaning}>
            <text x={L - 8} y={y + row / 2 + 4} textAnchor="end" className="label">{t.meaning.slice(0, 34)}</text>
            {bar(t.top20_mean, 6, GREEN)}
            {bar(t.rest_mean, 23, RED)}
          </g>
        );
      })}
    </Frame>
  );
}

export function Chart({ spec }: { spec: ChartSpec }) {
  if (spec.type === 'waterfall') return <Waterfall data={spec.data} meanings={spec.meanings} />;
  if (spec.type === 'histogram') return <Histogram data={spec.data} />;
  return <TermComparison data={spec.data} />;
}

/** Charts the Streamlit app auto-renders, taken from the tool results the model actually used. */
export function chartsFromTraces(traces: { name: string; result: ToolResult }[], meanings: Record<string, string>): ChartSpec[] {
  const out: ChartSpec[] = [];
  const seen = new Set<string>();
  for (const { name, result } of traces) {
    if (result.error) continue;
    let spec: ChartSpec | null = null;
    let key = '';
    if (name === 'get_customer_profile') { spec = { type: 'waterfall', data: result, meanings }; key = `w${result.customer_id}`; }
    else if (name === 'get_distribution') { spec = { type: 'histogram', data: result }; key = `h${result.column}${(result.counts as number[])?.length}`; }
    else if (name === 'term_contribution_summary') { spec = { type: 'term_comparison', data: result }; key = 't'; }
    if (spec && !seen.has(key) && out.length < 3) { seen.add(key); out.push(spec); }
  }
  return out;
}
