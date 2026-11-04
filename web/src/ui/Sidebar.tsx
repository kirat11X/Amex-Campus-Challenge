import { useEffect, useState } from 'react';
import type { Services } from '../services';
import { ModelPanel } from './ModelPanel';
import type { Prefs, useEngine } from './useEngine';

export const EXAMPLES = [
  'Profit breakdown for customer 1',
  'Why is customer 20 in the top 20%?',
  'Compare lounge usage between top 20% and rest',
  'Show the profit distribution',
  'Which cost term hurts profitability the most?',
  'Explain my profitability formula term by term',
];

const usd = (v: number) => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

interface Props {
  services: Services;
  hook: ReturnType<typeof useEngine>;
  prefs: Prefs;
  setPrefs: (p: Prefs) => void;
  includeUser: boolean;
  setIncludeUser: (v: boolean) => void;
  onExample: (text: string) => void;
  onReset: () => void;
  canAsk: boolean;
  promptMode: 'auto' | 'compact' | 'full';
  setPromptMode: (m: 'auto' | 'compact' | 'full') => void;
}

export function Sidebar(p: Props) {
  const { services, prefs } = p;
  const { meta } = services.ds;
  const sp = meta.stats;
  const [userChunks, setUserChunks] = useState(services.kb.userChunkCount());
  const [embed, setEmbed] = useState(services.embedder.status);
  useEffect(() => services.kb.subscribe(() => setUserChunks(services.kb.userChunkCount())), [services]);
  useEffect(() => services.embedder.onStatus(setEmbed), [services]);

  const v = meta.verification;
  const ctx = meta.runtime;

  return (
    <aside className="sidebar">
      {/* Top Green Badge matching screenshots */}
      <div className="badge-ok" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>📚</span>
        <span>Knowledge base: {services.kb.core.size} core + {userChunks} of your chunks</span>
      </div>

      {embed.state === 'loading' && (
        <div className="note">Loading retrieval model (~34 MB, once){embed.progress ? ` — ${Math.round(embed.progress)}%` : '…'}</div>
      )}
      {embed.state === 'error' && <div className="badge-fail">⚠️ Retrieval model unavailable: {embed.message}</div>}

      {/* Switches matching screenshot */}
      <label className="toggle" title="When off, retrieval is restricted to the built-in ProfitLens corpus.">
        <input type="checkbox" checked={p.includeUser} disabled={userChunks === 0} onChange={(e) => p.setIncludeUser(e.target.checked)} />
        <span>Include my uploaded files in answers</span>
        <span className="info-icon" title="Toggle inclusion of user ingested documents in answers">🛈</span>
      </label>

      <label className="toggle" title="On: multi-round tool calls over data. Off: direct model reply.">
        <input type="checkbox" checked={prefs.useTools} onChange={(e) => p.setPrefs({ ...prefs, useTools: e.target.checked })} />
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span>⚙️</span>
          <span>Tools &amp; retrieval</span>
        </span>
        <span className="info-icon" title="Grounded pandas tools and knowledge-base search">🛈</span>
      </label>

      {/* Dataset section matching screenshot */}
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span>📊</span>
        <span>Dataset</span>
      </h3>
      <div className="cards">
        <div className="card">
          <div className="label">CUSTOMERS</div>
          <div className="val">{sp.row_count.toLocaleString()}</div>
        </div>
        <div className="card">
          <div className="label">TOP-20% CUTOFF</div>
          <div className="val">{usd(sp.top20_cutoff)}</div>
        </div>
      </div>
      <div className="card">
        <div className="label">TOP-20% MEAN PROFIT</div>
        <div className="val">{usd(sp.profit_stats.top20.mean)}</div>
      </div>

      {/* Try asking matching screenshot */}
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span>💡</span>
        <span>Try asking</span>
      </h3>
      {EXAMPLES.map((ex) => (
        <button
          key={ex}
          className="example"
          disabled={!p.canAsk}
          title={p.canAsk ? '' : 'Load or activate model'}
          onClick={() => p.onExample(ex)}
        >
          {ex}
        </button>
      ))}

      {/* Engine & Settings accordion */}
      <details className="advanced" style={{ marginTop: '24px' }}>
        <summary>⚙️ Model &amp; Engine Settings</summary>
        <ModelPanel hook={p.hook} prefs={prefs} setPrefs={p.setPrefs} />
        {v.ok ? <div className="badge-ok" style={{ fontSize: '0.78rem' }}>{v.message}</div> : <div className="badge-fail">{v.message}</div>}
        {ctx.all_match ? <div className="badge-ok" style={{ fontSize: '0.78rem' }}>✓ Context stats verified</div>
          : <div className="badge-fail">⚠️ Context stats stale</div>}
        <button className="secondary" style={{ marginTop: '12px' }} onClick={p.onReset}>🔄 Reset chat</button>
      </details>
    </aside>
  );
}
