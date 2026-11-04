import { useEffect, useRef, useState } from 'react';
import { SUPPORTED_EXTS } from '../rag/ingest';
import type { Services } from '../services';

export function Digest({ services }: { services: Services }) {
  const { kb } = services;
  const [files, setFiles] = useState([...kb.manifest]);
  const [log, setLog] = useState<{ ok: boolean; text: string }[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => kb.subscribe(() => setFiles([...kb.manifest])), [kb]);

  async function digest(list: FileList | null) {
    if (!list) return;
    for (const f of Array.from(list)) {
      setBusy(f.name);
      const r = await kb.ingestFile(f.name, await f.arrayBuffer());
      setLog((l) => [{ ok: r.ok, text: r.message }, ...l].slice(0, 6));
      if (r.ok && !r.duplicate) services.prefs.includeUser = true;
    }
    setBusy(null);
    if (input.current) input.current.value = '';
  }

  return (
    <div className="digest">
      <h2>Do you want to digest your own files?</h2>
      <p className="muted">Upload documents and ProfitLens will read them and answer questions about them alongside the built-in analysis.
        Everything stays in this browser — files are parsed, chunked and embedded on your device and stored in IndexedDB.</p>

      <label className="drop">
        <input ref={input} type="file" multiple accept={SUPPORTED_EXTS.map((e) => '.' + e).join(',')} disabled={busy !== null}
          onChange={(e) => void digest(e.target.files)} />
        <span>{busy ? `Digesting ${busy}… (the first time also downloads the ~34 MB retrieval model)` : 'Drop files here, or click to choose'}</span>
      </label>
      <ul className="log" aria-live="polite">
        {log.map((l, i) => <li key={i} className={l.ok ? 'ok' : 'bad'}>{l.ok ? '' : '⚠️ '}{l.text}</li>)}
      </ul>

      <h3>📑 Your digested files</h3>
      {files.length === 0 ? (
        <div className="note">No files digested yet. Anything you upload stays local — only the text chunks are embedded.</div>
      ) : (
        <>
          <table>
            <thead><tr><th>File</th><th>Type</th><th>Chunks</th><th>Ingested</th><th /></tr></thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.filename}>
                  <td>{f.filename}</td><td>.{f.type}</td><td>{f.chunks}</td><td>{f.ingested_at}</td>
                  <td><button className="link" title={`Remove ${f.filename}`} onClick={() => void kb.deleteFile(f.filename)}>🗑</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          {confirm ? (
            <div className="note">
              Remove every file you've digested? This can't be undone.
              <div className="row">
                <button className="primary" onClick={() => { void kb.clearUser(); setConfirm(false); }}>Yes, clear all</button>
                <button className="secondary" onClick={() => setConfirm(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <button className="secondary" onClick={() => setConfirm(true)}>🧹 Clear all my files</button>
          )}
        </>
      )}
      <p className="muted small">
        Supported: {SUPPORTED_EXTS.map((e) => '.' + e).join(', ')} · max 25 MB per file. Spreadsheets and CSVs are digested as profile
        documents (schema, per-column statistics and a short preview) rather than row by row. Ask about them in the <b>Chat</b> tab —
        answers cite the filename and location.
      </p>
    </div>
  );
}
