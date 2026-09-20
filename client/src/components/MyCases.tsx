import { useEffect, useState } from "react";
import { addCaseNote, deleteCase, fetchCaseDetail, fetchSavedCases, saveCase, syncCase, type SavedCase } from "../api";
import type { UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
}

interface CaseDetail extends SavedCase {
  timeline: { id: string; eventType: string; eventDate: string | null; title: string | null; detail: string | null }[];
  notes: { id: string; noteText: string; createdAt: string | null }[];
  aiSummaries: { id: string; summaryText: string; createdAt: string | null }[];
}

export function MyCases({ user }: Props) {
  const [cases, setCases] = useState<SavedCase[]>([]);
  const [selected, setSelected] = useState<CaseDetail | null>(null);
  const [cnr, setCnr] = useState("");
  const [byNumber, setByNumber] = useState(false);
  const [court, setCourt] = useState("");
  const [caseNumber, setCaseNumber] = useState("");
  const [year, setYear] = useState("");
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    fetchSavedCases().then(setCases).catch(() => setError("Couldn't load your saved cases.")).finally(() => setLoading(false));
  };

  useEffect(() => { if (user) load(); }, [user]);

  const attempt = async (action: () => Promise<unknown>, fallback: string) => {
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : fallback);
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm("Stop tracking this case? Its notes and timeline will be deleted.")) return;
    await attempt(() => deleteCase(id), "Couldn't delete that case.");
    setSelected(null);
    load();
  };

  const openCase = (id: string) => {
    fetchCaseDetail(id)
      .then((detail) =>
        // Normalize once here rather than guarding every .length/.map below
        // — a partial/legacy response missing these arrays would otherwise
        // throw during render and take down the whole screen.
        setSelected({
          ...detail,
          timeline: detail.timeline ?? [],
          notes: detail.notes ?? [],
          aiSummaries: detail.aiSummaries ?? [],
        })
      )
      .catch(() => setError("Couldn't load case detail."));
  };

  const handleSave = async () => {
    const payload = byNumber
      ? { court: court.trim(), caseNumber: caseNumber.trim(), year: Number(year) }
      : { cnr: cnr.trim() };
    if (byNumber ? !payload.court || !payload.caseNumber || !payload.year : !cnr.trim()) return;
    setError(null);
    try {
      await saveCase(payload);
      setCnr("");
      setCourt("");
      setCaseNumber("");
      setYear("");
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save that case.");
    }
  };

  if (!user) {
    return <div className="container"><p className="page-sub">Sign in to track your cases.</p></div>;
  }

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">My Cases</h1>
        <p className="page-sub">Save cases by CNR and track hearings, orders, and AI summaries in one place.</p>

        <div className="card" style={{ padding: 20, marginBottom: 24 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {byNumber ? (
              <>
                <input className="input" style={{ flex: 2, minWidth: 160 }} placeholder="Court (e.g. Delhi High Court)" value={court} onChange={(e) => setCourt(e.target.value)} />
                <input className="input" style={{ flex: 1, minWidth: 120 }} placeholder="Case number" value={caseNumber} onChange={(e) => setCaseNumber(e.target.value)} />
                <input className="input" style={{ width: 90 }} placeholder="Year" inputMode="numeric" value={year} onChange={(e) => setYear(e.target.value)} />
              </>
            ) : (
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="Enter CNR number (e.g. DLHC010012342024)"
                value={cnr}
                onChange={(e) => setCnr(e.target.value)}
              />
            )}
            <button className="btn btn-primary" onClick={handleSave}>Save case</button>
          </div>
          <button className="btn btn-ghost btn-sm" style={{ marginTop: 8, paddingLeft: 0 }} onClick={() => setByNumber((v) => !v)}>
            {byNumber ? "Use a CNR instead" : "Don't have a CNR? Add by court and case number"}
          </button>
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        {selected ? (
          <div>
            <button className="btn btn-ghost btn-sm" style={{ marginBottom: 14, paddingLeft: 0 }} onClick={() => setSelected(null)}>← Back to all cases</button>
            <div className="card" style={{ padding: "18px 20px", marginBottom: 18 }}>
              <div className={selected.title ? "cite" : undefined} style={{ font: "700 16px var(--font-head)" }}>{selected.title || selected.cnr}</div>
              <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{selected.court} · {selected.status}</div>
              <button className="btn btn-outline btn-sm" style={{ marginTop: 10 }} onClick={() => attempt(() => syncCase(selected.id), "Couldn't re-sync this case.").then(() => openCase(selected.id))}>
                Re-sync now
              </button>
              <button className="btn btn-ghost btn-sm" style={{ marginTop: 10, marginLeft: 8 }} onClick={() => handleDelete(selected.id)}>Stop tracking</button>
            </div>

            {selected.aiSummaries.length > 0 && (
              <>
                <div className="section-label">AI summaries</div>
                {selected.aiSummaries.map((s) => (
                  <div key={s.id} className="card" style={{ padding: "14px 16px", marginBottom: 10 }}>{s.summaryText}</div>
                ))}
              </>
            )}

            <div className="section-label">Timeline</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 18 }}>
              {selected.timeline.length === 0 && <p style={{ color: "var(--muted-2)" }}>No events yet.</p>}
              {selected.timeline.map((e) => (
                <div key={e.id} className="card" style={{ padding: "14px 16px" }}>
                  <div style={{ font: "600 12px var(--font-body)", textTransform: "capitalize", color: "var(--accent)" }}>{e.eventType}</div>
                  <div style={{ font: "600 13.5px var(--font-body)" }}>{e.title}</div>
                  {e.eventDate && (
                    <div style={{ font: "500 12px var(--font-body)", color: "var(--muted-2)" }}>
                      {new Date(e.eventDate).toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" })}
                    </div>
                  )}
                  {e.detail && <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>{e.detail}</div>}
                </div>
              ))}
            </div>

            <div className="section-label">Notes</div>
            <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
              <input className="input" style={{ flex: 1 }} placeholder="Add a note…" value={note} onChange={(e) => setNote(e.target.value)} />
              <button
                className="btn btn-outline"
                onClick={async () => { if (!note.trim()) return; await attempt(() => addCaseNote(selected.id, note), "Couldn't save that note."); setNote(""); openCase(selected.id); }}
              >
                Add
              </button>
            </div>
            {selected.notes.map((n) => (
              <div key={n.id} className="card" style={{ padding: "10px 14px", marginBottom: 8, font: "400 13px var(--font-body)" }}>{n.noteText}</div>
            ))}
          </div>
        ) : loading ? (
          <div className="shimmer" style={{ height: 120, borderRadius: "var(--r-lg)" }} />
        ) : cases.length === 0 ? (
          <div className="empty-state">
            <div className="empty-sub">No saved cases yet — enter a CNR above to get started.</div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {cases.map((c) => (
              <div key={c.id} className="card" style={{ padding: "16px 18px", cursor: "pointer" }} onClick={() => openCase(c.id)}>
                <div className={c.title ? "cite" : undefined} style={{ font: "700 14.5px var(--font-head)" }}>{c.title || c.cnr}</div>
                <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{c.court} · {c.status}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
