import { useEffect, useState } from "react";
import { addCaseNote, fetchCaseDetail, fetchSavedCases, saveCase, syncCase, type SavedCase } from "../api";
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
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    fetchSavedCases().then(setCases).catch(() => setError("Couldn't load your saved cases.")).finally(() => setLoading(false));
  };

  useEffect(() => { if (user) load(); }, [user]);

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
    if (!cnr.trim()) return;
    setError(null);
    try {
      await saveCase({ cnr: cnr.trim() });
      setCnr("");
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

        <div className="card" style={{ padding: 20, marginBottom: 24, display: "flex", gap: 10 }}>
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="Enter CNR number (e.g. DLHC010012342024)"
            value={cnr}
            onChange={(e) => setCnr(e.target.value)}
          />
          <button className="btn btn-primary" onClick={handleSave}>Save case</button>
        </div>

        {error && <div className="card" style={{ borderColor: "#f0d9d6", color: "var(--danger)", marginBottom: 18 }}>{error}</div>}

        {selected ? (
          <div>
            <button className="btn btn-sm" style={{ marginBottom: 14 }} onClick={() => setSelected(null)}>← Back to all cases</button>
            <div className="card" style={{ padding: "18px 20px", marginBottom: 18 }}>
              <div style={{ font: "700 16px var(--font-head)" }}>{selected.title || selected.cnr}</div>
              <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{selected.court} · {selected.status}</div>
              <button className="btn btn-sm" style={{ marginTop: 10 }} onClick={() => syncCase(selected.id).then(() => openCase(selected.id))}>
                Re-sync now
              </button>
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
                  {e.detail && <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>{e.detail}</div>}
                </div>
              ))}
            </div>

            <div className="section-label">Notes</div>
            <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
              <input className="input" style={{ flex: 1 }} placeholder="Add a note…" value={note} onChange={(e) => setNote(e.target.value)} />
              <button
                className="btn"
                onClick={async () => { if (!note.trim()) return; await addCaseNote(selected.id, note); setNote(""); openCase(selected.id); }}
              >
                Add
              </button>
            </div>
            {selected.notes.map((n) => (
              <div key={n.id} className="card" style={{ padding: "10px 14px", marginBottom: 8, font: "400 13px var(--font-body)" }}>{n.noteText}</div>
            ))}
          </div>
        ) : loading ? (
          <div className="shimmer" style={{ height: 120, borderRadius: 16 }} />
        ) : cases.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: "center", borderStyle: "dashed" }}>
            No saved cases yet — enter a CNR above to get started.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {cases.map((c) => (
              <div key={c.id} className="card" style={{ padding: "16px 18px", cursor: "pointer" }} onClick={() => openCase(c.id)}>
                <div style={{ font: "700 14.5px var(--font-head)" }}>{c.title || c.cnr}</div>
                <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{c.court} · {c.status}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
