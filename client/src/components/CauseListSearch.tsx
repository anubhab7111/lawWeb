import { useState } from "react";
import { searchCauseList } from "../api";

interface Entry {
  court: string;
  caseNumber: string;
  title: string;
  advocate: string;
  judge: string;
  itemNumber: number;
}

function tomorrow(): string {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}

export function CauseListSearch() {
  const [court, setCourt] = useState("Delhi High Court");
  const [date, setDate] = useState(tomorrow());
  const [advocate, setAdvocate] = useState("");
  const [judge, setJudge] = useState("");
  const [results, setResults] = useState<Entry[] | null>(null);
  const [published, setPublished] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await searchCauseList({ court, date, advocate, judge });
      setResults(Array.isArray(data.results) ? data.results : []);
      setPublished(Boolean(data.publishedAt));
    } catch {
      setError("Couldn't fetch the cause list right now.");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">Court Cause List Search</h1>
        <p className="page-sub">Search a court's cause list, filtered by advocate, judge, or case.</p>

        <div className="card" style={{ padding: 20, marginBottom: 24 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
            <input className="input" placeholder="Court" value={court} onChange={(e) => setCourt(e.target.value)} />
            <input className="input" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
            <input className="input" placeholder="Advocate (optional)" value={advocate} onChange={(e) => setAdvocate(e.target.value)} />
            <input className="input" placeholder="Judge (optional)" value={judge} onChange={(e) => setJudge(e.target.value)} />
          </div>
          <button className="btn btn-primary" onClick={run} disabled={loading}>{loading ? "Searching…" : "Search cause list"}</button>
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}
        {loading && <div className="shimmer" style={{ height: 140, borderRadius: "var(--r-lg)" }} />}

        {results && !loading && (
          !published ? (
            <div className="card" style={{ padding: 30, textAlign: "center", color: "var(--muted-2)" }}>
              Cause list for this date hasn't been published yet — check back closer to the date.
            </div>
          ) : results.length === 0 ? (
            <div className="card" style={{ padding: 30, textAlign: "center", color: "var(--muted-2)" }}>No matching entries.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {results.map((e, i) => (
                <div key={i} className="card" style={{ padding: "14px 16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div style={{ font: "600 13.5px var(--font-body)" }}>{e.caseNumber}</div>
                    <span className="pill">Item {e.itemNumber}</span>
                  </div>
                  <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>
                    {e.court} · {e.judge} · Advocate: {e.advocate}
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  );
}
