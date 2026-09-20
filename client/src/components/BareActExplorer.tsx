import { useEffect, useState } from "react";
import { fetchBareActNames, searchBareAct } from "../api";

interface Match {
  actName: string;
  sectionNumber: string;
  title: string;
  text: string;
  domain: string;
}

interface Judgment {
  caseName: string;
  citation: string;
  court: string;
  date: string;
  url: string;
}

interface Result {
  query: string;
  isSectionLookup: boolean;
  ambiguous: boolean;
  matches: Match[];
  landmarkJudgments: Judgment[];
  explanation: string | null;
}

export function BareActExplorer() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [acts, setActs] = useState<string[]>([]);
  const [act, setAct] = useState("");

  useEffect(() => {
    fetchBareActNames().then(setActs).catch(() => {});
  }, []);

  const runSearch = async (q: string, actHint: string = act) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await searchBareAct(q, actHint || undefined);
      // Normalize once here rather than guarding every .length/.map below —
      // a partial/legacy response missing these arrays would otherwise
      // throw during render and take down the whole screen.
      setResult({
        ...data,
        matches: data.matches ?? [],
        landmarkJudgments: data.landmarkJudgments ?? [],
      });
    } catch {
      setError("Couldn't find that section or topic. Try rephrasing.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">Bare Act Explorer</h1>
        <p className="page-sub">
          Search "Section 302" or "arrest without warrant" — get the statute text, landmark judgments, and a plain-language explanation.
        </p>

        <form
          onSubmit={(e) => { e.preventDefault(); runSearch(query); }}
          style={{ display: "flex", gap: 10, marginBottom: 24 }}
        >
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="e.g. Section 302, or arrest without warrant"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select className="input" style={{ maxWidth: 220 }} value={act} onChange={(e) => setAct(e.target.value)} aria-label="Act">
            <option value="">Any act</option>
            {acts.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        {loading && <div className="shimmer" style={{ height: 140, borderRadius: "var(--r-lg)" }} />}

        {result && !loading && (
          <>
            {result.ambiguous && (
              <div className="card" style={{ padding: 14, marginBottom: 16, borderColor: "var(--accent-line)", background: "var(--accent-soft)" }}>
                This section number appears in multiple acts — showing all matches below. Narrow it to one:
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
                  {Array.from(new Set(result.matches.map((m) => m.actName))).map((name) => (
                    <button key={name} className="chip" type="button" onClick={() => { setAct(name); runSearch(result.query, name); }}>{name}</button>
                  ))}
                </div>
              </div>
            )}

            {result.matches.length === 0 && (
              <div className="empty-state"><div className="empty-sub">No matching provision found.</div></div>
            )}

            {result.matches.map((m, i) => (
              <div key={i} className="card" style={{ padding: "18px 20px", marginBottom: 14 }}>
                <div style={{ font: "700 15px var(--font-head)" }}>{m.actName} — Section {m.sectionNumber}</div>
                <div style={{ font: "600 13px var(--font-body)", color: "var(--muted-2)", marginTop: 2 }}>{m.title}</div>
                <p className="prose" style={{ font: "400 15.5px var(--font-serif)", color: "var(--text-2)", marginTop: 10, whiteSpace: "pre-wrap" }}>{m.text}</p>
              </div>
            ))}

            {result.explanation && (
              <div className="card" style={{ padding: "18px 20px", marginBottom: 14, background: "var(--surface-tint)" }}>
                <div className="section-label">In plain language</div>
                <p className="prose" style={{ font: "400 15.5px var(--font-serif)", color: "var(--text-2)" }}>{result.explanation}</p>
              </div>
            )}

            {result.landmarkJudgments.length > 0 && (
              <>
                <div className="section-label">Landmark judgments</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {result.landmarkJudgments.map((j, i) => (
                    <div key={i} className="card" style={{ padding: "14px 16px" }}>
                      <div className="cite" style={{ font: "600 13.5px var(--font-body)" }}>{j.caseName}</div>
                      <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                        {j.court}{j.citation ? ` · ${j.citation}` : ""}{j.date ? ` · ${j.date.slice(0, 4)}` : ""}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
