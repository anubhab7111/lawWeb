import { useRef, useState } from "react";
import { searchSimilarCases, searchSimilarCasesByText } from "../api";

interface CaseHit {
  caseName: string;
  citation: string;
  court: string;
  date: string;
  url: string;
  facts: string;
  issues: string[];
  holding: string;
  score: number;
}

interface Statute {
  actName: string;
  sectionNumber: string;
  title: string;
  text: string;
}

interface Result {
  firac: {
    facts: string;
    issues: string[];
    rules: string[];
    application: string;
    conclusion: string;
    legalDomainHint: string;
  };
  similarSupremeCourtCases: CaseHit[];
  relevantHighCourtCasesText: string;
  relevantStatutes: Statute[];
}

export function SimilarCaseSearch() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const run = async (fn: () => Promise<Result>) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      // Normalize once here rather than guarding every .length/.map below —
      // a partial/legacy response missing these arrays would otherwise
      // throw during render and take down the whole screen.
      setResult({
        ...data,
        firac: { ...data.firac, issues: data.firac?.issues ?? [] },
        similarSupremeCourtCases: (data.similarSupremeCourtCases ?? []).map((c) => ({
          ...c,
          issues: c.issues ?? [],
        })),
        relevantStatutes: data.relevantStatutes ?? [],
      });
    } catch {
      setError("Couldn't process that document. Try a shorter excerpt or a clearer scan.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">Similar Case Search</h1>
        <p className="page-sub">
          Upload an order, petition, or notice — we'll extract the key facts and find similar Supreme Court and High Court cases plus relevant statutes.
        </p>

        <div className="card" style={{ padding: 20, marginBottom: 20 }}>
          <textarea
            className="input"
            style={{ width: "100%", minHeight: 120, resize: "vertical" }}
            placeholder="Paste document text here, or upload a file below…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <button
              className="btn btn-primary"
              disabled={loading || text.trim().length < 20}
              onClick={() => run(() => searchSimilarCasesByText(text))}
            >
              {loading ? "Analyzing…" : "Search from text"}
            </button>
            <button className="btn btn-outline" disabled={loading} onClick={() => fileInput.current?.click()}>
              Upload file instead
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png"
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) run(() => searchSimilarCases(file));
              }}
            />
          </div>
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}
        {loading && <div className="shimmer" style={{ height: 160, borderRadius: "var(--r-lg)" }} />}

        {result && !loading && (
          <>
            <div className="card" style={{ padding: "18px 20px", marginBottom: 18 }}>
              <div className="section-label">Extracted summary</div>
              <p className="prose" style={{ font: "400 15.5px var(--font-serif)", color: "var(--text-2)" }}>{result.firac.facts || "—"}</p>
              {result.firac.issues.length > 0 && (
                <>
                  <div style={{ font: "600 12px var(--font-body)", color: "var(--muted-3)", marginTop: 10 }}>Legal issues</div>
                  <ul style={{ margin: "4px 0 0 18px", font: "400 13px var(--font-body)", color: "var(--text-2)" }}>
                    {result.firac.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                  </ul>
                </>
              )}
            </div>

            {result.similarSupremeCourtCases.length > 0 && (
              <>
                <div className="section-label">Similar Supreme Court cases</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 18 }}>
                  {result.similarSupremeCourtCases.map((c, i) => (
                    <div key={i} className="card" style={{ padding: "14px 16px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                        <div className="cite" style={{ font: "600 13.5px var(--font-body)" }}>{c.caseName}</div>
                        {c.score > 0 && (
                          <div className="mono-num" style={{ font: "600 11px var(--font-body)", color: "var(--muted-3)", whiteSpace: "nowrap" }}>
                            match {Math.round(c.score * 100)}%
                          </div>
                        )}
                      </div>
                      <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                        {c.court}{c.citation ? ` · ${c.citation}` : ""}
                      </div>
                      {c.holding && (
                        <div style={{ font: "400 12.5px var(--font-body)", color: "var(--text-2)", marginTop: 6 }}>
                          {c.holding}
                        </div>
                      )}
                      {c.issues.length > 0 && (
                        <ul style={{ margin: "6px 0 0 18px", font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>
                          {c.issues.map((issue, j) => <li key={j}>{issue}</li>)}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}

            {result.relevantStatutes.length > 0 && (
              <>
                <div className="section-label">Relevant statute sections</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {result.relevantStatutes.map((s, i) => (
                    <div key={i} className="card" style={{ padding: "14px 16px" }}>
                      <div style={{ font: "600 13.5px var(--font-body)" }}>{s.actName} — Section {s.sectionNumber}</div>
                      <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>{s.text.slice(0, 240)}…</div>
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
