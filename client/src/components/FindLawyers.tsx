import { useState, useEffect, useMemo } from "react";
import { fetchLawyers, recommendLawyers } from "../api";
import { LawyerCard } from "./LawyerCard";
import type { Lawyer } from "../lib/ui";

interface Props {
  onSelectLawyer: (l: Lawyer) => void;
  onBook: (l: Lawyer) => void;
}

export function FindLawyers({ onSelectLawyer, onBook }: Props) {
  const [all, setAll] = useState<Lawyer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [spec, setSpec] = useState("");
  const [loc, setLoc] = useState("");

  const [recOpen, setRecOpen] = useState(false);
  const [recText, setRecText] = useState("");
  const [recSpec, setRecSpec] = useState("");
  const [recommended, setRecommended] = useState<Lawyer[] | null>(null);
  const [recBusy, setRecBusy] = useState(false);

  useEffect(() => {
    fetchLawyers()
      .then((data) => setAll(Array.isArray(data) ? data : []))
      .catch(() => setError("Couldn't load lawyers. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  const specializations = useMemo(() => Array.from(new Set(all.map((l) => l.specialty))).sort(), [all]);
  const locations = useMemo(() => Array.from(new Set(all.map((l) => l.location))).sort(), [all]);

  const base = recommended ?? all;
  const filtered = base.filter((l) => {
    const s = search.toLowerCase();
    const matchesSearch = !s || l.name.toLowerCase().includes(s) || l.specialty.toLowerCase().includes(s);
    const matchesSpec = !spec || l.specialty === spec;
    const matchesLoc = !loc || l.location === loc;
    return matchesSearch && matchesSpec && matchesLoc;
  });

  const runRecommend = async () => {
    setRecBusy(true);
    try {
      const results = await recommendLawyers({ problemDescription: recText, specialty: recSpec || undefined });
      setRecommended(results);
      setRecOpen(false);
    } catch {
      setError("Recommendation failed.");
    } finally {
      setRecBusy(false);
    }
  };

  const activeChips = [
    spec && { label: spec, clear: () => setSpec("") },
    loc && { label: loc, clear: () => setLoc("") },
    recommended && { label: "Recommended for you", clear: () => setRecommended(null) },
  ].filter(Boolean) as { label: string; clear: () => void }[];

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 1080 }}>
        <h1 className="page-title">Find a lawyer</h1>
        <p className="page-sub">Filter by specialty and location, or describe your case for a ranked match.</p>

        <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 220, display: "flex", alignItems: "center", gap: 8, background: "var(--surface)", border: "1px solid var(--border-2)", borderRadius: 12, padding: "11px 16px" }}>
            <span style={{ color: "var(--muted-3)" }}>⌕</span>
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by name or specialty…" style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "400 14px var(--font-body)", color: "var(--text)" }} />
          </div>
          <div className="filter-select">
            <select value={spec} onChange={(e) => setSpec(e.target.value)}>
              <option value="">Specialization</option>
              {specializations.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="filter-select">
            <select value={loc} onChange={(e) => setLoc(e.target.value)}>
              <option value="">Location</option>
              {locations.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <button className="btn btn-primary" onClick={() => setRecOpen(true)}>Recommend for me</button>
        </div>

        {activeChips.length > 0 && (
          <div style={{ display: "flex", gap: 8, marginBottom: 26, flexWrap: "wrap" }}>
            {activeChips.map((c) => (
              <button key={c.label} className="pill pill-accent" onClick={c.clear} style={{ cursor: "pointer", border: "none" }}>{c.label} ✕</button>
            ))}
          </div>
        )}
        {activeChips.length === 0 && <div style={{ height: 26 }} />}

        {error && <div className="card" style={{ borderColor: "#f0d9d6", color: "var(--danger)", marginBottom: 18 }}>{error}</div>}

        {loading ? (
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div className="shimmer" style={{ height: 46, width: 46, borderRadius: 12 }} />
                <div className="shimmer" style={{ height: 14, width: "60%" }} />
                <div className="shimmer" style={{ height: 12, width: "80%" }} />
                <div className="shimmer" style={{ height: 34, width: "100%", borderRadius: 10 }} />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: "center", borderStyle: "dashed" }}>
            <div style={{ font: "600 15px var(--font-body)", marginBottom: 6 }}>No lawyers match those filters</div>
            <div style={{ font: "400 13.5px var(--font-body)", color: "var(--muted-2)" }}>Try clearing a filter or broadening your search.</div>
          </div>
        ) : (
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            {filtered.map((l) => (
              <LawyerCard key={l.id} lawyer={l} onView={onSelectLawyer} onBook={onBook} />
            ))}
          </div>
        )}
      </div>

      {recOpen && (
        <div className="backdrop" style={{ alignItems: "center" }} onClick={() => setRecOpen(false)}>
          <div className="modal" style={{ width: 480, maxWidth: "90vw", padding: 24 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ font: "700 20px var(--font-head)", marginBottom: 6 }}>Describe your case</h3>
            <p style={{ font: "400 13.5px var(--font-body)", color: "var(--muted-2)", marginBottom: 18 }}>We'll rank lawyers who fit your situation.</p>
            <div className="field">
              <label>What's going on?</label>
              <textarea className="input" value={recText} onChange={(e) => setRecText(e.target.value)} placeholder="e.g. My landlord is withholding my security deposit…" />
            </div>
            <div className="field">
              <label>Area of law (optional)</label>
              <select className="input" value={recSpec} onChange={(e) => setRecSpec(e.target.value)}>
                <option value="">Any</option>
                {specializations.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 8 }}>
              <button className="btn btn-outline" onClick={() => setRecOpen(false)}>Cancel</button>
              <button className="btn btn-primary" disabled={recBusy} onClick={runRecommend}>{recBusy ? "Matching…" : "Show matches"}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
