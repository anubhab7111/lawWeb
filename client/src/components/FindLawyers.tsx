import { useState, useEffect } from "react";
import { fetchLawyerFilters, fetchLawyersPage, recommendLawyers } from "../api";
import { LawyerCard } from "./LawyerCard";
import { IconSearch, IconClose } from "./icons";
import type { Lawyer } from "../lib/ui";

interface Props {
  onSelectLawyer: (l: Lawyer) => void;
  onBook: (l: Lawyer) => void;
}

export function FindLawyers({ onSelectLawyer, onBook }: Props) {
  const [all, setAll] = useState<Lawyer[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState<{ specialties: string[]; states: string[] }>({ specialties: [], states: [] });
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
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
    fetchLawyerFilters().then(setFilters).catch(() => {});
  }, []);

  // Filtering happens on the server (the directory has ~1,000 lawyers); the
  // search box is debounced so typing doesn't fire a request per keystroke.
  useEffect(() => {
    const handle = setTimeout(() => {
      setLoading(true);
      setError(null);
      fetchLawyersPage({ page: 1, q: search.trim() || undefined, specialty: spec || undefined, location: loc || undefined })
        .then((res) => { setAll(res.items); setTotal(res.total); setPage(1); })
        .catch(() => setError("Couldn't load lawyers. Is the backend running?"))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [search, spec, loc]);

  const loadMore = () => {
    setLoadingMore(true);
    fetchLawyersPage({ page: page + 1, q: search.trim() || undefined, specialty: spec || undefined, location: loc || undefined })
      .then((res) => { setAll((prev) => [...prev, ...res.items]); setPage(page + 1); })
      .catch(() => setError("Couldn't load more lawyers."))
      .finally(() => setLoadingMore(false));
  };

  const specializations = filters.specialties;
  const locations = filters.states;
  const filtered = recommended ?? all;

  const runRecommend = async () => {
    setRecBusy(true);
    try {
      const results = await recommendLawyers({ problemDescription: recText, specialty: recSpec || undefined });
      setRecommended(results);
      setError(null);
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
          <div className="search-field" style={{ flex: 1, minWidth: 220 }}>
            <IconSearch />
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by name or specialty…" />
          </div>
          <div className="filter-select">
            <select value={spec} onChange={(e) => setSpec(e.target.value)}>
              <option value="">Specialization</option>
              {specializations.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="filter-select">
            <select value={loc} onChange={(e) => setLoc(e.target.value)}>
              <option value="">State</option>
              {locations.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <button className="btn btn-primary" onClick={() => setRecOpen(true)}>Recommend for me</button>
        </div>

        {activeChips.length > 0 && (
          <div style={{ display: "flex", gap: 8, marginBottom: 26, flexWrap: "wrap" }}>
            {activeChips.map((c) => (
              <button key={c.label} className="pill pill-accent" onClick={c.clear} style={{ cursor: "pointer", border: "none" }}>{c.label} <IconClose size={11} /></button>
            ))}
          </div>
        )}
        {activeChips.length === 0 && <div style={{ height: 26 }} />}

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        {loading ? (
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div className="shimmer" style={{ height: 48, width: 48, borderRadius: "var(--r)" }} />
                <div className="shimmer" style={{ height: 14, width: "60%" }} />
                <div className="shimmer" style={{ height: 12, width: "80%" }} />
                <div className="shimmer" style={{ height: 34, width: "100%" }} />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-title">No lawyers match those filters</div>
            <div className="empty-sub">Try clearing a filter or broadening your search.</div>
          </div>
        ) : (
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            {filtered.map((l) => (
              <LawyerCard key={l.id} lawyer={l} onView={onSelectLawyer} onBook={onBook} />
            ))}
          </div>
        )}

        {!recommended && !loading && all.length < total && (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 24 }}>
            <button className="btn btn-outline" disabled={loadingMore} onClick={loadMore}>
              {loadingMore ? "Loading…" : `Show more (${total - all.length} remaining)`}
            </button>
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
