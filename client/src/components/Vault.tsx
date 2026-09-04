import { useEffect, useRef, useState } from "react";
import { deleteVaultDocument, fetchVaultDocuments, searchVaultDocuments, uploadVaultDocument, type VaultDocument } from "../api";
import type { UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
}

const DOC_TYPES = ["fir", "order", "agreement", "notice", "evidence", "judgment"];

export function Vault({ user }: Props) {
  const [documents, setDocuments] = useState<VaultDocument[]>([]);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[] | null>(null);
  const [docType, setDocType] = useState(DOC_TYPES[0]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = () => {
    fetchVaultDocuments().then(setDocuments).catch(() => setError("Couldn't load your vault.")).finally(() => setLoading(false));
  };

  useEffect(() => { if (user) load(); }, [user]);

  const handleUpload = async (file: File) => {
    setError(null);
    try {
      await uploadVaultDocument(file, file.name, docType);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) { setSearchResults(null); return; }
    try {
      setSearchResults(await searchVaultDocuments(query));
    } catch {
      setError("Search failed.");
    }
  };

  if (!user) {
    return <div className="container"><p className="page-sub">Sign in to use your Document Vault.</p></div>;
  }

  const list = searchResults ?? documents;

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">Legal Document Vault</h1>
        <p className="page-sub">Secure storage for FIRs, orders, agreements, notices, evidence, and judgments — searchable with AI.</p>

        <div className="card" style={{ padding: 20, marginBottom: 20, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <select className="input" value={docType} onChange={(e) => setDocType(e.target.value)} style={{ width: 160 }}>
            {DOC_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <button className="btn btn-primary" onClick={() => fileInput.current?.click()}>Upload document</button>
          <input
            ref={fileInput}
            type="file"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}
          />
        </div>

        <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
          <input className="input" style={{ flex: 1 }} placeholder="Search your documents…" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSearch()} />
          <button className="btn btn-outline" onClick={handleSearch}>Search</button>
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        {loading ? (
          <div className="shimmer" style={{ height: 140, borderRadius: "var(--r-lg)" }} />
        ) : list.length === 0 ? (
          <div className="empty-state"><div className="empty-sub">No documents yet.</div></div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {list.map((d) => (
              <div key={d.id} className="card" style={{ padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ font: "600 13.5px var(--font-body)" }}>{d.title}</div>
                    <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                      {d.documentType} · {d.indexingStatus === "indexing" ? "indexing…" : d.indexingStatus}
                    </div>
                    {d.matchedSnippet && (
                      <div style={{ font: "400 12.5px var(--font-body)", color: "var(--text-2)", marginTop: 6 }}>…{d.matchedSnippet}…</div>
                    )}
                  </div>
                  <button className="btn btn-sm" onClick={() => deleteVaultDocument(d.id).then(load)}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
