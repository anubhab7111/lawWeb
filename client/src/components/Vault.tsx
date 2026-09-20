import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteVaultDocument,
  downloadVaultDocument,
  fetchVaultDocuments,
  reindexVaultDocument,
  searchVaultDocuments,
  shareVaultDocument,
  uploadVaultDocument,
  type VaultDocument,
} from "../api";
import type { UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
}

const DOC_TYPES = ["fir", "order", "agreement", "notice", "evidence", "judgment"];

const message = (e: unknown, fallback: string) => (e instanceof Error && e.message ? e.message : fallback);

export function Vault({ user }: Props) {
  const [documents, setDocuments] = useState<VaultDocument[]>([]);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<VaultDocument[] | null>(null);
  const [tab, setTab] = useState<"mine" | "shared">("mine");
  const [docType, setDocType] = useState(DOC_TYPES[0]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [shareFor, setShareFor] = useState<VaultDocument | null>(null);
  const [shareEmail, setShareEmail] = useState("");
  const [sharePermission, setSharePermission] = useState<"view" | "edit">("view");
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    return fetchVaultDocuments()
      .then(setDocuments)
      .catch(() => setError("Couldn't load your vault."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { if (user) load(); }, [user, load]);

  // Indexing runs in the background after upload — keep refreshing until it settles.
  const settling = documents.some((d) => d.indexingStatus === "pending" || d.indexingStatus === "indexing");
  useEffect(() => {
    if (!settling) return;
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [settling, load]);

  const run = async (action: () => Promise<unknown>, fallback: string, success?: string) => {
    setError(null);
    setNotice(null);
    try {
      await action();
      if (success) setNotice(success);
      await load();
    } catch (e) {
      setError(message(e, fallback));
    }
  };

  const handleUpload = (file: File) => {
    run(() => uploadVaultDocument(file, file.name, docType), "Upload failed.");
    if (fileInput.current) fileInput.current.value = "";
  };

  const handleSearch = async () => {
    if (!query.trim()) { setSearchResults(null); return; }
    try {
      setSearchResults(await searchVaultDocuments(query));
    } catch {
      setError("Search failed.");
    }
  };

  const handleDelete = (d: VaultDocument) => {
    if (!window.confirm(`Delete "${d.title}"? This can't be undone.`)) return;
    setSearchResults(null);
    run(() => deleteVaultDocument(d.id), "Couldn't delete that document.");
  };

  const handleShare = () => {
    if (!shareFor || !shareEmail.trim()) return;
    const target = shareFor;
    run(
      () => shareVaultDocument(target.id, shareEmail.trim(), sharePermission),
      "Couldn't share that document.",
      `Shared "${target.title}" with ${shareEmail.trim()}.`,
    ).then(() => { setShareFor(null); setShareEmail(""); });
  };

  if (!user) {
    return <div className="container"><p className="page-sub">Sign in to use your Document Vault.</p></div>;
  }

  const visible = (searchResults ?? documents).filter((d) => (tab === "mine" ? d.isOwner : !d.isOwner));

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

        <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
          <input className="input" style={{ flex: 1 }} placeholder="Search your documents…" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSearch()} />
          <button className="btn btn-outline" onClick={handleSearch}>Search</button>
          {searchResults && <button className="btn btn-ghost" onClick={() => { setSearchResults(null); setQuery(""); }}>Clear</button>}
        </div>

        <div className="segmented" style={{ marginBottom: 18 }}>
          {(["mine", "shared"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)} className={tab === t ? "is-on" : ""}>
              {t === "mine" ? "My documents" : "Shared with me"}
            </button>
          ))}
        </div>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}
        {notice && <div className="card" style={{ padding: 12, marginBottom: 18 }}>{notice}</div>}

        {loading ? (
          <div className="shimmer" style={{ height: 140, borderRadius: "var(--r-lg)" }} />
        ) : visible.length === 0 ? (
          <div className="empty-state"><div className="empty-sub">{tab === "mine" ? "No documents yet." : "Nothing has been shared with you yet."}</div></div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {visible.map((d) => (
              <div key={d.id} className="card" style={{ padding: "14px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ font: "600 13.5px var(--font-body)" }}>{d.title}</div>
                    <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                      {d.documentType} · {d.indexingStatus === "indexing" || d.indexingStatus === "pending" ? "indexing…" : d.indexingStatus}
                    </div>
                    {d.matchedSnippet && (
                      <div style={{ font: "400 12.5px var(--font-body)", color: "var(--text-2)", marginTop: 6 }}>…{d.matchedSnippet}…</div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    <button className="btn btn-sm" onClick={() => run(() => downloadVaultDocument(d), "Download failed.")}>Download</button>
                    {d.isOwner && d.indexingStatus === "failed" && (
                      <button className="btn btn-sm" onClick={() => run(() => reindexVaultDocument(d.id), "Couldn't re-index that document.")}>Retry indexing</button>
                    )}
                    {d.isOwner && <button className="btn btn-sm" onClick={() => setShareFor(d)}>Share</button>}
                    {d.isOwner && <button className="btn btn-sm" onClick={() => handleDelete(d)}>Delete</button>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {shareFor && (
        <div className="backdrop" style={{ alignItems: "center" }} onClick={() => setShareFor(null)}>
          <div className="modal" style={{ width: 440, maxWidth: "90vw", padding: 24 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ font: "700 18px var(--font-head)", marginBottom: 6 }}>Share "{shareFor.title}"</h3>
            <p style={{ font: "400 13px var(--font-body)", color: "var(--muted-2)", marginBottom: 16 }}>
              Enter the email of a LawWeb user. They'll see it under "Shared with me".
            </p>
            <div className="field">
              <label>Email</label>
              <input className="input" type="email" value={shareEmail} onChange={(e) => setShareEmail(e.target.value)} placeholder="colleague@example.com" autoFocus />
            </div>
            <div className="field">
              <label>Access</label>
              <select className="input" value={sharePermission} onChange={(e) => setSharePermission(e.target.value as "view" | "edit")}>
                <option value="view">Can view</option>
                <option value="edit">Can edit title and type</option>
              </select>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button className="btn btn-outline" onClick={() => setShareFor(null)}>Cancel</button>
              <button className="btn btn-primary" disabled={!shareEmail.trim()} onClick={handleShare}>Share</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
