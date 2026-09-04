import { useState, useRef } from "react";
import { uploadDocumentForAnalysis, analyzeDocumentText, type ChatResponse } from "../api";
import { RichText } from "./RichText";
import { IconUpload, IconPaperclip } from "./icons";

export function DocumentAnalysis() {
  const [mode, setMode] = useState<"upload" | "paste">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const analyze = async () => {
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const data = mode === "upload" && file
        ? await uploadDocumentForAnalysis(file, "Analyze this document", undefined)
        : await analyzeDocumentText(text, undefined);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Analysis failed.");
    } finally {
      setBusy(false);
    }
  };

  const canSubmit = mode === "upload" ? !!file : text.trim().length >= 10;
  const details = result?.document_validation || result?.document_info;

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 760 }}>
        <h1 className="page-title">Analyze a document</h1>
        <p className="page-sub">Upload a contract or notice — we classify it, check statutory requirements, and flag likely defects.</p>

        {/* mode toggle */}
        <div className="segmented" style={{ marginBottom: 18 }}>
          {(["upload", "paste"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)} className={mode === m ? "is-on" : ""}>
              {m === "upload" ? "Upload file" : "Paste text"}
            </button>
          ))}
        </div>

        {mode === "upload" ? (
          <div
            className={`dropzone${dragOver ? " is-over" : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); setFile(e.dataTransfer.files?.[0] ?? null); }}
          >
            <input ref={fileRef} type="file" hidden accept=".pdf,.doc,.docx,.png,.jpg,.jpeg" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <div style={{ color: "var(--accent)", marginBottom: 12, display: "flex", justifyContent: "center" }}><IconUpload size={30} /></div>
            {file ? (
              <div style={{ font: "600 14.5px var(--font-body)", display: "inline-flex", alignItems: "center", gap: 7 }}><IconPaperclip size={15} /> {file.name}</div>
            ) : (
              <>
                <div style={{ font: "600 15px var(--font-body)" }}>Drop a file here, or click to browse</div>
                <div style={{ font: "400 13px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>PDF, DOCX, or an image (scanned docs are OCR'd)</div>
              </>
            )}
          </div>
        ) : (
          <textarea className="input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the document text here…" style={{ minHeight: 200 }} />
        )}

        <button className="btn btn-primary btn-lg" onClick={analyze} disabled={!canSubmit || busy} style={{ marginTop: 16 }}>
          {busy ? "Analyzing…" : "Analyze document"}
        </button>

        {busy && (
          <div style={{ marginTop: 16, font: "400 13px var(--font-body)", color: "var(--muted-2)" }}>
            Running the classification → statutory-check → defect pipeline. Scanned files are OCR'd first, so this can take a moment.
          </div>
        )}

        {error && <div className="error-banner" style={{ marginTop: 18 }}>{error}</div>}

        {result && (
          <div className="card" style={{ padding: 24, marginTop: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <span className="pill pill-accent">Analysis complete</span>
              {result.intent && <span className="pill pill-neutral" style={{ textTransform: "capitalize" }}>{result.intent.replace(/_/g, " ")}</span>}
            </div>
            <div className="prose" style={{ font: "400 16.5px var(--font-serif)", color: "var(--text-strong)" }}>
              <RichText text={result.response || ""} />
            </div>
            {details && (
              <details style={{ marginTop: 16 }}>
                <summary style={{ cursor: "pointer", font: "500 13px var(--font-body)", color: "var(--muted-2)" }}>Structured findings</summary>
                <pre className="data-pre">{JSON.stringify(details, null, 2)}</pre>
              </details>
            )}
            <div style={{ marginTop: 14, font: "400 12.5px var(--font-body)", color: "var(--muted-3)" }}>Educational information, not legal advice.</div>
          </div>
        )}
      </div>
    </div>
  );
}
