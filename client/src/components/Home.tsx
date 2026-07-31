import { useState } from "react";
import type { View } from "../App";

interface HomeProps {
  onNavigate: (v: View) => void;
  onAsk: (q: string) => void;
}

const ENTRY_CARDS = [
  { icon: "?", title: "Ask the Legal AI", body: "Get plain-language answers to legal questions, with the exact sections cited.", cta: "Start a conversation", view: "chat" as View },
  { icon: "⚖", title: "Find Lawyers", body: "Search by specialty, location, and language, and book a consultation.", cta: "Browse the directory", view: "lawyers" as View },
  { icon: "▤", title: "Analyze a Document", body: "Upload a contract or notice and get a statutory-requirement checklist.", cta: "Upload a document", view: "documents" as View },
];

const STEPS = [
  { n: "01", title: "Describe your situation", body: "In your own words — no legal terminology needed." },
  { n: "02", title: "Get a grounded answer", body: "Cited to the relevant Indian acts and sections, with a confidence indicator." },
  { n: "03", title: "Take the next step", body: "Report an incident, review a document, or book a lawyer if you need one." },
];

export function Home({ onNavigate, onAsk }: HomeProps) {
  const [q, setQ] = useState("");

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div style={{ width: "100%", maxWidth: 920, padding: "72px 24px 80px", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
        <h1 style={{ font: "700 42px/1.2 var(--font-head)", maxWidth: 680 }}>Understand your legal position</h1>
        <p style={{ font: "400 17px/1.6 var(--font-body)", color: "var(--muted)", marginTop: 16, maxWidth: 520 }}>
          Ask questions in plain language, analyze documents, and find a lawyer — grounded in Indian bare acts.
        </p>

        {/* hero search */}
        <form
          onSubmit={(e) => { e.preventDefault(); if (q.trim()) onAsk(q.trim()); }}
          style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 32, background: "var(--surface)", border: "1px solid var(--border-2)", borderRadius: 14, padding: "8px 8px 8px 20px", boxShadow: "var(--shadow-soft)", width: "100%", maxWidth: 480 }}
        >
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Type your legal question…"
            style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "400 14.5px var(--font-body)", color: "var(--text)" }}
          />
          <button type="submit" className="btn btn-primary" style={{ padding: "11px 20px" }}>Ask →</button>
        </form>

        {/* entry cards */}
        <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 18, marginTop: 72, width: "100%", maxWidth: 640, textAlign: "left" }}>
          {ENTRY_CARDS.map((c) => (
            <button key={c.title} className="card card-hover" onClick={() => onNavigate(c.view)} style={{ padding: 24, cursor: "pointer", textAlign: "left" }}>
              <div style={{ width: 38, height: 38, borderRadius: 10, background: "var(--accent-tint)", display: "flex", alignItems: "center", justifyContent: "center", font: "700 15px var(--font-head)", color: "var(--accent)", marginBottom: 14 }}>{c.icon}</div>
              <div style={{ font: "700 16px var(--font-head)", marginBottom: 6 }}>{c.title}</div>
              <div style={{ font: "400 13.5px/1.6 var(--font-body)", color: "var(--muted-2)", marginBottom: 14 }}>{c.body}</div>
              <span style={{ font: "600 13px var(--font-body)", color: "var(--accent)", display: "inline-flex", alignItems: "center", gap: 4 }}>{c.cta} →</span>
            </button>
          ))}
        </div>

        {/* how it works */}
        <div style={{ marginTop: 80, width: "100%" }}>
          <div className="eyebrow" style={{ marginBottom: 22, textAlign: "left" }}>How it works</div>
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 24, textAlign: "left" }}>
            {STEPS.map((s) => (
              <div key={s.n} style={{ padding: 14, borderRadius: 12 }}>
                <div style={{ font: "700 20px var(--font-head)", color: "var(--accent)", marginBottom: 8 }}>{s.n}</div>
                <div style={{ font: "600 14.5px var(--font-body)", marginBottom: 4 }}>{s.title}</div>
                <div style={{ font: "400 13px/1.6 var(--font-body)", color: "var(--muted-2)" }}>{s.body}</div>
              </div>
            ))}
          </div>
        </div>

        {/* trust note */}
        <div style={{ marginTop: 64, width: "100%", background: "var(--surface-alt)", border: "1px solid var(--border)", borderRadius: 14, padding: "18px 22px", font: "400 13px/1.6 var(--font-body)", color: "var(--muted)", textAlign: "left" }}>
          Educational information, grounded in Indian bare acts — not a substitute for a lawyer.
        </div>
      </div>
    </div>
  );
}
