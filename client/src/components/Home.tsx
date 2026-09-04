import { useState } from "react";
import type { View } from "../App";

interface HomeProps {
  onNavigate: (v: View) => void;
  onAsk: (q: string) => void;
}

const ENTRY_CARDS = [
  {
    n: "01",
    title: "Counsel, on call",
    body: "Ask in plain language and receive a considered answer — cited to the exact Indian acts and sections it rests on.",
    cta: "Begin a consultation",
    view: "chat" as View,
  },
  {
    n: "02",
    title: "The right advocate",
    body: "Search a curated directory by practice, jurisdiction and language, and reserve a consultation in moments.",
    cta: "Enter the directory",
    view: "lawyers" as View,
  },
  {
    n: "03",
    title: "Read the fine print",
    body: "Submit a contract or notice and receive a statutory-requirement review — obligations, gaps and next steps.",
    cta: "Submit a document",
    view: "documents" as View,
  },
];

const STEPS = [
  { n: "I", title: "Describe the matter", body: "In your own words. No legal vocabulary required — the nuance is ours to translate." },
  { n: "II", title: "Receive grounded counsel", body: "Cited to the relevant Indian acts and sections, with a candid indication of confidence." },
  { n: "III", title: "Decide your next move", body: "Review a document, understand a notice, or retain an advocate when the moment calls for one." },
];

const PRACTICE = [
  "Bail eligibility", "Rent & tenancy", "Consumer complaints",
  "Divorce & maintenance", "Trademark filing", "Employment rights",
  "Cheque bounce", "Property transfer",
];

export function Home({ onNavigate, onAsk }: HomeProps) {
  const [q, setQ] = useState("");

  return (
    <div className="fade-in" style={{ flex: 1 }}>
      {/* ── hero ─────────────────────────────────────────────── */}
      <section style={{ display: "flex", justifyContent: "center", padding: "clamp(72px, 12vh, 132px) 24px 0" }}>
        <div style={{ width: "100%", maxWidth: 760, display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
          <div className="eyebrow" style={{ marginBottom: 26 }}>Indian Law · Retrieval-backed Counsel</div>

          <h1 style={{ font: "600 clamp(40px, 6.4vw, 68px)/1.06 var(--font-head)", letterSpacing: "-.025em", maxWidth: 720 }}>
            Understand exactly
            <br />
            where you <span style={{ fontStyle: "italic", color: "var(--accent)" }}>stand.</span>
          </h1>

          <p style={{ font: "400 clamp(16px, 2vw, 19px)/1.65 var(--font-serif)", color: "var(--muted)", marginTop: 24, maxWidth: 540 }}>
            Considered answers to legal questions, drawn from the Indian bare acts and
            grounded in the sections that govern your situation.
          </p>

          <hr className="rule-gold" style={{ margin: "34px auto 34px", transformOrigin: "center" }} />

          {/* hero ask */}
          <form
            onSubmit={(e) => { e.preventDefault(); if (q.trim()) onAsk(q.trim()); }}
            style={{ display: "flex", alignItems: "center", gap: 8, background: "var(--surface)", border: "1px solid var(--border-2)", borderRadius: "var(--r)", padding: "9px 9px 9px 20px", boxShadow: "var(--shadow-soft)", width: "100%", maxWidth: 520 }}
          >
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Ask a legal question…"
              style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "400 15px var(--font-body)", color: "var(--text)" }}
            />
            <button type="submit" className="btn btn-primary" style={{ padding: "11px 22px" }}>Ask&nbsp;&rarr;</button>
          </form>

          {/* practice quick-asks */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 22, maxWidth: 640 }}>
            {PRACTICE.map((p) => (
              <button key={p} className="chip" style={{ padding: "7px 13px", fontSize: 12.5 }} onClick={() => onAsk(p)}>{p}</button>
            ))}
          </div>

          <div style={{ marginTop: 30, font: "400 12.5px var(--font-body)", color: "var(--muted-3)", letterSpacing: ".02em" }}>
            Educational information, grounded in Indian bare acts — not a substitute for a lawyer.
          </div>
        </div>
      </section>

      {/* ── the three services ───────────────────────────────── */}
      <section style={{ display: "flex", justifyContent: "center", padding: "clamp(72px, 12vh, 120px) 24px 0" }}>
        <div style={{ width: "100%", maxWidth: 1040 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>The practice</div>
          <h2 style={{ font: "600 clamp(24px, 3.4vw, 34px) var(--font-head)", letterSpacing: "-.02em", marginBottom: 34, maxWidth: 560 }}>
            Three ways to move forward with confidence.
          </h2>

          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, background: "var(--border)", border: "1px solid var(--border)", borderRadius: "var(--r-lg)", overflow: "hidden" }}>
            {ENTRY_CARDS.map((c) => (
              <button
                key={c.title}
                onClick={() => onNavigate(c.view)}
                className="lw-service"
                style={{ textAlign: "left", background: "var(--surface)", border: "none", cursor: "pointer", padding: "34px 30px 30px", display: "flex", flexDirection: "column", minHeight: 260, transition: "background .2s" }}
              >
                <span style={{ font: "500 13px var(--font-head)", color: "var(--accent)", letterSpacing: ".1em" }}>{c.n}</span>
                <div style={{ font: "600 21px var(--font-head)", letterSpacing: "-.01em", margin: "16px 0 12px" }}>{c.title}</div>
                <div style={{ font: "400 14.5px/1.65 var(--font-serif)", color: "var(--muted)", flex: 1 }}>{c.body}</div>
                <span style={{ font: "600 12.5px var(--font-body)", letterSpacing: ".04em", color: "var(--text-strong)", display: "inline-flex", alignItems: "center", gap: 6, marginTop: 22 }}>
                  {c.cta} <span style={{ color: "var(--accent)" }}>&rarr;</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── pull quote ───────────────────────────────────────── */}
      <section style={{ display: "flex", justifyContent: "center", padding: "clamp(80px, 13vh, 128px) 24px" }}>
        <blockquote style={{ maxWidth: 780, textAlign: "center", margin: 0 }}>
          <p style={{ font: "500 clamp(24px, 3.6vw, 38px)/1.32 var(--font-head)", letterSpacing: "-.02em", color: "var(--text-strong)" }}>
            “The law is reason, free from passion.”
          </p>
          <footer style={{ font: "500 12px var(--font-body)", letterSpacing: ".16em", textTransform: "uppercase", color: "var(--muted-2)", marginTop: 22 }}>
            Aristotle
          </footer>
        </blockquote>
      </section>

      {/* ── how it works ─────────────────────────────────────── */}
      <section style={{ display: "flex", justifyContent: "center", padding: "0 24px clamp(72px, 12vh, 120px)" }}>
        <div style={{ width: "100%", maxWidth: 1040 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>How it works</div>
          <h2 style={{ font: "600 clamp(24px, 3.4vw, 34px) var(--font-head)", letterSpacing: "-.02em", marginBottom: 42, maxWidth: 520 }}>
            From an unformed worry to a clear next step.
          </h2>
          <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 40 }}>
            {STEPS.map((s) => (
              <div key={s.n} style={{ borderTop: "1px solid var(--border-2)", paddingTop: 22 }}>
                <div style={{ font: "500 15px var(--font-head)", color: "var(--accent)", letterSpacing: ".08em", marginBottom: 16 }}>{s.n}</div>
                <div style={{ font: "600 18px var(--font-head)", letterSpacing: "-.01em", marginBottom: 10 }}>{s.title}</div>
                <div style={{ font: "400 14px/1.7 var(--font-serif)", color: "var(--muted)" }}>{s.body}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── closing CTA ──────────────────────────────────────── */}
      <section style={{ display: "flex", justifyContent: "center", padding: "0 24px clamp(96px, 16vh, 150px)" }}>
        <div style={{ width: "100%", maxWidth: 1040, background: "var(--ink)", borderRadius: "var(--r-2xl)", padding: "clamp(48px, 7vw, 84px)", textAlign: "center", position: "relative", overflow: "hidden" }}>
          <div className="eyebrow" style={{ color: "var(--accent-line)", marginBottom: 22 }}>LawWeb</div>
          <h2 style={{ font: "600 clamp(28px, 4.4vw, 46px)/1.1 var(--font-head)", letterSpacing: "-.02em", color: "var(--ink-fg)", maxWidth: 620, margin: "0 auto" }}>
            Your first question is waiting to be answered.
          </h2>
          <p style={{ font: "400 16px/1.6 var(--font-serif)", color: "color-mix(in srgb, var(--ink-fg) 70%, transparent)", maxWidth: 460, margin: "20px auto 0" }}>
            No account required to begin. Ask, understand, and decide on your own terms.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 34, flexWrap: "wrap" }}>
            <button className="btn btn-gold btn-lg" onClick={() => onNavigate("chat")}>Ask the Legal AI</button>
            <button
              className="btn btn-lg"
              onClick={() => onNavigate("lawyers")}
              style={{ background: "transparent", color: "var(--ink-fg)", border: "1px solid color-mix(in srgb, var(--ink-fg) 32%, transparent)" }}
            >
              Browse lawyers
            </button>
          </div>
        </div>
      </section>

      {/* ── footer ───────────────────────────────────────────── */}
      <footer style={{ borderTop: "1px solid var(--border)", padding: "34px 28px", display: "flex", justifyContent: "center" }}>
        <div style={{ width: "100%", maxWidth: 1040, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="brand-logo" style={{ width: 26, height: 26, fontSize: 14 }}>L</span>
            <span style={{ font: "600 16px var(--font-head)", letterSpacing: ".04em" }}>LawWeb</span>
          </div>
          <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>
            © {new Date().getFullYear()} LawWeb · Educational information, not legal advice.
          </div>
        </div>
      </footer>
    </div>
  );
}
