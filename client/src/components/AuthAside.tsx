import type { ReactNode } from "react";

interface Props {
  eyebrow: string;
  title: ReactNode;
  body: string;
}

// The ink brand panel shown beside the sign-in / sign-up forms. Hidden on
// narrow viewports (via .auth-aside in theme.css) so the form takes the full
// width on phones.
export function AuthAside({ eyebrow, title, body }: Props) {
  return (
    <aside className="auth-aside">
      <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
        <span className="brand-logo" style={{ borderColor: "var(--accent-line)", color: "var(--accent-line)" }}>L</span>
        <span style={{ font: "600 20px var(--font-head)", letterSpacing: ".04em", color: "var(--ink-fg)" }}>LawWeb</span>
      </div>

      <div style={{ marginTop: "auto" }}>
        <div className="eyebrow" style={{ color: "var(--accent-line)", marginBottom: 20 }}>{eyebrow}</div>
        <h2 style={{ font: "600 clamp(30px, 3.4vw, 42px)/1.12 var(--font-head)", letterSpacing: "-.02em", color: "var(--ink-fg)", maxWidth: 420 }}>
          {title}
        </h2>
        <p style={{ font: "400 15.5px/1.6 var(--font-serif)", color: "color-mix(in srgb, var(--ink-fg) 68%, transparent)", marginTop: 20, maxWidth: 380 }}>
          {body}
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, font: "400 12.5px var(--font-body)", color: "color-mix(in srgb, var(--ink-fg) 55%, transparent)" }}>
        <span className="rule-gold" style={{ width: 28 }} />
        Grounded in the Indian bare acts.
      </div>
    </aside>
  );
}
