import { Fragment } from "react";

// Minimal, dependency-free renderer for the light markdown the backend emits:
// **bold**, bullet lines (•, -, *), numbered lines, and paragraph breaks.
function renderInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? (
      <strong key={i}>{p.slice(2, -2)}</strong>
    ) : (
      <Fragment key={i}>{p}</Fragment>
    )
  );
}

export function RichText({ text }: { text: string }) {
  const lines = (text || "").split("\n");
  return (
    <>
      {lines.map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={i} style={{ height: 8 }} />;
        const bullet = /^[•\-*]\s+/.test(trimmed);
        const content = bullet ? trimmed.replace(/^[•\-*]\s+/, "") : trimmed;
        return (
          <div key={i} style={{ display: "flex", gap: bullet ? 8 : 0, marginBottom: 2 }}>
            {bullet && <span style={{ color: "var(--accent)", flex: "none" }}>•</span>}
            <span>{renderInline(content)}</span>
          </div>
        );
      })}
    </>
  );
}
