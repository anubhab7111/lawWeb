import { Fragment, type ReactNode } from "react";

// Dependency-free renderer for the markdown the backend emits: headings,
// **bold**, *italic*, `code`, [links](https://…), bullet and numbered lists,
// horizontal rules, and the <details>/<summary> blocks of the validation
// report (shown as a titled section).

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;

function renderInline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    const link = /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/.exec(part);
    if (link) {
      return <a key={i} href={link[2]} target="_blank" rel="noopener noreferrer">{link[1]}</a>;
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

function stripHtmlTags(line: string): { text: string; summary: boolean } {
  const summary = /<summary>/i.test(line);
  return { text: line.replace(/<\/?(details|summary|strong)>/gi, (m) => (/strong/i.test(m) ? "**" : "")), summary };
}

export function RichText({ text }: { text: string }) {
  const lines = (text || "").split("\n");
  return (
    <>
      {lines.map((raw, i) => {
        const { text: line, summary } = stripHtmlTags(raw);
        const trimmed = line.trim();
        if (!trimmed) return raw.trim() ? null : <div key={i} style={{ height: 8 }} />;

        if (/^(-{3,}|\*{3,})$/.test(trimmed)) return <hr key={i} style={{ border: 0, borderTop: "1px solid var(--border)", margin: "12px 0" }} />;

        const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
        if (heading) {
          return (
            <div key={i} style={{ fontWeight: 700, fontSize: heading[1].length <= 2 ? "1.12em" : "1em", margin: "14px 0 4px" }}>
              {renderInline(heading[2])}
            </div>
          );
        }
        if (summary) {
          return <div key={i} style={{ fontWeight: 600, margin: "10px 0 2px" }}>{renderInline(trimmed)}</div>;
        }

        const bullet = /^[•\-*]\s+/.test(trimmed);
        const numbered = /^\d+[.)]\s+/.exec(trimmed);
        const content = bullet ? trimmed.replace(/^[•\-*]\s+/, "") : trimmed;
        return (
          <div key={i} style={{ display: "flex", gap: bullet || numbered ? 8 : 0, marginBottom: 2 }}>
            {bullet && <span style={{ color: "var(--accent)", flex: "none" }}>•</span>}
            <span>{renderInline(content)}</span>
          </div>
        );
      })}
    </>
  );
}
