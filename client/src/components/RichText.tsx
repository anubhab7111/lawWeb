import { Fragment, type ReactNode } from "react";

// Dependency-free renderer for the markdown the backend emits: headings,
// **bold**, *italic*, `code`, [links](https://…), bullet and numbered lists,
// horizontal rules, and the <details>/<summary> blocks of the validation
// report (rendered collapsed; a block still open mid-stream runs to the end).

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

function stripHtmlTags(line: string): string {
  return line.replace(/<\/?(details|summary|strong)>/gi, (m) => (/strong/i.test(m) ? "**" : ""));
}

function renderLine(raw: string, key: string): ReactNode {
  const trimmed = stripHtmlTags(raw).trim();
  if (!trimmed) return raw.trim() ? null : <div key={key} style={{ height: 8 }} />;

  if (/^(-{3,}|\*{3,})$/.test(trimmed)) return <hr key={key} style={{ border: 0, borderTop: "1px solid var(--border)", margin: "12px 0" }} />;

  const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
  if (heading) {
    return (
      <div key={key} style={{ fontWeight: 700, fontSize: heading[1].length <= 2 ? "1.12em" : "1em", margin: "14px 0 4px" }}>
        {renderInline(heading[2])}
      </div>
    );
  }

  const bullet = /^[•\-*]\s+/.test(trimmed);
  const numbered = /^\d+[.)]\s+/.exec(trimmed);
  const content = bullet ? trimmed.replace(/^[•\-*]\s+/, "") : trimmed;
  return (
    <div key={key} style={{ display: "flex", gap: bullet || numbered ? 8 : 0, marginBottom: 2 }}>
      {bullet && <span style={{ color: "var(--accent)", flex: "none" }}>•</span>}
      <span>{renderInline(content)}</span>
    </div>
  );
}

function renderLines(lines: string[], keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*<details>/i.test(lines[i])) {
      out.push(renderLine(lines[i], `${keyPrefix}${i}`));
      continue;
    }
    let end = lines.findIndex((l, j) => j > i && /<\/details>/i.test(l));
    if (end === -1) end = lines.length;
    const block = lines.slice(i + 1, end);
    const s = block.findIndex((l) => /<summary>/i.test(l));
    const summary = s === -1 ? "Details" : stripHtmlTags(block[s]).trim();
    const body = s === -1 ? block : block.filter((_, j) => j !== s);
    out.push(
      <details key={`${keyPrefix}${i}`} style={{ margin: "10px 0" }}>
        <summary style={{ fontWeight: 600, cursor: "pointer" }}>{renderInline(summary)}</summary>
        <div style={{ marginTop: 6 }}>{renderLines(body, `${keyPrefix}${i}.`)}</div>
      </details>,
    );
    i = end;
  }
  return out;
}

export function RichText({ text }: { text: string }) {
  return <>{renderLines((text || "").split("\n"), "")}</>;
}
