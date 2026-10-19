import { Fragment, type ReactNode } from "react";
import { balanceBold, depthTracker, parseLine, type ParsedLine } from "./listLine";

// Dependency-free renderer for the markdown the backend emits: headings,
// **bold**, *italic*, `code`, [links](https://…), nested bullet and numbered
// lists (including (i)/(a) enumerations),
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

function renderLine(raw: string, line: ParsedLine, depth: number, key: string): ReactNode {
  const text = line.text.trim();
  if (!text) return raw.trim() ? null : <div key={key} style={{ height: 8 }} />;

  if (/^(-{3,}|\*{3,})$/.test(text)) return <hr key={key} style={{ border: 0, borderTop: "1px solid var(--border)", margin: "12px 0" }} />;
  const trimmed = balanceBold(text);

  const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
  if (heading) {
    return (
      <div key={key} style={{ fontWeight: 700, fontSize: heading[1].length <= 2 ? "1.12em" : "1em", margin: "14px 0 4px" }}>
        {renderInline(heading[2])}
      </div>
    );
  }

  if (!line.marker) {
    return (
      <div key={key} style={{ marginBottom: 2 }}>
        <span>{renderInline(trimmed)}</span>
      </div>
    );
  }

  const bullet = line.marker.kind === "bullet";
  return (
    <div key={key} style={{ display: "flex", gap: 8, marginBottom: 2, marginLeft: `${depth * 1.4}em` }}>
      <span style={bullet ? { color: "var(--accent)", flex: "none" } : { flex: "none", minWidth: "1.7em", fontVariantNumeric: "tabular-nums" }}>
        {bullet ? (depth ? "◦" : "•") : line.marker.label}
      </span>
      <span>{renderInline(trimmed)}</span>
    </div>
  );
}

function renderLines(lines: string[], keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  const depthOf = depthTracker();
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*<details>/i.test(lines[i])) {
      const line = parseLine(stripHtmlTags(lines[i]));
      out.push(renderLine(lines[i], line, depthOf(line), `${keyPrefix}${i}`));
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
