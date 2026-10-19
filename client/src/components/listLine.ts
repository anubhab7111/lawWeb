export type ListMarker = { kind: "bullet" | "ordered"; label: string };
export type ParsedLine = { indent: number; marker: ListMarker | null; text: string };

const BULLET = /^[•\-*+]\s+/;
// 1. / 1) / (1) / (a) / (iv) / a) / iv) — the forms statutes and the model use for enumerations,
// in any script: translated answers come back with (क), (१), (ক) and the like.
const ORDERED = /^(\p{Nd}{1,3}[.)]|\((?:\p{Nd}{1,3}|[ivxlc]{1,6}|\p{L}\p{M}*)\)|(?:[ivxlc]{1,6}|\p{L}\p{M}*)\))\s+/iu;

export function parseLine(raw: string): ParsedLine {
  const lead = /^[ \t]*/.exec(raw)![0];
  const indent = lead.replace(/\t/g, "    ").length;
  const rest = raw.slice(lead.length).trimEnd();
  const bullet = BULLET.exec(rest);
  if (bullet) return { indent, marker: { kind: "bullet", label: "•" }, text: rest.slice(bullet[0].length) };
  const ordered = ORDERED.exec(rest);
  if (ordered) return { indent, marker: { kind: "ordered", label: ordered[1] }, text: rest.slice(ordered[0].length) };
  return { indent, marker: null, text: rest };
}

// While an answer streams, a bold label's closing ** arrives a few tokens after the opening one;
// close it early (or drop a bare trailing **) so raw asterisks never flash on screen.
export function balanceBold(text: string): string {
  const parts = text.split("**");
  if (parts.length % 2 === 1) return text;
  return parts[parts.length - 1] ? `${text}**` : text.slice(0, -2);
}

// Nesting comes from indentation relative to the list items still open above, so 2-space and
// 4-space styles both step one level at a time; an unindented paragraph closes the list.
export function depthTracker(): (line: ParsedLine) => number {
  const open: number[] = [];
  return (line) => {
    if (!line.marker) {
      if (line.text && line.indent === 0) open.length = 0;
      return 0;
    }
    while (open.length && open[open.length - 1] >= line.indent) open.pop();
    open.push(line.indent);
    return open.length - 1;
  };
}
