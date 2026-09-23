"""
Geometry-based text extraction for Supreme Court judgment PDFs.

`pdftotext -bbox-layout` gives every text line — and every word in it — with
coordinates. The law-report template puts margin marker letters (A-H) in a column
just outside the body text, running headers at the top of each page, and
paragraph starts indented. All of that is unambiguous by position, where text-only
heuristics leak stray letters ("under the D provisions") and header fragments into
sentences. Markers can be separate lines or sit inside a body line's word list, so
they are removed per word.
"""

from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.ingest.judgment_clean import (
    _HYPHEN_BREAK,
    _NUMBERED_PARA,
    _PAGE_NUMBER,
    _RUNNING_HEADER,
    _SENTENCE_END,
)

_PAGE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)
_LINE = re.compile(
    r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', re.S
)
_WORD = re.compile(r'<word xMin="([\d.]+)" yMin="[\d.]+" xMax="([\d.]+)" yMax="[\d.]+">(.*?)</word>', re.S)
_REF_END = re.compile(r"\(\d+(?:-[A-H])+[)\]\[]?\s*$")

HEADER_BAND = 0.105  # fraction of page height at the top that holds running headers
FOOTER_BAND = 0.945
INDENT_PT = 8.0  # a paragraph-first line starts at least this far right of the body
MARGIN_SLACK = 3.0  # short words this far outside the body edges are margin marks
MARGIN_WORD_MAX_LEN = 2

Word = Tuple[float, float, str]  # x0, x1, text


@dataclass
class Line:
    x0: float
    x1: float
    y0: float
    y1: float
    words: List[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w[2] for w in self.words)


@dataclass
class Page:
    width: float
    height: float
    lines: List[Line]


def parse_bbox_xml(xml: str) -> List[Page]:
    pages: List[Page] = []
    for width, height, body in _PAGE.findall(xml):
        lines = []
        for x0, y0, x1, y1, inner in _LINE.findall(body):
            words = [
                (float(a), float(b), html.unescape(t).strip())
                for a, b, t in _WORD.findall(inner)
                if html.unescape(t).strip()
            ]
            if words:
                lines.append(Line(float(x0), float(x1), float(y0), float(y1), words))
        lines.sort(key=lambda l: (round(l.y0), l.x0))
        pages.append(Page(float(width), float(height), lines))
    return pages


def extract_pages(pdf_path: str, timeout: int = 240) -> List[Page]:
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", pdf_path, "-"],
        capture_output=True, text=True, timeout=timeout, check=True,
    ).stdout
    return parse_bbox_xml(xml)


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def _body_edges(page: Page) -> Optional[Tuple[float, float]]:
    """Left/right edges of the body text column, from words long enough to be
    real text (so margin letters don't pull the estimate outward)."""
    lefts: List[float] = []
    rights: List[float] = []
    for line in page.lines:
        if line.x1 - line.x0 <= 0.45 * page.width:
            continue
        solid = [w for w in line.words if len(w[2]) > MARGIN_WORD_MAX_LEN]
        if solid:
            lefts.append(solid[0][0])
            rights.append(solid[-1][1])
    if len(lefts) < 4:
        return None
    return _percentile(lefts, 0.2), _percentile(rights, 0.8)


def _strip_margin_words(line: Line, edges: Optional[Tuple[float, float]], page: Page) -> Optional[Line]:
    if edges:
        left, right = edges

        def outside(w: Word) -> bool:
            return w[0] > right + MARGIN_SLACK or w[1] < left - MARGIN_SLACK

    else:

        def outside(w: Word) -> bool:
            return w[0] > 0.85 * page.width or w[1] < 0.1 * page.width

    words = [w for w in line.words if not (len(w[2]) <= MARGIN_WORD_MAX_LEN and outside(w))]
    if edges:  # anything far outside the column that survived is a stray mark
        words = [w for w in words if w[1] >= edges[0] - 25]
    if not words:
        return None
    return Line(words[0][0], words[-1][1], line.y0, line.y1, words)


def page_lines(page: Page, first_page: bool) -> List[Line]:
    """Body lines of a page: margin marks, running headers and page numbers removed."""
    edges = _body_edges(page)
    kept: List[Line] = []
    for raw in page.lines:
        line = _strip_margin_words(raw, edges, page)
        if line is None:
            continue
        text = line.text
        if not first_page and len(text) < 90 and (
            line.y0 < HEADER_BAND * page.height or line.y1 > FOOTER_BAND * page.height
        ):
            continue
        if _PAGE_NUMBER.match(text) or (len(text) < 110 and _RUNNING_HEADER.search(text)):
            continue
        kept.append(line)
    return kept


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text) or _REF_END.search(text))


def pages_to_paragraphs(pages: List[Page]) -> List[str]:
    paragraphs: List[List[str]] = []
    current: List[str] = []
    prev: Optional[Line] = None
    for index, page in enumerate(pages):
        lines = page_lines(page, first_page=index == 0)
        edges = _body_edges(page)
        left = edges[0] if edges else min((l.x0 for l in lines), default=0.0)
        for line in lines:
            text = line.text
            if current and prev is not None and _ends_sentence(current[-1]):
                # Across a page break line.y0 < prev.y1, so this is never a gap there.
                gap = line.y0 - prev.y1 > 0.9 * (prev.y1 - prev.y0)
                indented = line.x0 - left >= INDENT_PT
                if indented or gap or _NUMBERED_PARA.match(text):
                    paragraphs.append(current)
                    current = []
            current.append(text)
            prev = line
    if current:
        paragraphs.append(current)

    out: List[str] = []
    for para in paragraphs:
        joined = ""
        for text in para:
            if joined and _HYPHEN_BREAK.search(joined) and text[:1].islower():
                joined = joined[:-1] + text
            else:
                joined = f"{joined} {text}" if joined else text
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            out.append(joined)
    return out


def plain_text(pages: List[Page]) -> str:
    """All extracted lines, pages separated by form feeds (for quality scoring)."""
    return "\f".join("\n".join(l.text for l in p.lines) for p in pages)
