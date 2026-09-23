"""
Text quality scoring and cleaning for Supreme Court judgment PDFs.

The PDFs are law-report scans with an OCR text layer: margin marker letters
(A, B, C ... down the side of each page), running headers, page numbers,
hyphenated line breaks, and paragraphs that only show up as indentation in
`pdftotext -layout` output. `clean_judgment_text` turns that into plain
paragraphs separated by blank lines; `text_quality` scores the raw extraction
so unusable documents are quarantined instead of indexed.
"""

from __future__ import annotations

import re
from typing import Dict, List

MIN_CHARS_PER_PAGE = 400
MAX_GARBLED_RATIO = 0.06
MIN_ALPHA_RATIO = 0.70

_MARGIN_LEAD = re.compile(r"^[A-Ha-h](?:\s{3,}|\s*$)")
_MARGIN_TRAIL = re.compile(r"\s{5,}[A-Ha-h][,.;:]?\s*$")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")
_RUNNING_HEADER = re.compile(
    r"supreme\s+court\s+reports?|\[\d{4}\]\s*(?:suppl?\.?\s*)?\d*\s*s\.c\.r\.", re.IGNORECASE
)
_GARBLED_TOKEN = re.compile(r"[A-Za-z]{2,}[\d\[\]!£|~;][A-Za-z]{2,}|[A-Za-z]+[!£~|]+[A-Za-z]*")
_HYPHEN_BREAK = re.compile(r"(\w)-$")


def text_quality(text: str, pages: int) -> Dict:
    """Score raw extracted text; `ok` False carries a `reason` for quarantine."""
    chars = len(text.strip())
    pages = max(pages, 1)
    tokens = text.split()
    letters = sum(c.isalpha() for c in text)
    non_space = sum(not c.isspace() for c in text) or 1
    garbled = sum(1 for t in tokens if _GARBLED_TOKEN.search(t))
    metrics = {
        "chars": chars,
        "pages": pages,
        "chars_per_page": chars / pages,
        "alpha_ratio": letters / non_space,
        "garbled_ratio": garbled / max(len(tokens), 1),
    }
    reason = None
    if metrics["chars_per_page"] < MIN_CHARS_PER_PAGE:
        reason = "no text layer"
    elif metrics["alpha_ratio"] < MIN_ALPHA_RATIO:
        reason = "low alphabetic ratio"
    elif metrics["garbled_ratio"] > MAX_GARBLED_RATIO:
        reason = "garbled OCR"
    return {**metrics, "ok": reason is None, "reason": reason}


def _strip_margins(line: str) -> str:
    line = _MARGIN_TRAIL.sub("", line)
    # Blank the marker in place: the indentation still tells paragraph starts apart.
    line = _MARGIN_LEAD.sub(lambda m: " " * len(m.group(0)), line)
    return line.rstrip()


# Page-top/bottom case-name headers: "BALKRISHNA v. SWADESHI POLYTEX 855".
_CASE_HEADER = re.compile(r"^\s*(?:\d{1,4}\s+)?[A-Z][A-Z .,&'()-]{2,45}\sv\.?\s[A-Z .,&'()-]{2,45}(?:\s+\S{1,4})?\s*$")


def _is_noise(line: str) -> bool:
    if _PAGE_NUMBER.match(line):
        return True
    if len(line) < 110 and _RUNNING_HEADER.search(line):
        return True
    return len(line) < 100 and bool(_CASE_HEADER.match(line)) and bool(re.search(r"\d", line))


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


_SENTENCE_END = re.compile(r"""[.;:?!"')\]]\s*$""")
_NUMBERED_PARA = re.compile(r"^\d{1,3}(?:\.\d{1,3})*\.?\s+\S")


def _page_base(lines: List[str]) -> int:
    """Body indent of a page: a low percentile, since paragraph-first lines
    (indented further) must not be mistaken for the body."""
    indents = sorted(_indent(l) for l in lines if l.strip() and _indent(l) < 40)
    return indents[len(indents) // 10] if indents else 0


def _paragraph_lines(raw: str) -> List[List[str]]:
    """Group layout lines into paragraphs.

    A paragraph can only start after a line that ended a sentence, and then
    on a blank line, a numbered-paragraph marker, or an indent jump measured
    against that page's own body indent (layouts differ page to page, so no
    absolute threshold works). Everything else is a wrapped line.
    """
    paragraphs: List[List[str]] = []
    current: List[str] = []
    pending_blank = False
    for page in raw.split("\f"):
        lines = [_strip_margins(l.expandtabs(4)) for l in page.splitlines()]
        lines = [l for l in lines if not (l.strip() and _is_noise(l))]
        base = _page_base(lines)
        for line in lines:
            if not line.strip():
                pending_blank = True
                continue
            if current and _SENTENCE_END.search(current[-1]):
                jump = _indent(line) - base >= 4 and _indent(line) > _indent(current[-1]) + 2
                if pending_blank or jump or _NUMBERED_PARA.match(line.strip()):
                    paragraphs.append(current)
                    current = []
            current.append(line.strip())
            pending_blank = False
    if current:
        paragraphs.append(current)
    return paragraphs


def clean_judgment_text(raw: str) -> str:
    out: List[str] = []
    for para in _paragraph_lines(raw):
        text = ""
        for line in para:
            if text and _HYPHEN_BREAK.search(text) and line[:1].islower():
                text = text[:-1] + line
            else:
                text = f"{text} {line}" if text else line
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return "\n\n".join(out)
