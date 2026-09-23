"""
Structure-aware chunking of cleaned Supreme Court judgments.

A judgment is split into its headnote (the reporter's summary of the law — dense
with statute references) and its body. Chunks are built from whole paragraphs,
merged up to a word budget; an over-long paragraph is split at sentence
boundaries with one sentence of overlap, never mid-sentence. Every chunk carries
the case metadata and the statute citations found in its own text, so retrieval
results can be cited ("Case, [1985] 3 S.C.R. 985") and filtered by section.

Chunk `text` is the clean passage only; callers add any embedding prefix.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.ingest.citations import extract_citations

PIPELINE_VERSION = "1"
TARGET_WORDS = 260
MAX_PARA_WORDS = 380
MIN_CHUNK_WORDS = 15
MAX_CHUNK_WORDS = 700
ROLES = ("headnote", "body", "order")

_BENCH = re.compile(r"\[[^\]]{4,160}\b(?:JJ?|C\.?\s?J)\.?\s*\]")
_JURISDICTION = re.compile(
    r"^(?:CIVIL|CRIMINAL|ORIGINAL|SPECIAL|WRIT|TRANSFER|ADVISORY|REVIEW)[A-Z ,/()-]{0,60}JURISDICTION"
    r"|^CASE LAW REFERENCE",
    re.IGNORECASE,
)
_DELIVERED_BY = re.compile(r"delivered\s+by\b", re.IGNORECASE)
_COUNSEL = re.compile(
    r"\bfor\s+the\s+(?:appell|respond|petition|interven|state|union|caveat|complain|accused)|"
    r"\b(?:Senior\s+)?Advocates?\b|\bAdvocate[\s-]General\b|\bAttorney[\s-]General\b|"
    r"\bSolicitor[\s-]General\b|\b[A-Z][a-z]+\s+(?:and|with)\s+.*\bfor\s+the\b",
)
_ORDER = re.compile(
    r"\b(?:appeals?|petitions?|applications?|writ\s+petitions?)\s+(?:is|are|stands?|shall\s+stand)\s+"
    r"(?:accordingly\s+)?(?:allowed|dismissed|disposed)|"
    r"\bwe\s+(?:accordingly\s+)?(?:allow|dismiss|set\s+aside|direct)\b|\bno\s+order\s+as\s+to\s+costs\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z0-9(\"'\[])")


class ChunkValidationError(ValueError):
    pass


def _words(text: str) -> int:
    return len(text.split())


def split_sections(paragraphs: List[str]) -> Dict[str, List[str]]:
    """Separate a judgment's paragraphs into header, headnote and body.

    The body starts after "... delivered by" when the reporter's front matter has
    one, else after the jurisdiction line and any counsel lines. Documents with
    none of these markers are treated as all body.
    """
    limit = min(len(paragraphs), 45)
    delivered = next((i for i in range(limit) if _DELIVERED_BY.search(paragraphs[i]) and _words(paragraphs[i]) < 40), None)
    jurisdiction = next((i for i in range(limit) if _JURISDICTION.match(paragraphs[i])), None)
    bench = next((i for i in range(min(len(paragraphs), 15)) if _BENCH.search(paragraphs[i])), None)

    if jurisdiction is not None:
        headnote_end = jurisdiction
    elif delivered is not None:
        headnote_end = delivered
    else:
        return {"header": [], "headnote": [], "body": list(paragraphs)}

    if delivered is not None:
        body_start = delivered + 1
    else:
        body_start = jurisdiction + 1
        while body_start < min(len(paragraphs), jurisdiction + 12) and (
            _words(paragraphs[body_start]) < 40 and _COUNSEL.search(paragraphs[body_start])
        ):
            body_start += 1

    headnote_start = (bench + 1) if bench is not None and bench < headnote_end else 0
    header = list(paragraphs[:headnote_start])
    headnote = [p for p in paragraphs[headnote_start:headnote_end] if _words(p) >= 8]
    return {"header": header, "headnote": headnote, "body": list(paragraphs[body_start:])}


def _split_long(paragraph: str) -> List[str]:
    """Sentence-aligned pieces of at most TARGET_WORDS, one sentence of overlap."""
    sentences: List[str] = []
    for sentence in _SENTENCE.split(paragraph):
        tokens = sentence.split()
        # OCR that lost its punctuation leaves run-on "sentences": window them.
        for start in range(0, len(tokens), TARGET_WORDS):
            sentences.append(" ".join(tokens[start : start + TARGET_WORDS]))
    pieces: List[List[str]] = []
    current: List[str] = []
    count = 0
    for sentence in sentences:
        n = _words(sentence)
        if current and count + n > TARGET_WORDS:
            pieces.append(current)
            overlap = current[-1] if _words(current[-1]) <= TARGET_WORDS // 3 else None
            current, count = ([overlap], _words(overlap)) if overlap else ([], 0)
        current.append(sentence)
        count += n
    if current:
        pieces.append(current)
    return [" ".join(p) for p in pieces]


def _pack(paragraphs: List[str]) -> List[tuple]:
    """(text, first_para, last_para) chunks packed from whole paragraphs."""
    units: List[tuple] = []
    for idx, para in enumerate(paragraphs):
        if _words(para) > MAX_PARA_WORDS:
            units.extend((piece, idx, idx) for piece in _split_long(para))
        else:
            units.append((para, idx, idx))

    chunks: List[tuple] = []
    text, first, last = "", 0, 0
    for unit_text, i, j in units:
        if text and _words(text) + _words(unit_text) > TARGET_WORDS:
            chunks.append((text, first, last))
            text = ""
        if not text:
            first = i
        text = f"{text}\n\n{unit_text}" if text else unit_text
        last = j
    if text:
        chunks.append((text, first, last))

    # A tiny trailing chunk carries no retrievable meaning on its own: fold it in.
    if len(chunks) > 1 and _words(chunks[-1][0]) < MIN_CHUNK_WORDS * 2:
        prev = chunks[-2]
        chunks[-2:] = [(f"{prev[0]}\n\n{chunks[-1][0]}", prev[1], chunks[-1][2])]
    return chunks


def validate_chunk(chunk: Dict) -> None:
    for field in ("chunk_id", "doc_id", "case_title", "text", "role"):
        if not chunk.get(field):
            raise ChunkValidationError(f"missing {field}")
    if chunk["role"] not in ROLES:
        raise ChunkValidationError(f"bad role {chunk['role']!r}")
    if "\f" in chunk["text"] or "\x00" in chunk["text"]:
        raise ChunkValidationError("control characters in text")
    n = _words(chunk["text"])
    if not MIN_CHUNK_WORDS <= n <= MAX_CHUNK_WORDS:
        raise ChunkValidationError(f"{n} words outside [{MIN_CHUNK_WORDS}, {MAX_CHUNK_WORDS}]")


def chunk_judgment(
    doc_id: str,
    year: int,
    clean_text: str,
    meta: Optional[Dict] = None,
) -> List[Dict]:
    """Chunk records for one cleaned judgment (not yet validated)."""
    meta = meta or {}
    paragraphs = [p for p in clean_text.split("\n\n") if p.strip()]
    parts = split_sections(paragraphs)
    base = {
        "doc_id": doc_id,
        "year": year,
        "court": "Supreme Court of India",
        "case_title": (meta.get("title") or "").replace("  ", " ").strip() or doc_id,
        "citation": meta.get("citation") or "",
        "decision_date": meta.get("decision_date") or "",
        "bench": meta.get("judge") or "",
        "disposal": meta.get("disposal_nature") or "",
        "pipeline_version": PIPELINE_VERSION,
    }

    out: List[Dict] = []

    def emit(text: str, role: str, first: int, last: int) -> None:
        out.append(
            {
                **base,
                "chunk_id": f"{doc_id}#{len(out):03d}",
                "role": role,
                "para_range": [first, last],
                "text": text,
                "n_words": _words(text),
                "sections_cited": extract_citations(text),
            }
        )

    for text, first, last in _pack(parts["headnote"]):
        emit(text, "headnote", first, last)
    body = parts["body"]
    packed = _pack(body)
    for i, (text, first, last) in enumerate(packed):
        is_order = i >= len(packed) - 2 and bool(_ORDER.search(text))
        emit(text, "order" if is_order else "body", first, last)
    return out
