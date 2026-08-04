"""
Legal-entity masking for translation.

IndicTrans2 will happily "translate" a section number, an Act name or a case
citation into gibberish or a transliterated form. Legal references must survive
translation *verbatim* (Requirement 3), so before translating we replace every
protected span with an opaque sentinel token the model leaves untouched, then
restore the originals afterward.

The sentinel format (``__LX7Q_{n}__``) is deliberately alphanumeric-with-
underscores: IndicTrans2's tokenizer keeps it intact and never reorders or
translates it. We restore by the same tokens after generation.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Distinctive, translation-stable sentinel. Underscores + uppercase survive
# IndicTrans2 tokenization; the random-ish infix avoids clashing with corpus
# text. {n} is the running index into the restore mapping.
_SENTINEL = "__LX7Q_{n}__"
_SENTINEL_RE = re.compile(r"__LX7Q_(\d+)__")

# Order matters: earlier patterns are masked first, so longer/more-specific
# spans (full citations) win over shorter ones (a bare section number) that
# would otherwise carve them up.
_PROTECTED_PATTERNS: list[re.Pattern[str]] = [
    # Reported citations: (2019) 3 SCC 12 ; AIR 1973 SC 1461 ; 2018 SCC OnLine
    re.compile(r"\bAIR\s+\d{4}\s+[A-Z]{2,}\s+\d+\b"),
    re.compile(r"\(\d{4}\)\s+\d+\s+[A-Z]{2,}(?:\s+OnLine)?\s+\d+\b"),
    re.compile(r"\b\d{4}\s+[A-Z]{2,}\s+OnLine\s+[A-Z]{2,}\s+\d+\b"),
    # Case names: "Kesavananda Bharati v. State of Kerala" (v. / vs / versus)
    re.compile(
        r"\b[A-Z][A-Za-z.&'-]+(?:\s+[A-Z][A-Za-z.&'-]+){0,5}\s+"
        r"(?:v\.?|vs\.?|versus)\s+"
        r"[A-Z][A-Za-z.&'-]+(?:\s+[A-Za-z.&'-]+){0,6}"
    ),
    # Statute names with a year: "Indian Penal Code, 1860", "... Act, 2023"
    re.compile(
        r"\b(?:[A-Z][A-Za-z]+\s+){1,6}(?:Act|Code|Sanhita|Adhiniyam),?\s+\d{4}\b"
    ),
    # Section / Article / Rule / Order references, incl. §65B, s. 138, Art. 21
    re.compile(
        r"\b(?:Sections?|Secs?\.?|Articles?|Arts?\.?|Rules?|Orders?|Clauses?|"
        r"Sub-?sections?)\s+\d+[A-Za-z]*(?:\(\w+\))*\b",
        re.IGNORECASE,
    ),
    re.compile(r"§\s?\d+[A-Za-z]*(?:\(\w+\))*"),
    re.compile(r"\bs\.\s?\d+[A-Za-z]*\b"),
    # Common statute acronyms that must never be transliterated.
    re.compile(r"\b(?:IPC|CrPC|CPC|BNS|BNSS|BSA|POCSO|NDPS|SEBI|RBI|GST|PMLA)\b"),
    # ISO-ish and DD-MM-YYYY / DD Month YYYY dates.
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
    ),
]


def mask(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected legal spans with sentinels.

    Returns the masked text and a ``{sentinel: original}`` mapping to pass to
    :func:`unmask`. Idempotent-safe: overlapping matches are skipped because we
    substitute left-to-right and never re-scan already-masked regions.
    """
    mapping: dict[str, str] = {}
    counter = 0

    for pattern in _PROTECTED_PATTERNS:
        # Re-run against the progressively-masked text so we never mask inside
        # a sentinel we already inserted.
        def _replace(m: re.Match[str]) -> str:
            nonlocal counter
            span = m.group(0)
            # Don't re-wrap an existing sentinel.
            if _SENTINEL_RE.fullmatch(span.strip()):
                return span
            token = _SENTINEL.format(n=counter)
            mapping[token] = span
            counter += 1
            return token

        text = pattern.sub(_replace, text)

    return text, mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore original spans, replacing any sentinel left in ``text``.

    Tolerant of sentinels the translator may have spaced out (``__LX7Q_ 3__``)
    or whose surrounding whitespace shifted; unresolved sentinels are stripped
    rather than shown to the user.
    """
    if not mapping:
        return text

    for token, original in mapping.items():
        text = text.replace(token, original)

    # Belt-and-suspenders: collapse any mangled/whitespaced sentinel the loop
    # above missed, then drop leftovers so raw tokens never reach the user.
    def _resolve(m: re.Match[str]) -> str:
        return mapping.get(_SENTINEL.format(n=m.group(1)), "")

    leftover = re.sub(r"__LX7Q_\s*(\d+)\s*__", _resolve, text)
    if leftover != text:
        logger.debug("entity_guard: recovered spaced/mangled sentinels")
    return leftover
