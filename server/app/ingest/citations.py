"""
Statute-citation extraction from judgment text.

Finds references like "Ss. 302/34 IPC", "section 438(1) Cr.P.C.", "sections 148,
149 and 302 read with section 34 of the Indian Penal Code" and returns
normalised "ACT:number" strings ("IPC:302", "CrPC:438", "contract act 1872:10",
"Constitution:21"). A reference whose Act can't be determined from the text
right after the number list is skipped rather than guessed.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_NUM = r"\d{1,4}(?:-?[A-Z]{1,2}\b)?(?:\(\w{1,3}\))*"
_SEP = (
    r"\s*(?:,|/|&|\band\b|\bor\b|\bto\b|\bread\s+with\b|\br/w\b)\s*"
    r"(?:(?:sections?|secs?\.?|ss?\.)\s*)?"
)
_LIST = rf"{_NUM}(?:{_SEP}{_NUM})*"
_TRIGGER = r"\b(?:sections?|secs?\.?|ss?\.|u/s\.?)\s*"
_SECTION_RE = re.compile(rf"{_TRIGGER}({_LIST})", re.IGNORECASE)
_ARTICLE_RE = re.compile(rf"\barticles?\s+({_LIST})", re.IGNORECASE)
_NUM_RE = re.compile(r"\d{1,4}(?:-?[A-Z]{1,2}\b)?", re.IGNORECASE)

# Order matters: BNSS before BNS, and the criminal codes before the generic
# "... Act" fallback.
_ACTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:B\.?\s?N\.?\s?S\.?\s?S\.?\b|Bharatiya\s+Nagarik\s+Suraksha)", re.I), "BNSS"),
    (re.compile(r"(?:B\.?\s?N\.?\s?S\.?\b|Bharatiya\s+Nyaya\s+Sanhita)", re.I), "BNS"),
    (re.compile(r"(?:I\.?\s?P\.?\s?C\.?\b|Indian\s+Penal\s+Code|Penal\s+Code)", re.I), "IPC"),
    (
        re.compile(
            r"(?:Cr\.?\s?P\.?\s?C\.?\b|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code)",
            re.I,
        ),
        "CrPC",
    ),
    (re.compile(r"(?:Indian\s+)?Evidence\s+Act", re.I), "IEA"),
]
_LEAD = r"^[\s,\-:]*(?:of\s+)?(?:the\s+)?"
_TAIL_WINDOW = 90
_GENERIC_ACT = re.compile(
    _LEAD + r"((?:[A-Z][\w&\-]*\s+){1,7}Act(?:,?\s*(?:18|19|20)\d\d)?)"
)
_CONSTITUTION = re.compile(_LEAD + r"Constitution\b", re.I)


def _numbers(list_text: str) -> List[str]:
    return [
        m.group(0).upper().replace("-", "")
        for m in _NUM_RE.finditer(re.sub(r"\(\w{1,3}\)", "", list_text))
    ]


def _resolve_act(tail: str) -> str | None:
    for pattern, code in _ACTS:
        if re.match(_LEAD + pattern.pattern, tail, re.I):
            return code
    generic = _GENERIC_ACT.match(tail)
    if generic:
        name = re.sub(r"[,\s]+", " ", generic.group(1)).strip().lower()
        return None if name in ("the act", "this act", "said act") else name
    return None


def extract_citations(text: str) -> List[str]:
    out: List[str] = []

    def add(item: str) -> None:
        if item not in out:
            out.append(item)

    for m in _SECTION_RE.finditer(text):
        act = _resolve_act(text[m.end() : m.end() + _TAIL_WINDOW])
        if act is None:
            continue
        for number in _numbers(m.group(1)):
            add(f"{act}:{number}")

    for m in _ARTICLE_RE.finditer(text):
        if _CONSTITUTION.match(text[m.end() : m.end() + _TAIL_WINDOW]):
            for number in _numbers(m.group(1)):
                add(f"Constitution:{number}")
    return out
