"""Legal-entity masking for translation."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


_DIGIT_LETTERS = "ABCDEFGHIJ"
_SENTINEL_PREFIX = "QZXQ"
_SENTINEL_SUFFIX = "QXZQ"
_SENTINEL_RE = re.compile(rf"{_SENTINEL_PREFIX}([A-J]+){_SENTINEL_SUFFIX}")
# What IndicTrans2 sometimes leaves of a sentinel it mangled (e.g. QZXQAQXZQ -> QZXQXZQ).
_REMNANT_RE = re.compile(r"(?<![A-Z])[QZXA-J]*(?:ZX|XZ)[QZXA-J]*(?![A-Z])")


def _make_sentinel(n: int) -> str:
    code = "".join(_DIGIT_LETTERS[int(d)] for d in str(n))
    return f"{_SENTINEL_PREFIX}{code}{_SENTINEL_SUFFIX}"


_PROTECTED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bAIR\s+\d{4}\s+[A-Z]{2,}\s+\d+\b"),
    re.compile(r"\(\d{4}\)\s+\d+\s+[A-Z]{2,}(?:\s+OnLine)?\s+\d+\b"),
    re.compile(r"\b\d{4}\s+[A-Z]{2,}\s+OnLine\s+[A-Z]{2,}\s+\d+\b"),
    re.compile(
        r"\b[A-Z][A-Za-z.&'-]+(?:\s+[A-Z][A-Za-z.&'-]+){0,5}\s+"
        r"(?:v\.?|vs\.?|versus)\s+"
        r"[A-Z][A-Za-z.&'-]+(?:\s+[A-Za-z.&'-]+){0,6}"
    ),
    re.compile(
        r"\b(?:[A-Z][A-Za-z]+\s+){1,6}(?:Act|Code|Sanhita|Adhiniyam),?\s+\d{4}\b"
    ),
    re.compile(
        r"\b(?:Sections?|Secs?\.?|Articles?|Arts?\.?|Rules?|Orders?|Clauses?|"
        r"Sub-?sections?)\s+\d+[A-Za-z]*(?:\(\w+\))*\b",
        re.IGNORECASE,
    ),
    re.compile(r"§\s?\d+[A-Za-z]*(?:\(\w+\))*"),
    re.compile(r"\bs\.\s?\d+[A-Za-z]*\b"),
    re.compile(r"\b(?:IPC|CrPC|CPC|BNS|BNSS|BSA|POCSO|NDPS|SEBI|RBI|GST|PMLA)\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
    ),
]


def mask(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected legal spans with sentinels."""
    mapping: dict[str, str] = {}
    counter = 0

    for pattern in _PROTECTED_PATTERNS:

        def _replace(m: re.Match[str]) -> str:
            nonlocal counter
            span = m.group(0)

            if _SENTINEL_RE.fullmatch(span.strip()):
                return span
            token = _make_sentinel(counter)
            mapping[token] = span
            counter += 1
            return token

        text = pattern.sub(_replace, text)

    # Patterns run one after another, so renumber by position: unmask() can then match
    # mangled sentinels to the spans they replaced, in reading order.
    order: dict[str, str] = {}

    def _renumber(m: re.Match[str]) -> str:
        token = m.group(0)
        if token not in mapping:
            return token
        order.setdefault(token, _make_sentinel(len(order)))
        return order[token]

    text = _SENTINEL_RE.sub(_renumber, text)
    return text, {order[t]: mapping[t] for t in order}


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore original spans, replacing any sentinel left in ``text``."""
    if not mapping:
        return text

    missing = [original for token, original in mapping.items() if token not in text]
    for token, original in mapping.items():
        text = text.replace(token, original)

    def _resolve(m: re.Match[str]) -> str:
        return mapping.get(m.group(0), "")

    leftover = _SENTINEL_RE.sub(_resolve, text)
    if leftover != text:
        logger.debug("entity_guard: recovered leftover sentinels")

    remnants = _REMNANT_RE.findall(leftover)
    if remnants and len(remnants) == len(missing):
        restore = iter(missing)
        leftover = _REMNANT_RE.sub(lambda _: next(restore), leftover)
        logger.info("entity_guard: restored %d mangled sentinel(s)", len(remnants))
    elif remnants:
        leftover = re.sub(rf"\s*{_REMNANT_RE.pattern}", "", leftover)
        logger.info("entity_guard: dropped %d unmatched sentinel remnant(s)", len(remnants))
    return leftover


def is_complete(text: str, mapping: dict[str, str]) -> bool:
    """True if every protected span made it into ``text``."""
    return all(original in text for original in mapping.values())
