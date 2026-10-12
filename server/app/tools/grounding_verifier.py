"""grounding_verifier.py — sentence-level semantic grounding gate."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from app.tools.citation_verifier import (
    VerificationReport,
    iter_citation_occurrences,
    verify_citations,
)

SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNGROUNDED = "UNGROUNDED"

_STATUS_WEIGHT = {
    SUPPORTED: 1.0,
    PARTIALLY_SUPPORTED: 0.5,
    UNGROUNDED: 0.0,
    CONTRADICTED: 0.0,
}


CONTRADICTION_TRIGGERS = (
    "except",
    "unless",
    "subject to",
    "provided that",
    "shall not",
    "only if",
)


HIGH_RISK_ABSOLUTES = (
    "must",
    "cannot",
    "always",
    "never",
    "abolished",
    "removed",
    "unconstitutional",
    "invalid",
    "no longer",
)


_SUPPORTED_THRESHOLD = 0.25
_PARTIAL_THRESHOLD = 0.12
_HIGH_RISK_SUPPORTED_THRESHOLD = 0.45

_VERBATIM_FRACTION = 0.6
_CITED_REVIEW_BELOW = 0.6
_MAX_LLM_CORRECTIONS = 8
_MAX_EVIDENCE_CHARS = 800


@dataclass
class SentenceGrounding:
    text: str
    start: int
    end: int
    citations: List[str] = field(default_factory=list)
    is_claim: bool = False
    is_high_risk: bool = False
    status: str = SUPPORTED
    overlap: float = 1.0
    reason: str = ""
    evidence: str = ""
    needs_llm: bool = False
    outcome: str = "unchanged"

    det_status: str = SUPPORTED


@dataclass
class GroundingReport:
    sentences: List[SentenceGrounding]
    citation_report: VerificationReport

    llm_succeeded: bool = False

    @property
    def claim_sentences(self) -> List[SentenceGrounding]:
        return [s for s in self.sentences if s.is_claim]

    @property
    def flagged(self) -> List[SentenceGrounding]:
        return [s for s in self.claim_sentences if s.status != SUPPORTED]

    @property
    def confirmed_flagged(self) -> List[SentenceGrounding]:
        """Flagged claims the deterministic evidence also condemned (fabricated quantity, unresolvable citation, invented condition, near-zero overlap)."""
        return [s for s in self.flagged if s.det_status in (CONTRADICTED, UNGROUNDED)]

    @property
    def overall_score(self) -> float:
        claims = self.claim_sentences
        if not claims:
            return 1.0
        return round(sum(_STATUS_WEIGHT[s.status] for s in claims) / len(claims), 3)


_LEAD_RE = re.compile(r"^(\s*(?:[-*•]\s+|\d+\.\s+|#{1,6}\s+)?)")
_TRAIL_WS_RE = re.compile(r"(\s*)$")
_WORD_RE = re.compile(r"\b[a-z]{4,}\b")


_ABBREVIATIONS = frozenset(
    {
        "v",
        "vs",
        "no",
        "nos",
        "sec",
        "art",
        "dr",
        "mr",
        "mrs",
        "ms",
        "cf",
        "viz",
        "eg",
        "ie",
        "s",
        "ss",
    }
)


def _period_is_boundary(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and (text[j].isalnum()):
        j -= 1
    token = text[j + 1 : i].lower()
    return not (token in _ABBREVIATIONS or len(token) == 1)


def _split_sentences(text: str) -> List[Tuple[int, int]]:
    """Partition `text` into contiguous (start, end) spans covering every character exactly once: splits on '.'/'!'/'?' followed by whitespace or end-of-string, and on newlines (so..."""
    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            spans.append((start, i + 1))
            i += 1
            start = i
            continue
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            if (j >= n or text[j].isspace()) and (
                ch != "." or j - i > 1 or _period_is_boundary(text, i)
            ):
                spans.append((start, j))
                i = j
                start = j
                continue
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans


def _assign_citations_to_sentences(spans, occurrences):
    assigned: Dict[int, list] = {}
    occ_idx = 0
    for i, (_, end) in enumerate(spans):
        while occ_idx < len(occurrences) and occurrences[occ_idx].start < end:
            assigned.setdefault(i, []).append(occurrences[occ_idx])
            occ_idx += 1
    return assigned


_STEM_LEN = 6


def _content_words(text: str) -> set:
    return {w[:_STEM_LEN] for w in _WORD_RE.findall(text.lower())}


def _word_overlap(claim: str, evidence: str) -> float:
    """What fraction of the claim's content vocabulary appears in the evidence text."""
    claim_words = _content_words(claim)
    if not claim_words:
        return 1.0
    return len(claim_words & _content_words(evidence)) / len(claim_words)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _trigrams(text: str) -> set:
    words = _TOKEN_RE.findall(text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def _verbatim_fraction(claim: str, context_text: str) -> float:
    """Fraction of the claim's word-trigrams that occur in the context."""
    claim_tris = _trigrams(claim)
    if not claim_tris:
        return 0.0
    return len(claim_tris & _trigrams(context_text)) / len(claim_tris)


def _trigger_set(text: str) -> set:
    t = text.lower()
    return {
        trig
        for trig in CONTRADICTION_TRIGGERS
        if re.search(r"\b" + re.escape(trig) + r"\b", t)
    }


_EVIDENCE_QUALIFIER_MARKERS = CONTRADICTION_TRIGGERS + (
    "nothing herein",
    "nothing contained",
    "save as",
    "saving clause",
    "without prejudice",
    "other than",
    "excepting",
    "in the absence of",
)


def _has_qualifier_language(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _EVIDENCE_QUALIFIER_MARKERS)


def _has_high_risk(text: str) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in HIGH_RISK_ABSOLUTES)


_ABSOLUTE_MARKERS = (
    "always",
    "never",
    "absolute",
    "absolutely",
    "under any circumstances",
    "in all cases",
    "without exception",
    "unconditional",
    "unconditionally",
)


_NEGATION_LOOKBACK = re.compile(
    r"\b(?:no|not|never|isn'?t|aren'?t|doesn'?t|don'?t|without)\b(?:\s+\w+){0,2}\s*$"
)


def _has_absolute(text: str) -> bool:
    t = text.lower()
    for m in _ABSOLUTE_MARKERS:
        for match in re.finditer(r"\b" + re.escape(m) + r"\b", t):
            if not _NEGATION_LOOKBACK.search(t[: match.start()]):
                return True
    return False


def _contradiction_reason(claim: str, evidence: str) -> Optional[str]:
    """Deterministic polarity check against ONE provision: did the claim invent a condition it doesn't state, or assert an exceptionless rule for a provision that is qualified?"""
    if not evidence.strip():
        return None
    claim_trig = _trigger_set(claim)
    evidence_trig = _trigger_set(evidence)

    invented = (
        claim_trig if claim_trig and not _has_qualifier_language(evidence) else set()
    )
    dropped = evidence_trig - claim_trig
    if invented:
        return (
            f"claim adds {', '.join(sorted(invented))!r}, a condition not present "
            f"in the retrieved provision"
        )
    if dropped and _has_absolute(claim):
        return (
            f"claim states an exceptionless rule but the retrieved provision qualifies it "
            f"with {', '.join(sorted(dropped))!r}"
        )
    return None


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}
_UNITS = r"(?:years?|months?|weeks?|days?|hours?|rupees?|lakhs?|crores?|per\s?cent|percent|%)"
_QUANTITY_RE = re.compile(
    r"(?:(?:rs\.?|₹)\s?(?P<cur>\d[\d,]*)|(?P<num>\d[\d,]*|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")[\s-]+(?P<unit>"
    + _UNITS
    + r"))",
    re.IGNORECASE,
)


def _quantities(text: str) -> set:
    """{(value, unit)} for punishments/periods/amounts, number words folded to digits so "seven years" == "7 years"."""
    found = set()
    for m in _QUANTITY_RE.finditer(text):
        raw = (m.group("cur") or m.group("num")).lower().replace(",", "")
        value = _NUMBER_WORDS.get(raw, raw)
        unit = (
            "rs"
            if m.group("cur")
            else re.sub(r"s$", "", m.group("unit").lower().replace(" ", ""))
        )
        unit = {"percent": "%", "rupee": "rs", "lakh": "lakh", "crore": "crore"}.get(
            unit, unit
        )
        found.add((str(value), unit))
    return found


def _unsupported_quantity(claim: str, passages: List[str]) -> Optional[str]:
    """A quantity the claim states that no candidate passage contains."""
    claimed = _quantities(claim)
    if not claimed:
        return None
    available = set()
    for p in passages:
        available |= _quantities(p)
    missing = sorted(f"{v} {u}" for v, u in claimed - available)
    if missing:
        return (
            f"claim states {', '.join(missing)}, which the retrieved provisions do not"
        )
    return None


def _classify_overlap(overlap: float, is_high_risk: bool) -> str:
    supported_cut = (
        _HIGH_RISK_SUPPORTED_THRESHOLD if is_high_risk else _SUPPORTED_THRESHOLD
    )
    if overlap >= supported_cut:
        return SUPPORTED
    if overlap >= _PARTIAL_THRESHOLD:
        return PARTIALLY_SUPPORTED
    return UNGROUNDED


_HEADING_RE = re.compile(r"^\s*#{1,6}\s")


_HEDGE_RE = re.compile(
    r"\b(?:retrieved (?:context|provisions?|sources?|evidence|materials?)|"
    r"(?:context|provisions?) (?:does not|do not|lacks?|doesn't) |"
    r"(?:does not|do not|doesn't) (?:specify|define|address|mention|cover|provide)|"
    r"not (?:specified|defined|covered|addressed) in|"
    r"\bno (?:provisions?|case law|explanations?|sections?|articles?|references?)\b"
    r".{0,60}?\b(?:mention|address|cover|specify|define|state)s?\b)",
    re.IGNORECASE,
)


def _is_non_claim(stripped: str) -> bool:
    return (
        bool(_HEADING_RE.match(stripped))
        or "|" in stripped
        or stripped.endswith(":")
        or bool(_HEDGE_RE.search(stripped))
    )


_ENTRY_SPLIT_RE = re.compile(r"\n+(?=• \*\*)")
_ENTRY_HEAD_RE = re.compile(r"^• \*\*(?P<head>.+?)\*\*")
_SECTION_HEAD_RE = re.compile(r"^.+?\s+(?:§\s*|Article\s+)(?P<sec>[0-9A-Za-z\-]+)$")


def _context_passages(context_text: str) -> List[Tuple[str, str]]:
    """(section_number_upper or "", text) for each entry of the retrieved-context block the model was actually shown."""
    passages: List[Tuple[str, str]] = []
    for entry in _ENTRY_SPLIT_RE.split(context_text or ""):
        entry = entry.strip()
        head = _ENTRY_HEAD_RE.match(entry)
        if not head:
            continue
        sec = _SECTION_HEAD_RE.match(head.group("head"))
        passages.append((sec.group("sec").upper() if sec else "", entry))
    return passages


_WINDOW = 800


def _best_window(claim: str, passage: str, size: int = _WINDOW) -> str:
    """The stretch of `passage` (≤ size chars) that best matches the claim."""
    if len(passage) <= size:
        return passage
    step = max(size // 3, 1)
    best, best_score = passage[:size], -1.0
    for start in range(0, len(passage) - size + step, step):
        window = passage[start : start + size]
        score = _word_overlap(claim, window)
        if score > best_score:
            best, best_score = window, score
    return best


def assess_grounding(
    answer: str,
    rag,
    retrieved_sections: Optional[set] = None,
    retrieved_context_text: str = "",
) -> GroundingReport:
    """Deterministic pass: sentence-split the answer, resolve evidence for every citation-bearing or high-risk-absolute sentence, and classify each."""
    citation_report = verify_citations(answer, rag, retrieved_sections)
    occurrences = iter_citation_occurrences(answer)
    spans = _split_sentences(answer)
    by_sentence = _assign_citations_to_sentences(spans, occurrences)
    context_passages = _context_passages(retrieved_context_text)
    context_texts = [t for _, t in context_passages]
    context_blob = "\n".join(context_texts)

    case_law_texts = [t for sec, t in context_passages if not sec]

    sentences: List[SentenceGrounding] = []
    for i, (start, end) in enumerate(spans):
        raw_span = answer[start:end]
        stripped = raw_span.strip()
        occs = by_sentence.get(i, [])
        high_risk = _has_high_risk(stripped) if stripped else False

        if not stripped or _is_non_claim(stripped) or (not occs and not high_risk):
            sentences.append(
                SentenceGrounding(text=raw_span, start=start, end=end, is_claim=False)
            )
            continue

        cited: List[str] = []
        for occ in occs:
            hits = (
                rag.find_section(occ.act_hint, occ.section, max_parts=2)
                if occ.act_hint
                else []
            )
            if hits:
                cited.extend(h.text for h in hits)
            else:
                cited.extend(
                    t for sec, t in context_passages if sec == occ.section.upper()
                )
        citations = [occ.raw for occ in occs]

        candidates = list(
            dict.fromkeys((cited + case_law_texts) if occs else context_texts)
        )
        if len(cited) > 1:

            candidates = ["\n".join(cited)] + candidates

        if not candidates:
            status, overlap, evidence = UNGROUNDED, 0.0, ""
            reason = (
                "the cited provision is not among the retrieved provisions"
                if occs
                else "no retrieved text is available to verify this claim"
            )
        else:
            scored = [(_word_overlap(stripped, c), c) for c in candidates]
            overlap, best = max(scored, key=lambda pair: pair[0])
            evidence = _best_window(stripped, best)
            verbatim_frac = _verbatim_fraction(stripped, context_blob)

            if verbatim_frac >= _VERBATIM_FRACTION:
                overlap = max(overlap, 1.0)
            status = _classify_overlap(overlap, high_risk)
            reason = f"~{overlap:.0%} term overlap with the retrieved evidence"

            contradiction = (
                _contradiction_reason(stripped, best)
                if overlap >= _PARTIAL_THRESHOLD
                else None
            ) or _unsupported_quantity(stripped, candidates)
            if contradiction:
                status, reason = CONTRADICTED, contradiction
            elif occs and not cited and status != SUPPORTED:

                reason = "the cited provision is not among the retrieved provisions"

        needs_llm = bool(candidates) and (
            (high_risk and status == PARTIALLY_SUPPORTED)
            or (bool(occs) and overlap < _CITED_REVIEW_BELOW)
        )

        sentences.append(
            SentenceGrounding(
                text=raw_span,
                start=start,
                end=end,
                citations=citations,
                is_claim=True,
                is_high_risk=high_risk,
                status=status,
                det_status=status,
                overlap=overlap,
                reason=reason,
                evidence=evidence,
                needs_llm=needs_llm,
            )
        )

    return GroundingReport(sentences=sentences, citation_report=citation_report)


_CORRECTION_PROMPT = """You are fact-checking claims from an Indian legal answer against the exact \
statutory text that was retrieved for this question. For EACH numbered claim, compare it to its \
evidence and decide a status:
- SUPPORTED: the evidence confirms the claim as stated
- PARTIALLY_SUPPORTED: the evidence confirms part of the claim, or the claim omits a condition/exception present in the evidence
- CONTRADICTED: the evidence states the opposite, or the claim reverses a condition/exception (e.g. drops "unless", "except", "subject to", "shall not")
- UNGROUNDED: the evidence does not address the claim at all

Then fill "corrected":
- For a claim marked REWRITE: no, always set "corrected" to an empty string "".
- For a claim marked REWRITE: yes: if SUPPORTED, repeat the claim unchanged; otherwise rewrite it as a \
single sentence using ONLY facts present in its evidence. If the evidence supports nothing useful, write \
a short sentence stating the retrieved sources do not confirm this and recommend consulting a lawyer. \
Never invent new facts, numbers, or section references that are not in the evidence.

CLAIMS:
{claims_block}

Respond with ONLY a JSON array, no other text, in exactly this shape:
[
  {{"index": 1, "status": "SUPPORTED|PARTIALLY_SUPPORTED|CONTRADICTED|UNGROUNDED", "corrected": "<sentence text>"}}
]
"""


CORRECTION_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": [SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNGROUNDED],
            },
            "corrected": {"type": "string"},
        },
        "required": ["index", "status", "corrected"],
    },
}


def _may_rewrite(s: SentenceGrounding) -> bool:
    """Only text the deterministic evidence already condemned is ever rewritten; for the rest the LLM's verdict is all that's used, so asking it for a sentence just costs tokens."""
    return s.det_status in (CONTRADICTED, UNGROUNDED)


def _build_claims_block(sentences: List[SentenceGrounding]) -> str:
    lines = []
    for i, s in enumerate(sentences, 1):
        evidence = (s.evidence.strip() or "(no retrieved evidence available)")[
            :_MAX_EVIDENCE_CHARS
        ]
        rewrite = "yes" if _may_rewrite(s) else "no"
        lines.append(
            f'{i}. CLAIM: "{s.text.strip()}"\n   EVIDENCE: "{evidence}"\n   REWRITE: {rewrite}'
        )
    return "\n\n".join(lines)


def _extract_json_array(raw: str) -> list:
    """Find the first balanced top-level `[...]` in `raw` and parse it."""
    raw = raw.strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : i + 1])
    raise ValueError("No balanced JSON array found in LLM output")


async def _llm_adjudicate_and_correct(
    sentences: List[SentenceGrounding],
    llm_invoke: Callable[[str], Awaitable[str]],
) -> Dict[int, Tuple[str, str]]:
    from app.chatbot import strip_reasoning_tags

    prompt = _CORRECTION_PROMPT.format(claims_block=_build_claims_block(sentences))
    raw = await llm_invoke(prompt)
    parsed = _extract_json_array(strip_reasoning_tags(raw))

    valid_statuses = {SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNGROUNDED}
    result: Dict[int, Tuple[str, str]] = {}
    for item in parsed:
        try:
            idx = int(item.get("index", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(sentences)):
            continue
        status = str(item.get("status", "")).upper().strip()
        if status not in valid_statuses:
            status = UNGROUNDED
        result[idx] = (status, str(item.get("corrected", "")).strip())
    return result


def _reassemble(original_span: str, corrected_sentence: str) -> str:
    """Splice a rewritten claim back into its original span, preserving leading markdown structure (bullet/heading markers) and trailing whitespace/newline so the surrounding document..."""
    lead_match = _LEAD_RE.match(original_span)
    trail_match = _TRAIL_WS_RE.search(original_span)
    lead = lead_match.group(1) if lead_match else ""
    trail = trail_match.group(1) if trail_match else ""
    body = corrected_sentence.strip()
    if not body:
        return original_span
    stripped_original = original_span.strip()
    if not body.endswith((".", "!", "?")) and stripped_original.endswith(
        (".", "!", "?")
    ):
        body += "."
    return f"{lead}{body}{trail}"


async def ground_and_correct(
    answer: str,
    rag,
    retrieved_sections: Optional[set] = None,
    retrieved_context_text: str = "",
    llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
) -> Tuple[str, GroundingReport]:
    """Run the deterministic grounding pass, then — only for sentences that are CONTRADICTED, UNGROUNDED-with-a-citation, or high-risk-and-not- clearly-supported — a single batched..."""
    report = assess_grounding(answer, rag, retrieved_sections, retrieved_context_text)

    to_fix = sorted(
        (
            s
            for s in report.sentences
            if s.is_claim
            and (
                (
                    s.status != SUPPORTED
                    and (s.status in (CONTRADICTED, UNGROUNDED) or s.is_high_risk)
                )
                or s.needs_llm
            )
        ),
        key=lambda s: s.overlap,
    )[:_MAX_LLM_CORRECTIONS]

    if not to_fix or llm_invoke is None:
        return answer, report

    try:
        corrections = await _llm_adjudicate_and_correct(to_fix, llm_invoke)
    except Exception as e:
        print(f"[GroundingGate] LLM correction skipped: {e}")
        return answer, report

    report.llm_succeeded = True
    if not corrections:
        return answer, report

    new_text = answer
    for i, s in sorted(enumerate(to_fix), key=lambda pair: pair[1].start, reverse=True):
        fix = corrections.get(i)
        if fix is None:
            continue
        new_status, corrected_sentence = fix
        original_span = answer[s.start : s.end]
        if new_status == SUPPORTED:

            s.status = SUPPORTED
            continue
        if _may_rewrite(s) and not corrected_sentence:
            s.status = new_status
            continue
        if not _may_rewrite(s):

            s.status = new_status
            s.reason = (
                "the fact-check could not confirm this against the retrieved provisions"
            )
            continue
        replacement = _reassemble(original_span, corrected_sentence)
        new_text = new_text[: s.start] + replacement + new_text[s.end :]
        s.status = new_status
        s.outcome = "corrected"

    return new_text, report


def grounding_footer(report: GroundingReport) -> str:
    """Advisory footer summarizing claim-level (not just citation-level) grounding."""
    claims = report.claim_sentences
    corrected = [s for s in claims if s.outcome == "corrected"]
    confirmed = [s for s in report.confirmed_flagged if s.outcome != "corrected"]
    advisory = [
        s
        for s in claims
        if s.status != SUPPORTED and s.outcome != "corrected" and s not in confirmed
    ]

    if not corrected and not confirmed:
        return ""
    if not report.llm_succeeded or report.overall_score >= 0.5:
        return ""

    lines = [
        "",
        "---",
        f"🧭 **Grounding check** (claim-level, confidence {report.overall_score:.0%}):",
    ]
    for s in corrected:
        lines.append(
            "- A claim was rewritten to match the retrieved text — the original "
            "statement was not adequately supported."
        )
    for s in confirmed:
        cite = f" ({', '.join(s.citations)})" if s.citations else ""
        lines.append(
            f"- {s.status.replace('_', ' ').title()}{cite}: {s.reason}. "
            f'"{s.text.strip()[:140]}"'
        )
    if advisory:
        lines.append(
            f"- {len(advisory)} other statement(s) could not be confirmed against the "
            f"retrieved provisions."
        )
    return "\n".join(lines)
