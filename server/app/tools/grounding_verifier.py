"""
grounding_verifier.py — sentence-level semantic grounding gate.

citation_verifier.py answers "does this citation exist, under the right
Act, among what was retrieved?" — but a citation can pass all three checks
and the sentence built around it can still misstate what the provision
says: drop an exception, reverse a condition, or assert an absolute rule
the text doesn't support. This module checks the *claim*, not just the
citation token.

Pipeline (cheapest checks first, LLM used only as a last resort):
  1. Split the answer into sentence-ish spans (deterministic, regex-free
     manual scan — markdown headers/bullets become their own span).
  2. For each sentence carrying a citation, pull the evidence text for
     that exact provision via `rag.find_section` — no new retrieval,
     no LLM, same corpus citation_verifier already trusts.
     Sentences with no citation but a high-risk absolute claim (see below)
     fall back to whatever retrieved-context text the caller passed in.
  3. Score word overlap between claim and evidence (same heuristic family
     as metrics/generation_metrics._keyword_faithfulness) -> base status.
  4. Scan both claim and evidence for exception/negation trigger words
     (except, unless, subject to, provided that, shall not, only if) and
     diff them -> CONTRADICTED override when the claim invents or drops one.
  5. High-risk absolute language (must/cannot/always/never/abolished/
     unconstitutional/invalid/removed/no longer) demands a higher overlap
     bar; sentences that clear UNGROUNDED but fall short of that bar are
     provisionally PARTIALLY_SUPPORTED pending step 6, instead of being
     trusted on weak word overlap alone.
  6. Sentences that are CONTRADICTED, UNGROUNDED-with-a-citation, or
     high-risk-and-not-clearly-supported go into ONE batched LLM call that
     both finalizes their status and rewrites them from evidence only.
     Everything else never touches the LLM.

With no flagged sentences (the common case) this adds zero LLM calls on
top of the existing citation check — regex + dict lookups only. Worst
case is exactly one extra small-model call per answer, never one per
sentence, and a failed/unavailable LLM degrades to "deterministic report,
advisory footer only" — the same fail-open behavior citation_verifier uses.
"""

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

# Presence/absence of these between a claim and its evidence is checkable
# without real NLI: if the claim states one but the evidence doesn't (or
# vice versa), the claim has reversed or invented a condition.
CONTRADICTION_TRIGGERS = (
    "except",
    "unless",
    "subject to",
    "provided that",
    "shall not",
    "only if",
)

# Absolute language that raises the bar for what counts as "supported" —
# these are the claims where being wrong is worst, so weak word-overlap
# alone shouldn't be trusted to clear them.
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

# Word overlap between a fluent generated sentence and bare statutory text is
# inherently low even when the claim is accurate (legal prose paraphrases).
# These bars were originally calibrated too high (0.45/0.60) and flagged the
# large majority of claims on every answer; lowered after inspecting real
# qwen3 outputs against their cited evidence.
_SUPPORTED_THRESHOLD = 0.25
_PARTIAL_THRESHOLD = 0.12
_HIGH_RISK_SUPPORTED_THRESHOLD = 0.45

_VERBATIM_FRACTION = 0.6  # this share of a claim's word-trigrams in the context = quoted
_CITED_REVIEW_BELOW = 0.6  # cited claims under this overlap are adjudicated, not trusted
_MAX_LLM_CORRECTIONS = 8  # bound prompt size / latency regardless of how much is flagged
_MAX_EVIDENCE_CHARS = 800


@dataclass
class SentenceGrounding:
    text: str
    start: int
    end: int
    citations: List[str] = field(default_factory=list)
    is_claim: bool = False  # False for boilerplate/connective sentences — never scored
    is_high_risk: bool = False
    status: str = SUPPORTED
    overlap: float = 1.0
    reason: str = ""
    evidence: str = ""
    needs_llm: bool = False
    outcome: str = "unchanged"  # unchanged | corrected
    # Status from the deterministic pass alone, before any LLM adjudication. A
    # sentence is only ever rewritten when this already condemned it.
    det_status: str = SUPPORTED


@dataclass
class GroundingReport:
    sentences: List[SentenceGrounding]
    citation_report: VerificationReport
    # True only once the LLM correction/adjudication call has run and its
    # output parsed successfully — distinguishes "verified low confidence"
    # from "the deterministic pass flagged some claims but we couldn't
    # double-check them" (grounding_footer treats the two differently).
    llm_succeeded: bool = False

    @property
    def claim_sentences(self) -> List[SentenceGrounding]:
        return [s for s in self.sentences if s.is_claim]

    @property
    def flagged(self) -> List[SentenceGrounding]:
        return [s for s in self.claim_sentences if s.status != SUPPORTED]

    @property
    def confirmed_flagged(self) -> List[SentenceGrounding]:
        """Flagged claims the deterministic evidence also condemned (fabricated
        quantity, unresolvable citation, invented condition, near-zero overlap).
        A flag only the small LLM raises is advisory: measured on real answers it
        varies run to run and objected to faithful paraphrases, so it may lower
        the score but is never enough to rewrite text, regenerate the answer or
        be itemised for the user."""
        return [s for s in self.flagged if s.det_status in (CONTRADICTED, UNGROUNDED)]

    @property
    def overall_score(self) -> float:
        claims = self.claim_sentences
        if not claims:
            return 1.0
        return round(sum(_STATUS_WEIGHT[s.status] for s in claims) / len(claims), 3)


# ---------------------------------------------------------------------------
# Deterministic layer
# ---------------------------------------------------------------------------

_LEAD_RE = re.compile(r"^(\s*(?:[-*•]\s+|\d+\.\s+|#{1,6}\s+)?)")
_TRAIL_WS_RE = re.compile(r"(\s*)$")
_WORD_RE = re.compile(r"\b[a-z]{4,}\b")


# "B. Heading", "1. Item", "Ghose v. Mugneeram", "No. 5", "Sec. 3" — a period here
# does not end a sentence, and splitting there produced fragments that were then
# graded (and rewritten) as if they were claims.
_ABBREVIATIONS = frozenset(
    {"v", "vs", "no", "nos", "sec", "art", "dr", "mr", "mrs", "ms", "cf", "viz", "eg", "ie", "s", "ss"}
)


def _period_is_boundary(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and (text[j].isalnum()):
        j -= 1
    token = text[j + 1 : i].lower()
    return not (token in _ABBREVIATIONS or len(token) == 1)


def _split_sentences(text: str) -> List[Tuple[int, int]]:
    """
    Partition `text` into contiguous (start, end) spans covering every
    character exactly once: splits on '.'/'!'/'?' followed by whitespace
    or end-of-string, and on newlines (so markdown headers/bullets become
    their own span rather than bleeding into the next line's claim).
    """
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
            if (j >= n or text[j].isspace()) and (ch != "." or j - i > 1 or _period_is_boundary(text, i)):
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


# Suffix-insensitive matching: "established"/"establishes" or "violated"/"violation"
# are the same word for grounding purposes. A fixed-length prefix is crude but
# robust for legal English, and errs toward matching (fewer false flags).
_STEM_LEN = 6


def _content_words(text: str) -> set:
    return {w[:_STEM_LEN] for w in _WORD_RE.findall(text.lower())}


def _word_overlap(claim: str, evidence: str) -> float:
    """What fraction of the claim's content vocabulary appears in the
    evidence text. Same family of heuristic as
    metrics/generation_metrics._keyword_faithfulness — rough by design,
    it only needs to separate "clearly grounded" from "clearly not"."""
    claim_words = _content_words(claim)
    if not claim_words:
        return 1.0
    return len(claim_words & _content_words(evidence)) / len(claim_words)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _trigrams(text: str) -> set:
    words = _TOKEN_RE.findall(text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def _verbatim_fraction(claim: str, context_text: str) -> float:
    """Fraction of the claim's word-trigrams that occur in the context. Statute
    quoted or closely followed scores near 1.0 even when it spans several
    passages, which bag-of-words overlap cannot see; a fluent paraphrase or a
    fabricated rule scores near 0."""
    claim_tris = _trigrams(claim)
    if not claim_tris:
        return 0.0
    return len(claim_tris & _trigrams(context_text)) / len(claim_tris)


def _trigger_set(text: str) -> set:
    t = text.lower()
    return {trig for trig in CONTRADICTION_TRIGGERS if re.search(r"\b" + re.escape(trig) + r"\b", t)}


# Real statutory drafting expresses the same exception many ways ("Nothing
# herein contained shall affect any law... requiring a contract to be in
# writing" is the same idea as "unless required in writing", worded
# differently). CONTRADICTION_TRIGGERS alone is a literal-word match; used to
# decide whether a claim "invented" a qualifier, it flagged a claim that
# faithfully paraphrased a real one just because the provision phrased it
# differently. This wider list is evidence-side only, and answers a coarser,
# safer question: does the SOURCE qualify itself at all (regardless of exact
# wording), not whether it uses the claim's specific word.
_EVIDENCE_QUALIFIER_MARKERS = CONTRADICTION_TRIGGERS + (
    "nothing herein", "nothing contained", "save as", "saving clause",
    "without prejudice", "other than", "excepting", "in the absence of",
)


def _has_qualifier_language(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _EVIDENCE_QUALIFIER_MARKERS)


def _has_high_risk(text: str) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in HIGH_RISK_ABSOLUTES)


# Words that assert a rule with no exceptions. Only these make *dropping* a
# qualifier from the provision a contradiction ("X can never be restricted" vs
# "X … except according to procedure established by law"). Ordinary obligation
# words (must, cannot) are common in faithful restatements — "no person shall be
# deprived … except according to procedure" is fairly restated as "cannot be
# deprived without procedure" — so they are not enough.
_ABSOLUTE_MARKERS = (
    "always", "never", "absolute", "absolutely", "under any circumstances",
    "in all cases", "without exception", "unconditional", "unconditionally",
)


# A negation within a few words before the marker flips its meaning: "No
# absolute right exists" ASSERTS a qualified right (agrees with a qualified
# provision) — the opposite of "This right is absolute" (asserts an
# exceptionless one). Found live: "**No absolute right exists** under Indian
# law because... subject to 'procedure established by law'" was flagged
# CONTRADICTED against Article 21's real "except..." qualifier, because the
# claim was correctly AGREEING with it.
_NEGATION_LOOKBACK = re.compile(
    r"\b(?:no|not|never|isn'?t|aren'?t|doesn'?t|don'?t|without)\b(?:\s+\w+){0,2}\s*$"
)


def _has_absolute(text: str) -> bool:
    t = text.lower()
    for m in _ABSOLUTE_MARKERS:
        for match in re.finditer(r"\b" + re.escape(m) + r"\b", t):
            if not _NEGATION_LOOKBACK.search(t[:match.start()]):
                return True
    return False


def _contradiction_reason(claim: str, evidence: str) -> Optional[str]:
    """Deterministic polarity check against ONE provision: did the claim invent
    a condition it doesn't state, or assert an exceptionless rule for a provision
    that is qualified? Word overlap can't catch either — same vocabulary,
    opposite meaning. Callers must pass the specific provision the claim is
    about, never the whole retrieved context: that always contains some
    "except"/"unless" somewhere, which makes the dropped-qualifier test fire on
    everything."""
    if not evidence.strip():
        return None
    claim_trig = _trigger_set(claim)
    evidence_trig = _trigger_set(evidence)
    # A claim's exception-word only counts as "invented" when the source shows
    # NO qualifying language of any kind — not merely a different word for the
    # same qualifier the source actually has.
    invented = claim_trig if claim_trig and not _has_qualifier_language(evidence) else set()
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


# Quantities that carry legal weight: a punishment term, a deadline, a fine.
# Word overlap ignores digits entirely, so "imprisonment up to two years" passed
# as SUPPORTED against a provision that says seven.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}
_UNITS = r"(?:years?|months?|weeks?|days?|hours?|rupees?|lakhs?|crores?|per\s?cent|percent|%)"
_QUANTITY_RE = re.compile(
    r"(?:(?:rs\.?|₹)\s?(?P<cur>\d[\d,]*)|(?P<num>\d[\d,]*|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")[\s-]+(?P<unit>" + _UNITS + r"))",
    re.IGNORECASE,
)


def _quantities(text: str) -> set:
    """{(value, unit)} for punishments/periods/amounts, number words folded to
    digits so "seven years" == "7 years"."""
    found = set()
    for m in _QUANTITY_RE.finditer(text):
        raw = (m.group("cur") or m.group("num")).lower().replace(",", "")
        value = _NUMBER_WORDS.get(raw, raw)
        unit = "rs" if m.group("cur") else re.sub(r"s$", "", m.group("unit").lower().replace(" ", ""))
        unit = {"percent": "%", "rupee": "rs", "lakh": "lakh", "crore": "crore"}.get(unit, unit)
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
        return f"claim states {', '.join(missing)}, which the retrieved provisions do not"
    return None


def _classify_overlap(overlap: float, is_high_risk: bool) -> str:
    supported_cut = _HIGH_RISK_SUPPORTED_THRESHOLD if is_high_risk else _SUPPORTED_THRESHOLD
    if overlap >= supported_cut:
        return SUPPORTED
    if overlap >= _PARTIAL_THRESHOLD:
        return PARTIALLY_SUPPORTED
    return UNGROUNDED


# --- what counts as a claim --------------------------------------------------

_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
# Statements ABOUT the retrieved context ("the context does not specify …") are
# the hedges GROUNDED_QUERY_PROMPT tells the model to write. They assert nothing
# to verify, and rewriting them would remove the disclosure.
_HEDGE_RE = re.compile(
    r"\b(?:retrieved (?:context|provisions?|sources?|evidence|materials?)|"
    r"(?:context|provisions?) (?:does not|do not|lacks?|doesn't) |"
    r"(?:does not|do not|doesn't) (?:specify|define|address|mention|cover|provide)|"
    r"not (?:specified|defined|covered|addressed) in|"
    # "no provisions ... mention/address/cover X" — the same disclosure the
    # prompt asks for ("I don't have specific references for this aspect"),
    # phrased as a negated noun instead of "the context does not mention".
    r"\bno (?:provisions?|case law|explanations?|sections?|articles?|references?)\b"
    r".{0,60}?\b(?:mention|address|cover|specify|define|state)s?\b)",
    re.IGNORECASE,
)


def _is_non_claim(stripped: str) -> bool:
    return (
        bool(_HEADING_RE.match(stripped))
        or "|" in stripped
        or stripped.endswith(":")  # introduces a list; the items are the claims
        or bool(_HEDGE_RE.search(stripped))
    )


# --- evidence ------------------------------------------------------------------

# Entries are separated by one or two newlines: the first provision follows the
# block's header line directly, and splitting on blank lines only would drop it.
_ENTRY_SPLIT_RE = re.compile(r"\n+(?=• \*\*)")
_ENTRY_HEAD_RE = re.compile(r"^• \*\*(?P<head>.+?)\*\*")
_SECTION_HEAD_RE = re.compile(r"^.+?\s+(?:§\s*|Article\s+)(?P<sec>[0-9A-Za-z\-]+)$")


def _context_passages(context_text: str) -> List[Tuple[str, str]]:
    """(section_number_upper or "", text) for each entry of the retrieved-context
    block the model was actually shown. Case-law entries have no section."""
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
    """The stretch of `passage` (≤ size chars) that best matches the claim. The
    adjudicator used to be shown the first `size` chars, so a claim about the
    third sub-clause of a long section was judged against its preamble."""
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
    """
    Deterministic pass: sentence-split the answer, resolve evidence for
    every citation-bearing or high-risk-absolute sentence, and classify
    each. No LLM calls. Reuses citation_verifier.verify_citations for the
    existing existence/act/retrieved-section checks unchanged.

    A claim is judged against the best-matching *passage* — the provision it
    cites, or any provision/case in the retrieved context — never against the
    context as one blob. A citation that names no Act resolves to the provision
    that was retrieved for this query, not to whichever Act happens to have a
    section with that number.
    """
    citation_report = verify_citations(answer, rag, retrieved_sections)
    occurrences = iter_citation_occurrences(answer)
    spans = _split_sentences(answer)
    by_sentence = _assign_citations_to_sentences(spans, occurrences)
    context_passages = _context_passages(retrieved_context_text)
    context_texts = [t for _, t in context_passages]
    context_blob = "\n".join(context_texts)
    # Case law carries no section number; a claim that cites a section may still
    # legitimately rest on it ("Article 19 … (Shreya Singhal)").
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
            hits = rag.find_section(occ.act_hint, occ.section, max_parts=2) if occ.act_hint else []
            if hits:
                cited.extend(h.text for h in hits)
            else:
                cited.extend(t for sec, t in context_passages if sec == occ.section.upper())
        citations = [occ.raw for occ in occs]
        # A claim that cites a provision is judged against THAT provision (and
        # case law). Falling back to any retrieved passage would let a citation
        # to a nonexistent section ride on vocabulary shared with its neighbours.
        # An uncited claim may rest on anything that was retrieved.
        candidates = list(dict.fromkeys((cited + case_law_texts) if occs else context_texts))
        if len(cited) > 1:
            # A claim may state something that only holds when its cited
            # provisions are read together (e.g. a right Article 19 defines,
            # qualified by the "except ..." Article 21 states) — the joined
            # text is a real candidate, not just an overlap-score bonus. Using
            # it only for the overlap NUMBER while contradiction/quantity still
            # ran against whichever single provision scored higher generically
            # produced false CONTRADICTED flags on faithful multi-citation
            # claims, evidenced against the wrong (higher-overlap but
            # unrelated) provision of the two.
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
            # Quoted or near-verbatim statute is grounded however it spreads
            # across passages.
            if verbatim_frac >= _VERBATIM_FRACTION:
                overlap = max(overlap, 1.0)
            status = _classify_overlap(overlap, high_risk)
            reason = f"~{overlap:.0%} term overlap with the retrieved evidence"
            # Judged against the one passage it matches best, and only when it
            # plausibly is about that passage.
            contradiction = (
                _contradiction_reason(stripped, best) if overlap >= _PARTIAL_THRESHOLD else None
            ) or _unsupported_quantity(stripped, candidates)
            if contradiction:
                status, reason = CONTRADICTED, contradiction
            elif occs and not cited and status != SUPPORTED:
                # Say what is actually wrong, not a meaningless overlap figure
                # against whatever case law happened to be retrieved.
                reason = "the cited provision is not among the retrieved provisions"

        # Worth the adjudicator's time: a high-risk claim that only partly
        # matches, and any cited claim that is not near-verbatim — overlap cannot
        # tell a faithful paraphrase of a section from a fabricated rule about it.
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


# ---------------------------------------------------------------------------
# LLM layer — one batched call, only for what the deterministic pass
# couldn't clear or already condemned.
# ---------------------------------------------------------------------------

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


# Ollama structured-output schema for the correction call above — passed as
# ChatOllama(format=...) by whatever LLM the caller supplies. Grammar-
# constrains the small model's decoding to this exact shape, which is what
# actually gets it to comply with "respond with ONLY a JSON array": plain
# prompting alone reliably produced free-form step-by-step prose instead
# (verified directly — qwen3:4b would reason through each claim in prose
# and get cut off by num_predict before ever emitting JSON).
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
    """Only text the deterministic evidence already condemned is ever rewritten; for the rest
    the LLM's verdict is all that's used, so asking it for a sentence just costs tokens."""
    return s.det_status in (CONTRADICTED, UNGROUNDED)


def _build_claims_block(sentences: List[SentenceGrounding]) -> str:
    lines = []
    for i, s in enumerate(sentences, 1):
        evidence = (s.evidence.strip() or "(no retrieved evidence available)")[:_MAX_EVIDENCE_CHARS]
        rewrite = "yes" if _may_rewrite(s) else "no"
        lines.append(f'{i}. CLAIM: "{s.text.strip()}"\n   EVIDENCE: "{evidence}"\n   REWRITE: {rewrite}')
    return "\n\n".join(lines)


def _extract_json_array(raw: str) -> list:
    """Find the first balanced top-level `[...]` in `raw` and parse it.
    qwen3 leaves a `</think>` preamble in the output even with reasoning
    off/stripped elsewhere, and a greedy `\\[.*\\]` regex over that text can
    span across unrelated brackets in the preamble; scanning for the first
    balanced bracket (string-aware, so brackets inside quoted text don't
    throw off the depth count) is what actually finds the JSON array."""
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
    """Splice a rewritten claim back into its original span, preserving
    leading markdown structure (bullet/heading markers) and trailing
    whitespace/newline so the surrounding document layout is untouched."""
    lead_match = _LEAD_RE.match(original_span)
    trail_match = _TRAIL_WS_RE.search(original_span)
    lead = lead_match.group(1) if lead_match else ""
    trail = trail_match.group(1) if trail_match else ""
    body = corrected_sentence.strip()
    if not body:
        return original_span
    stripped_original = original_span.strip()
    if not body.endswith((".", "!", "?")) and stripped_original.endswith((".", "!", "?")):
        body += "."
    return f"{lead}{body}{trail}"


async def ground_and_correct(
    answer: str,
    rag,
    retrieved_sections: Optional[set] = None,
    retrieved_context_text: str = "",
    llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
) -> Tuple[str, GroundingReport]:
    """
    Run the deterministic grounding pass, then — only for sentences that
    are CONTRADICTED, UNGROUNDED-with-a-citation, or high-risk-and-not-
    clearly-supported — a single batched LLM call to finalize status and
    rewrite from evidence only. Supported sentences are never touched.

    Falls back to the deterministic report (untouched answer text) if
    nothing needs correcting, no `llm_invoke` was supplied, or the LLM
    call/parse fails for any reason — this must never raise or block chat.
    """
    report = assess_grounding(answer, rag, retrieved_sections, retrieved_context_text)

    to_fix = sorted(
        (
            s
            for s in report.sentences
            if s.is_claim
            and (
                (s.status != SUPPORTED and (s.status in (CONTRADICTED, UNGROUNDED) or s.is_high_risk))
                or s.needs_llm
            )
        ),
        key=lambda s: s.overlap,  # weakest evidence first when over the cap
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
            # Verified as stated: never rewrite it, whatever wording came back.
            s.status = SUPPORTED
            continue
        if _may_rewrite(s) and not corrected_sentence:
            s.status = new_status
            continue
        if not _may_rewrite(s):
            # Only the small model objects; the deterministic evidence (quantities,
            # citations, conditions, overlap) does not. Measured on real answers,
            # a 4B model rewriting a sentence from an 800-char window flips
            # correct claims ("not applicable" -> "applicable") and overwrites
            # faithful paraphrases with boilerplate. Flag it, keep the text.
            s.status = new_status
            s.reason = "the fact-check could not confirm this against the retrieved provisions"
            continue
        replacement = _reassemble(original_span, corrected_sentence)
        new_text = new_text[: s.start] + replacement + new_text[s.end :]
        s.status = new_status
        s.outcome = "corrected"

    return new_text, report


def grounding_footer(report: GroundingReport) -> str:
    """Advisory footer summarizing claim-level (not just citation-level)
    grounding. Silent when every claim is supported and nothing needed
    correction — same no-noise-on-good-answers policy as citation_verifier.

    Also silent whenever the LLM adjudication pass didn't run or its output
    couldn't be parsed: the deterministic word-overlap pass alone is too
    blunt (fluent legal prose paraphrases statutory text) to justify
    surfacing every flag to the user, so an unverified deterministic flag
    is treated as "not confident enough to show", not "confidently wrong".
    And even with a verified pass, only surface it once confidence is
    genuinely low — a couple of flagged claims among many supported ones
    isn't worth a scary footer on an otherwise good answer.
    """
    claims = report.claim_sentences
    corrected = [s for s in claims if s.outcome == "corrected"]
    confirmed = [s for s in report.confirmed_flagged if s.outcome != "corrected"]
    advisory = [
        s for s in claims
        if s.status != SUPPORTED and s.outcome != "corrected" and s not in confirmed
    ]
    # Only what the deterministic evidence also condemned is worth a footer; an
    # LLM-only objection alone never produces one.
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
            f"\"{s.text.strip()[:140]}\""
        )
    if advisory:
        lines.append(
            f"- {len(advisory)} other statement(s) could not be confirmed against the "
            f"retrieved provisions."
        )
    return "\n".join(lines)
