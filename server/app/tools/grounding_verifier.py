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

_SUPPORTED_THRESHOLD = 0.45
_PARTIAL_THRESHOLD = 0.18
_HIGH_RISK_SUPPORTED_THRESHOLD = 0.60

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


@dataclass
class GroundingReport:
    sentences: List[SentenceGrounding]
    citation_report: VerificationReport

    @property
    def claim_sentences(self) -> List[SentenceGrounding]:
        return [s for s in self.sentences if s.is_claim]

    @property
    def flagged(self) -> List[SentenceGrounding]:
        return [s for s in self.claim_sentences if s.status != SUPPORTED]

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
            if j >= n or text[j].isspace():
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


def _word_overlap(claim: str, evidence: str) -> float:
    """What fraction of the claim's content vocabulary appears in the
    evidence text. Same family of heuristic as
    metrics/generation_metrics._keyword_faithfulness — rough by design,
    it only needs to separate "clearly grounded" from "clearly not"."""
    claim_words = set(_WORD_RE.findall(claim.lower()))
    if not claim_words:
        return 1.0
    evidence_words = set(_WORD_RE.findall(evidence.lower()))
    return len(claim_words & evidence_words) / len(claim_words)


def _trigger_set(text: str) -> set:
    t = text.lower()
    return {trig for trig in CONTRADICTION_TRIGGERS if trig in t}


def _has_high_risk(text: str) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in HIGH_RISK_ABSOLUTES)


def _contradiction_reason(claim: str, evidence: str) -> Optional[str]:
    """Deterministic polarity check: did the claim invent a condition the
    evidence doesn't state, or drop one it does (while asserting an
    absolute)? Word overlap can't catch either — same vocabulary, opposite
    meaning — which is exactly why this check exists separately."""
    if not evidence.strip():
        return None
    claim_trig = _trigger_set(claim)
    evidence_trig = _trigger_set(evidence)
    invented = claim_trig - evidence_trig
    dropped = evidence_trig - claim_trig
    if invented:
        return (
            f"claim adds {', '.join(sorted(invented))!r}, a condition not present "
            f"in the retrieved provision"
        )
    if dropped and _has_high_risk(claim):
        return (
            f"claim states an absolute rule but the retrieved provision qualifies it "
            f"with {', '.join(sorted(dropped))!r}"
        )
    return None


def _classify_overlap(overlap: float, is_high_risk: bool) -> str:
    supported_cut = _HIGH_RISK_SUPPORTED_THRESHOLD if is_high_risk else _SUPPORTED_THRESHOLD
    if overlap >= supported_cut:
        return SUPPORTED
    if overlap >= _PARTIAL_THRESHOLD:
        return PARTIALLY_SUPPORTED
    return UNGROUNDED


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
    """
    citation_report = verify_citations(answer, rag, retrieved_sections)
    occurrences = iter_citation_occurrences(answer)
    spans = _split_sentences(answer)
    by_sentence = _assign_citations_to_sentences(spans, occurrences)

    sentences: List[SentenceGrounding] = []
    for i, (start, end) in enumerate(spans):
        raw_span = answer[start:end]
        stripped = raw_span.strip()
        occs = by_sentence.get(i, [])
        high_risk = _has_high_risk(stripped) if stripped else False

        if not stripped or (not occs and not high_risk):
            sentences.append(
                SentenceGrounding(text=raw_span, start=start, end=end, is_claim=False)
            )
            continue

        if occs:
            evidence_parts: List[str] = []
            for occ in occs:
                hits = rag.find_section(occ.act_hint, occ.section, max_parts=2)
                evidence_parts.extend(h.text for h in hits)
            evidence = "\n".join(evidence_parts)
            citations = [occ.raw for occ in occs]
        else:
            evidence = retrieved_context_text
            citations = []

        if not evidence.strip():
            status, overlap = UNGROUNDED, 0.0
            reason = "no retrieved text is available to verify this claim"
        else:
            overlap = _word_overlap(stripped, evidence)
            status = _classify_overlap(overlap, high_risk)
            reason = f"~{overlap:.0%} term overlap with the retrieved evidence"
            contradiction = _contradiction_reason(stripped, evidence)
            if contradiction:
                status, reason = CONTRADICTED, contradiction

        needs_llm = high_risk and status == PARTIALLY_SUPPORTED

        sentences.append(
            SentenceGrounding(
                text=raw_span,
                start=start,
                end=end,
                citations=citations,
                is_claim=True,
                is_high_risk=high_risk,
                status=status,
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

Then rewrite the claim as a single "corrected" sentence:
- If SUPPORTED, repeat the claim unchanged.
- Otherwise, rewrite it using ONLY facts present in its evidence. If the evidence supports nothing \
useful, write a short sentence stating the retrieved sources do not confirm this and recommend \
consulting a lawyer. Never invent new facts, numbers, or section references that are not in the evidence.

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


def _build_claims_block(sentences: List[SentenceGrounding]) -> str:
    lines = []
    for i, s in enumerate(sentences, 1):
        evidence = (s.evidence.strip() or "(no retrieved evidence available)")[:_MAX_EVIDENCE_CHARS]
        lines.append(f'{i}. CLAIM: "{s.text.strip()}"\n   EVIDENCE: "{evidence}"')
    return "\n\n".join(lines)


def _extract_json_array(raw: str) -> list:
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM output")
    return json.loads(match.group(0))


async def _llm_adjudicate_and_correct(
    sentences: List[SentenceGrounding],
    llm_invoke: Callable[[str], Awaitable[str]],
) -> Dict[int, Tuple[str, str]]:
    prompt = _CORRECTION_PROMPT.format(claims_block=_build_claims_block(sentences))
    raw = await llm_invoke(prompt)
    parsed = _extract_json_array(raw)

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
        corrected = str(item.get("corrected", "")).strip()
        if not corrected:
            continue
        result[idx] = (status, corrected)
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

    to_fix = [
        s
        for s in report.sentences
        if s.is_claim
        and s.status != SUPPORTED
        and (s.status in (CONTRADICTED, UNGROUNDED) or s.is_high_risk)
    ][:_MAX_LLM_CORRECTIONS]

    if not to_fix or llm_invoke is None:
        return answer, report

    try:
        corrections = await _llm_adjudicate_and_correct(to_fix, llm_invoke)
    except Exception as e:
        print(f"[GroundingGate] LLM correction skipped: {e}")
        return answer, report

    if not corrections:
        return answer, report

    new_text = answer
    for i, s in sorted(enumerate(to_fix), key=lambda pair: pair[1].start, reverse=True):
        fix = corrections.get(i)
        if fix is None:
            continue
        new_status, corrected_sentence = fix
        original_span = answer[s.start : s.end]
        if new_status == SUPPORTED and corrected_sentence.strip() == s.text.strip():
            s.status = SUPPORTED
            continue
        replacement = _reassemble(original_span, corrected_sentence)
        new_text = new_text[: s.start] + replacement + new_text[s.end :]
        s.status = new_status
        s.outcome = "corrected"

    return new_text, report


def grounding_footer(report: GroundingReport) -> str:
    """Advisory footer summarizing claim-level (not just citation-level)
    grounding. Silent when every claim is supported and nothing needed
    correction — same no-noise-on-good-answers policy as citation_verifier."""
    claims = report.claim_sentences
    corrected = [s for s in claims if s.outcome == "corrected"]
    still_flagged = [s for s in claims if s.status != SUPPORTED and s.outcome != "corrected"]
    if not corrected and not still_flagged:
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
    for s in still_flagged:
        cite = f" ({', '.join(s.citations)})" if s.citations else ""
        lines.append(
            f"- {s.status.replace('_', ' ').title()}{cite}: {s.reason}. "
            f"\"{s.text.strip()[:140]}\""
        )
    return "\n".join(lines)
