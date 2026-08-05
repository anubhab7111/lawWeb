"""
citation_verifier.py — post-generation statutory citation check (no LLM).

Extracts every "Section N of the X Act" / "Article N" claim from a generated
answer and checks it against the indexed statute corpus:

  - verified:    the section exists under the cited act
  - wrong_act:   the section number exists, but only under a different act
                 (the classic hallucination: right number, wrong statute)
  - unverified:  not found in the indexed corpus at all (may be a real
                 provision of a statute we don't index — flagged, not called
                 false)

Case names are NOT checked (no case-law corpus yet — Phase 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.tools.legal_query_parser import ACT_HINTS

# "Section 420 of the IPC", "Section 438, CrPC", "Sections 73 and 74 of the
# Indian Contract Act", "u/s 302 IPC", "Article 21 of the Constitution"
_CITE_RE = re.compile(
    r"(?:\bsections?\s+(\d{1,4}[A-Z]{0,2})(?:\(\d+[a-z]?\))?\b"
    r"|\bu/s\s*(\d{1,4}[A-Z]{0,2})\b"
    r"|\barticles?\s+(\d{1,3}[A-Z]?)\b)",
    re.IGNORECASE,
)

# Sort longest-first so "code of criminal procedure" beats "crpc" etc.
_ACT_ALIASES = sorted(ACT_HINTS.items(), key=lambda kv: -len(kv[0]))


@dataclass
class CitationCheck:
    raw: str  # e.g. "Section 420 of the IPC"
    section: str
    act_hint: str  # resolved act-name hint ("" if none found nearby)
    status: str  # verified | not_retrieved | wrong_act | unverified
    found_in: List[str] = field(default_factory=list)  # acts that DO have it


@dataclass
class VerificationReport:
    checks: List[CitationCheck] = field(default_factory=list)

    @property
    def verified(self) -> List[CitationCheck]:
        return [c for c in self.checks if c.status == "verified"]

    @property
    def not_retrieved(self) -> List[CitationCheck]:
        return [c for c in self.checks if c.status == "not_retrieved"]

    @property
    def wrong_act(self) -> List[CitationCheck]:
        return [c for c in self.checks if c.status == "wrong_act"]

    @property
    def unverified(self) -> List[CitationCheck]:
        return [c for c in self.checks if c.status == "unverified"]

    @property
    def pass_rate(self) -> float:
        return len(self.verified) / len(self.checks) if self.checks else 1.0


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    """Bounds of the sentence containing [start, end), so act-hint search
    doesn't bleed into a neighbouring citation's own act name."""
    left = text.rfind(".", 0, start)
    left = left + 1 if left != -1 else 0
    right = text.find(".", end)
    right = right if right != -1 else len(text)
    return left, right


def _act_hint_near(text: str, start: int, end: int, is_article: bool) -> str:
    """
    Resolve which act a citation refers to from its own sentence, preferring
    the alias positioned CLOSEST to the citation (not just longest-alias-
    wins) — "Section 420 of the IPC. Section 438 of the CrPC..." must not
    let "CrPC" in the next sentence steal the first citation's act.
    """
    left, right = _sentence_bounds(text, start, end)
    window = text[left:right].lower()
    offset = start - left

    best_hint = ""
    best_dist = None
    for alias, hint in _ACT_ALIASES:
        idx = window.find(alias)
        if idx == -1:
            continue
        dist = abs(idx - offset)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_hint = hint
    if best_hint:
        return best_hint
    return "Constitution of India" if is_article else ""


@dataclass
class CitationOccurrence:
    """One raw citation mention in the answer, with its resolved act hint.

    Unlike ``VerificationReport.checks`` (deduplicated — one entry per unique
    provision cited anywhere in the answer), this keeps every mention so
    callers that need to tie a citation back to the specific sentence it
    appears in (e.g. claim-level grounding) can do so.
    """

    start: int
    end: int
    raw: str
    section: str
    act_hint: str
    is_article: bool


def iter_citation_occurrences(answer: str) -> List[CitationOccurrence]:
    """Every citation mention in `answer`, in reading order, undeduplicated."""
    occurrences = []
    for m in _CITE_RE.finditer(answer):
        section = (m.group(1) or m.group(2) or m.group(3)).upper()
        is_article = m.group(3) is not None
        act_hint = _act_hint_near(answer, m.start(), m.end(), is_article)
        raw = answer[m.start() : m.end()]
        if act_hint:
            raw = f"{raw} ({act_hint})"
        occurrences.append(
            CitationOccurrence(m.start(), m.end(), raw, section, act_hint, is_article)
        )
    return occurrences


def verify_citations(answer: str, rag, retrieved_sections=None) -> VerificationReport:
    """
    Check every statutory citation in `answer` against the chunk index of
    an initialized RAG system (must expose find_section()).

    When ``retrieved_sections`` (a set of normalized section-number strings,
    e.g. ``{"438", "482", "21"}``) is supplied, a citation that exists in the
    corpus but was NOT among the sections actually retrieved for this query is
    flagged ``not_retrieved`` — the model stated it from general knowledge
    rather than from the grounded context.
    """
    report = VerificationReport()
    seen: set = set()

    for occ in iter_citation_occurrences(answer):
        key = (occ.act_hint, occ.section, occ.is_article)
        if key in seen:
            continue
        seen.add(key)

        hits = rag.find_section(occ.act_hint, occ.section, max_parts=1)
        if hits:
            status = "verified"
            if retrieved_sections is not None and occ.section not in retrieved_sections:
                status = "not_retrieved"
            report.checks.append(CitationCheck(occ.raw, occ.section, occ.act_hint, status))
            continue

        # Right number, wrong act?
        anywhere = rag.find_section("", occ.section, max_parts=5)
        acts = sorted({c.act_name for c in anywhere})
        status = "wrong_act" if (occ.act_hint and acts) else "unverified"
        report.checks.append(
            CitationCheck(occ.raw, occ.section, occ.act_hint, status, found_in=acts[:4])
        )

    return report


def verification_footer(report: VerificationReport) -> str:
    """
    Human-readable footer appended to answers when any citation could not
    be confirmed. Silent when everything verified (no noise on good answers).
    """
    problems = report.wrong_act + report.unverified + report.not_retrieved
    if not problems:
        return ""
    lines = ["", "---", "⚠️ **Citation check** (against indexed statutes):"]
    for c in report.not_retrieved:
        lines.append(
            f"- {c.raw}: exists in the statute database but was NOT among the "
            f"provisions retrieved for this query — stated from general legal "
            f"knowledge, not the grounded context. Please verify."
        )
    for c in report.wrong_act:
        alts = ", ".join(c.found_in) if c.found_in else "other statutes"
        lines.append(
            f"- {c.raw}: not found under the cited Act — a section with this "
            f"number exists in {alts}. Please verify."
        )
    for c in report.unverified:
        lines.append(
            f"- {c.raw}: could not be verified against the indexed corpus. "
            f"It may belong to a statute outside this database — please verify."
        )
    return "\n".join(lines)
