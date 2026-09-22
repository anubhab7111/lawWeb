"""
case_citation_verifier.py — post-generation case-name citation check (no LLM).

Extracts every "<Party> v. <Party> (<Year>)"-shaped case-name citation from
a generated answer and checks the name against the indexed landmark-judgment
corpus (case_law_rag.py):

  - verified:   the name matches an indexed case (exact or close fuzzy match)
  - unverified: not found in the indexed corpus (may be a real judgment
                outside this curated database — flagged, not called false)

Formal citation strings (AIR/SCC/SCR reporter numbers) are NOT checked: the
corpus's `citation` field is a heterogeneous blob of many alternate reporter
citations per case in inconsistent formats (inspected directly against the
real corpus during design), unlike the clean, consistent `case_name` field.
Matching those reliably needs its own normalization effort — tracked as
future work, not attempted half-reliably here.

Extraction requires a trailing "(<Year>)" — Indian case citations in this
corpus's style almost always carry one, and requiring it keeps this from
flagging ordinary "X versus Y" prose that has nothing to do with a court
judgment. This is a deliberate precision-over-recall choice: a missed real
citation without a year is a false negative (safe, per this module's family
of checks); flagging unrelated prose as an unverifiable case would be a
false positive (not safe — see citation_verifier.py's docstring for the
same principle applied to statute citations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from app.tools.case_law_rag import normalize_case_name

# A party phrase: a capitalized word, optionally followed by more capitalized
# words or a small set of lowercase connectors ("Union of India", "Attorney
# General for India", "State of Punjab").
_CONN = r"(?:of|for|the|and|de|van|von)"
_WORD = r"[A-Z][A-Za-z.&'’-]*"
_PARTY_RE = rf"{_WORD}(?:\s+(?:{_WORD}|{_CONN})){{0,6}}"

# "\*{0,2}" between the second party and the year tolerates a generated
# answer bold-wrapping the case name ("**X v. Y** (2015)"), the format
# CASE_LAW_CONTEXT_BLOCK's retrieved entries already use.
_CASE_CITE_RE = re.compile(
    rf"(?P<p1>{_PARTY_RE})\s+(?:v\.?|vs\.?|versus)\s+(?P<p2>{_PARTY_RE})"
    rf"\*{{0,2}}\s*\((?P<year>\d{{4}})\)"
)


@dataclass
class CaseCitationCheck:
    raw: str  # e.g. "Shreya Singhal v. Union of India (2015)"
    name: str  # "Shreya Singhal v. Union of India" (year stripped)
    status: str  # verified | unverified


@dataclass
class CaseCitationReport:
    checks: List[CaseCitationCheck] = field(default_factory=list)

    @property
    def verified(self) -> List[CaseCitationCheck]:
        return [c for c in self.checks if c.status == "verified"]

    @property
    def unverified(self) -> List[CaseCitationCheck]:
        return [c for c in self.checks if c.status == "unverified"]


def iter_case_citations(answer: str) -> List[Tuple[str, str]]:
    """Every distinct "<Party> v. <Party> (<Year>)" case-name citation found
    in `answer`, in reading order, deduplicated by normalized name. Returns
    (raw_text, display_name) pairs — display_name has the year stripped."""
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for m in _CASE_CITE_RE.finditer(answer):
        name = f"{m.group('p1')} v. {m.group('p2')}"
        key = normalize_case_name(name)
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(0), name))
    return out


def verify_case_citations(answer: str, case_rag) -> CaseCitationReport:
    """Check every case-name citation in `answer` against `case_rag.find_case`.
    Caller must confirm case_rag.initialized before calling this — it never
    initializes the RAG system itself (initialization does real I/O and must
    not happen inside a verification gate)."""
    report = CaseCitationReport()
    for raw, name in iter_case_citations(answer):
        found = case_rag.find_case(name)
        report.checks.append(
            CaseCitationCheck(raw=raw, name=name, status="verified" if found else "unverified")
        )
    return report


def case_verification_footer(report: CaseCitationReport) -> str:
    """Advisory footer for unverified case-name citations. Silent when
    everything verified or nothing was cited — same no-noise-on-good-answers
    convention as citation_verifier.verification_footer."""
    if not report.unverified:
        return ""
    lines = ["", "---", "⚠️ **Case citation check** (against the indexed case-law corpus):"]
    for c in report.unverified:
        lines.append(
            f"- {c.name}: could not be verified against the indexed case-law corpus. "
            f"It may be a real judgment outside this database — please verify."
        )
    return "\n".join(lines)
