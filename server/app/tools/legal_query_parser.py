"""
legal_query_parser.py — understand the legal question BEFORE retrieval.

Turns a user query into a structured ParsedLegalQuery:
  - query_type: section_lookup | doctrine | comparison | general
  - pinned_sections: (act_hint, section) pairs that retrieval fetches
    deterministically — no embedding roulette for known doctrine law
  - expansion_terms / doctrines / landmark_cases from the ontology

Three cheap stages, no heavy logic:
  1. Citation regex ("section 420 IPC", "article 21") → direct pins
  2. Ontology alias match (app/data/legal_ontology.json) → doctrine pins
  3. Optional fast-LLM assist (injected callable) that maps unmatched
     phrasing to a known doctrine name — choose-from-list only, so a small
     model does it reliably; any failure silently degrades to stage 1+2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_ontology.json"

# Common act abbreviations → act-name hints (matched as substrings against
# indexed act names, case-insensitive).
ACT_HINTS: Dict[str, str] = {
    "ipc": "Indian Penal Code",
    "indian penal code": "Indian Penal Code",
    "bns": "Bharatiya Nyaya Sanhita",
    "bharatiya nyaya sanhita": "Bharatiya Nyaya Sanhita",
    "crpc": "Code of Criminal Procedure",
    "cr.p.c": "Code of Criminal Procedure",
    "code of criminal procedure": "Code of Criminal Procedure",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "bharatiya nagarik suraksha sanhita": "Bharatiya Nagarik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
    "bharatiya sakshya": "Bharatiya Sakshya Adhiniyam",
    "evidence act": "Indian Evidence Act",
    "contract act": "Indian Contract",
    "ica": "Indian Contract",
    "constitution": "Constitution of India",
    "it act": "Information Technology",
    "information technology act": "Information Technology",
    "consumer protection act": "Consumer Protection",
    "cpa": "Consumer Protection",
    "ni act": "Negotiable Instruments",
    "negotiable instruments act": "Negotiable Instruments",
    "hindu marriage act": "Hindu Marriage",
    "hma": "Hindu Marriage",
    "hindu succession act": "Hindu Succession",
    "hsa": "Hindu Succession",
    "ibc": "Insolvency and Bankruptcy",
    "insolvency and bankruptcy code": "Insolvency and Bankruptcy",
    "companies act": "Companies",
    "cgst": "Central Goods and Services Tax",
    "gst act": "Central Goods and Services Tax",
    "income tax act": "Income Tax",
    "rera": "Real Estate",
    "arbitration act": "Arbitration and Conciliation",
    "transfer of property act": "Transfer of Property",
    "tpa": "Transfer of Property",
    "specific relief act": "Specific Relief",
    "pocso": "POCSO",
    "ndps": "NDPS",
    "pmla": "Prevention of Money Laundering",
    "uapa": "Unlawful Activities",
    "posh": "POSH",
    "copyright act": "Copyright",
    "patents act": "Patents",
    "trade marks act": "Trade Marks",
    "rti act": "Right to Information",
    "domestic violence act": "Protection of Women from Domestic Violence",
    "pwdva": "Protection of Women from Domestic Violence",
}

# "section 420 of the IPC", "sec. 438 CrPC", "u/s 302", "article 21", "art. 356"
_CITATION_RE = re.compile(
    r"(?:\b(?:section|sec\.?|s\.|u/s)\s*(\d{1,4}[A-Z]{0,2})\b|\b(?:article|art\.?)\s*(\d{1,3}[A-Z]?)\b)",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(?:compare|difference between|vs\.?|versus|equivalent of|corresponding)\b",
    re.IGNORECASE,
)


@dataclass
class ParsedLegalQuery:
    query_type: str = "general"  # section_lookup | doctrine | comparison | general
    domains: List[str] = field(default_factory=list)
    doctrines: List[str] = field(default_factory=list)
    # (act_hint, section) — act_hint may be "" when the citation named no act
    pinned_sections: List[Tuple[str, str]] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)
    landmark_cases: List[str] = field(default_factory=list)


@lru_cache()
def _load_ontology() -> dict:
    try:
        with open(_ONTOLOGY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[parser] Could not load legal ontology: {e}")
        return {"doctrines": [], "concordance": []}


@lru_cache()
def doctrine_names() -> List[str]:
    return [d["doctrine"] for d in _load_ontology()["doctrines"]]


def _doctrine_by_name(name: str) -> Optional[dict]:
    for d in _load_ontology()["doctrines"]:
        if d["doctrine"] == name:
            return d
    return None


def concordance_pairs(act_hint: str, section: str) -> List[Tuple[str, str]]:
    """Old↔new code equivalents for a cited section, both directions."""
    out: List[Tuple[str, str]] = []
    hint = act_hint.lower()
    for row in _load_ontology()["concordance"]:
        if row["old_section"] == section and (
            not hint or hint in row["old_act"].lower()
        ):
            out.append((row["new_act"], row["new_section"]))
        if row["new_section"] == section and (
            not hint or hint in row["new_act"].lower()
        ):
            out.append((row["old_act"], row["old_section"]))
    return out


def _find_act_hint(query_lower: str) -> str:
    """Longest act abbreviation/name mentioned in the query, if any."""
    best = ""
    for alias, hint in ACT_HINTS.items():
        if alias in query_lower and len(alias) > len(best):
            best = alias
    return ACT_HINTS.get(best, "")


def _alias_in(alias: str, query_lower: str) -> bool:
    """Word-boundary alias match ('itc' must not match inside 'switch')."""
    return bool(
        re.search(
            r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", query_lower
        )
    )


def _match_doctrines(query_lower: str) -> List[dict]:
    matched = []
    for d in _load_ontology()["doctrines"]:
        if any(_alias_in(alias, query_lower) for alias in d["aliases"]):
            matched.append(d)
    return matched


def parse_legal_query(query: str) -> ParsedLegalQuery:
    """Deterministic parse: citations + ontology aliases. Fast, no LLM."""
    q = query.lower()
    parsed = ParsedLegalQuery()

    # ── 1. Explicit citations ───────────────────────────────────
    act_hint = _find_act_hint(q)
    cited: List[Tuple[str, str]] = []
    for m in _CITATION_RE.finditer(query):
        if m.group(1):  # section N
            cited.append((act_hint, m.group(1).upper()))
        else:  # article N → Constitution unless another act named
            cited.append((act_hint or "Constitution of India", m.group(2).upper()))
    parsed.pinned_sections.extend(cited)

    # ── 2. Doctrine aliases ─────────────────────────────────────
    for d in _match_doctrines(q):
        parsed.doctrines.append(d["doctrine"])
        if d.get("domain") and d["domain"] not in parsed.domains:
            parsed.domains.append(d["domain"])
        for act, sec in d.get("sections", []):
            if (act, sec) not in parsed.pinned_sections:
                parsed.pinned_sections.append((act, sec))
        parsed.landmark_cases.extend(d.get("landmark_cases", []))
        parsed.expansion_terms.append(d["doctrine"])

    # ── 3. Query type ───────────────────────────────────────────
    if _COMPARISON_RE.search(query) and (cited or parsed.doctrines):
        parsed.query_type = "comparison"
        # Pull cross-code equivalents for every cited section
        for act, sec in list(cited):
            for pair in concordance_pairs(act, sec):
                if pair not in parsed.pinned_sections:
                    parsed.pinned_sections.append(pair)
    elif cited and len(query.split()) <= 10:
        parsed.query_type = "section_lookup"
    elif parsed.doctrines:
        parsed.query_type = "doctrine"

    # Keep pins bounded — top doctrines already come first
    parsed.pinned_sections = parsed.pinned_sections[:8]
    return parsed


async def parse_legal_query_llm(
    query: str,
    llm_invoke: Callable[[str], Awaitable[str]],
) -> ParsedLegalQuery:
    """
    Deterministic parse, plus a fast-LLM assist for queries the ontology
    aliases missed: the model picks matching doctrine names from the known
    list (choose-from-list, reliable for small models). Failures degrade
    silently to the deterministic result.
    """
    parsed = parse_legal_query(query)
    if parsed.doctrines or parsed.pinned_sections:
        return parsed

    names = doctrine_names()
    prompt = (
        "You classify Indian legal questions. From this list of legal "
        "doctrines, pick the 1-2 that best match the user's question. "
        "Reply with ONLY the exact doctrine name(s), one per line, or NONE.\n\n"
        f"Doctrines: {', '.join(names)}\n\nQuestion: {query}\n"
    )
    try:
        reply = await llm_invoke(prompt)
        for line in reply.strip().splitlines()[:2]:
            d = _doctrine_by_name(line.strip().strip('"').strip("-• ").lower())
            if d is None:
                # tolerate case differences
                for name in names:
                    if name.lower() == line.strip().lower():
                        d = _doctrine_by_name(name)
                        break
            if d:
                parsed.doctrines.append(d["doctrine"])
                if d.get("domain") and d["domain"] not in parsed.domains:
                    parsed.domains.append(d["domain"])
                for act, sec in d.get("sections", []):
                    if (act, sec) not in parsed.pinned_sections:
                        parsed.pinned_sections.append((act, sec))
                parsed.landmark_cases.extend(d.get("landmark_cases", []))
                parsed.expansion_terms.append(d["doctrine"])
        if parsed.doctrines and parsed.query_type == "general":
            parsed.query_type = "doctrine"
        parsed.pinned_sections = parsed.pinned_sections[:8]
    except Exception as e:
        print(f"[parser] LLM doctrine assist failed ({e}) — deterministic only.")
    return parsed
