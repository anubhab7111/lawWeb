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
from typing import Dict, List, Optional, Tuple

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
    r"(?:\b(?:section|sec\.?|s\.|u/s)\s*(\d{1,4}[A-Z]{0,2})\b|\b(?:article|art\.?)\s*(\d{1,3}[A-Z]{0,2})\b)",
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
        if len(alias) > len(best) and _alias_in(alias, query_lower):
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


# Cosine (BGE-M3) between a query and a doctrine's "name: aliases" text above which
# the doctrine is taken. Picked on 112 ontology-miss questions (ILSIC dev, ground
# truth, live chat): at 0.6 the match agreed with the fast-LLM pick 4/5 and caught
# 2 the LLM gave up on; the LLM gave up on 69/112 after ~19 s each.
DOCTRINE_MATCH_MIN_COS = 0.6
_doctrine_vectors = None


def _add_doctrine(parsed: ParsedLegalQuery, d: dict) -> None:
    parsed.doctrines.append(d["doctrine"])
    if d.get("domain") and d["domain"] not in parsed.domains:
        parsed.domains.append(d["domain"])
    for act, sec in d.get("sections", []):
        if (act, sec) not in parsed.pinned_sections:
            parsed.pinned_sections.append((act, sec))
    parsed.landmark_cases.extend(d.get("landmark_cases", []))
    parsed.expansion_terms.append(d["doctrine"])
    if parsed.query_type == "general":
        parsed.query_type = "doctrine"
    parsed.pinned_sections = parsed.pinned_sections[:8]


async def parse_legal_query_embedding(query: str) -> ParsedLegalQuery:
    """Deterministic parse, plus the closest ontology doctrine by embedding for
    queries the aliases missed. Failures degrade to the deterministic result."""
    global _doctrine_vectors
    parsed = parse_legal_query(query)
    if parsed.doctrines or parsed.pinned_sections:
        return parsed
    try:
        import numpy as np

        from app.tools.base_legal_rag import _get_shared_embeddings

        emb = await _get_shared_embeddings()
        doctrines = _load_ontology()["doctrines"]
        if _doctrine_vectors is None:
            D = np.asarray(emb.embed_documents(
                [f"{d['doctrine']}: {', '.join(d.get('aliases', []))}" for d in doctrines]
            ), dtype=np.float32)
            _doctrine_vectors = D / np.linalg.norm(D, axis=1, keepdims=True)
        v = np.asarray(emb.embed_query(query), dtype=np.float32)
        scores = _doctrine_vectors @ (v / np.linalg.norm(v))
        best = int(np.argmax(scores))
        if scores[best] >= DOCTRINE_MATCH_MIN_COS:
            _add_doctrine(parsed, doctrines[best])
    except Exception as e:
        print(f"[parser] embedding doctrine match failed ({e}) — deterministic only.")
    return parsed
