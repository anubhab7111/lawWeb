"""Scoring for the small-model benchmark (eval_models.py): per-answer metrics, gates and paired bootstrap."""

from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional, Sequence

from app.tools.citation_verifier import iter_citation_occurrences, verify_citations
from app.tools.grounding_verifier import (
    _context_passages,
    _extract_json_array,
    assess_grounding,
)

# Fixed before any model was run; metrics that do not apply to an item are dropped and the rest renormalised.
WEIGHTS = {
    "facts": 0.25,
    "cite_recall": 0.15,
    "cite_precision": 0.20,
    "supported": 0.15,
    "honesty": 0.15,
    "format": 0.10,
}

# Production limits per call role (chatbot.py); a slower call falls back in the app, so it counts as a failure.
ROLE_TIMEOUTS = {"main": 180, "prose": 45, "correction": 60, "fast": 15, "tiebreak": 12}

GATES = {
    "max_failure_rate": 0.02,
    "min_json_valid": 0.95,
    "max_leak_rate": 0.02,
}
QUALITY_TIE_MARGIN = 0.03

_ABSTAIN_RE = re.compile(
    r"(do(es)? not|don't|doesn't|did not|could not|couldn't|cannot|can't|no|not)\s+"
    r"(have|specify|specif\w+|state[sd]?|mention\w*|cover\w*|contain\w*|provide\w*|address\w*|include\w*|"
    r"find|found|retrieve\w*|available|sufficient|enough|any|explicit\w*)"
    r"|not (covered|specified|stated|mentioned|provided|addressed|included|available)"
    r"|no (specific|relevant|matching|such) (reference|provision|section|information|figure)",
    re.IGNORECASE,
)
_LEAK_RE = re.compile(
    r"</?think>|CRITICAL WARNING|CRITICAL ACCURACY RULES|you are HALLUCINATING|your critical rules|"
    r"follows your (critical )?rules|as (instructed|required) (by|in) (the|your) (prompt|rules|instructions)|"
    r"^\s*(okay|ok|alright|hmm)[,.]? (the user|let me|i need)|\bthe user (is asking|wants|asked)\b",
    re.IGNORECASE | re.MULTILINE,
)
_DISCLAIMER_RE = re.compile(
    r"lawyer|advocate|legal (aid|practitioner|professional|advice)|bar council",
    re.IGNORECASE,
)
_CRIME_HEADERS = ("**Crime:**", "**Statute:**", "**Punishment:**", "**Further Steps:**")
_LENGTH_BOUNDS = {
    "A": (200, 9000),
    "B": (150, 9000),
    "C": (120, 9000),
    "D": (150, 4000),
    "E": (300, 16000),
}


def normalise(text: str) -> str:
    text = text.lower().replace("§", " section ")
    text = re.sub(r"[*_`#>]", "", text)
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _alt_found(alt: str, norm_answer: str) -> bool:
    if alt.startswith("re:"):
        return bool(re.search(alt[3:], norm_answer))
    return normalise(alt) in norm_answer


def fact_recall(answer: str, facts: Sequence[Sequence[str]]) -> Optional[float]:
    """Share of gold facts present; each fact is a list of acceptable phrasings ("re:" marks a regex)."""
    if not facts:
        return None
    norm = normalise(answer)
    return sum(any(_alt_found(a, norm) for a in alts) for alts in facts) / len(facts)


def _occurrences(answer: str):
    return iter_citation_occurrences(answer.replace("§", " Section "))


def cite_recall(answer: str, gold: Sequence[dict]) -> Optional[float]:
    """Share of gold {"act", "section"} citations made; a citation with no act named counts for any act."""
    if not gold:
        return None
    occs = _occurrences(answer)
    hits = 0
    for g in gold:
        want_sections = {str(s).upper() for s in g["section"]} if isinstance(g["section"], list) else {str(g["section"]).upper()}
        acts = [a.lower() for a in g.get("acts", [])]
        hits += any(
            o.section in want_sections
            and (not o.act_hint or not acts or any(a in o.act_hint.lower() for a in acts))
            for o in occs
        )
    return hits / len(gold)


def context_sections(prompt: str) -> set:
    return {sec for sec, _ in _context_passages(prompt) if sec}


def cite_precision(answer: str, rag, prompt: str) -> Optional[float]:
    """Share of the answer's statutory citations that exist and were in the context the model was shown."""
    report = verify_citations(answer.replace("§", " Section "), rag, context_sections(prompt))
    if not report.checks:
        return None
    return len(report.verified) / len(report.checks)


def supported_claims(answer: str, rag, prompt: str) -> Optional[float]:
    """Rules-only grounding score (no LLM adjudication) of the answer's claims against the supplied context."""
    report = assess_grounding(
        answer.replace("§", " Section "),
        rag,
        retrieved_sections=context_sections(prompt),
        retrieved_context_text=prompt,
    )
    if not report.claim_sentences:
        return None
    return report.overall_score


def honesty(answer: str, stratum: str, gold: dict) -> Optional[float]:
    """B: nothing forbidden asserted and no section cited from memory. C: abstains and asserts nothing forbidden."""
    if stratum not in ("B", "C"):
        return None
    norm = normalise(answer)
    clean = not any(_alt_found(f, norm) for f in gold.get("forbidden", []))
    if stratum == "B":
        return (clean + (not _occurrences(answer))) / 2
    abstains = bool(_ABSTAIN_RE.search(answer)) or any(
        _alt_found(a, norm) for a in gold.get("abstain", [])
    )
    return (clean + abstains) / 2


def leaked(answer: str) -> bool:
    return bool(_LEAK_RE.search(answer))


def format_score(answer: str, stratum: str, done_reason: str) -> float:
    lo, hi = _LENGTH_BOUNDS[stratum]
    letters = [c for c in answer if c.isalpha()]
    checks = [
        done_reason == "stop",
        lo <= len(answer) <= hi,
        bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.9,
    ]
    if stratum == "D":
        checks.append(all(h in answer for h in _CRIME_HEADERS))
    else:
        checks.append(bool(_DISCLAIMER_RE.search(answer)))
    return sum(checks) / len(checks)


def answer_metrics(answer: str, item: dict, prompt: str, done_reason: str, rag) -> Dict[str, Optional[float]]:
    """All quality metrics for one answer; None marks a metric that does not apply."""
    stratum, gold = item["stratum"], item.get("gold", {})
    if not answer.strip():
        return {k: 0.0 for k in WEIGHTS}
    grounded = stratum != "B"
    return {
        "facts": fact_recall(answer, gold.get("facts", [])),
        "cite_recall": cite_recall(answer, gold.get("cite", [])),
        "cite_precision": cite_precision(answer, rag, prompt),
        "supported": supported_claims(answer, rag, prompt) if grounded else None,
        "honesty": honesty(answer, stratum, gold),
        "format": format_score(answer, stratum, done_reason),
    }


def quality(metrics: Dict[str, Optional[float]]) -> float:
    used = {k: v for k, v in metrics.items() if v is not None and k in WEIGHTS}
    total = sum(WEIGHTS[k] for k in used)
    return sum(WEIGHTS[k] * v for k, v in used.items()) / total if total else 0.0


def json_call_ok(role: str, visible: str, expected: Optional[str] = None, candidates: Sequence[str] = ()) -> Dict[str, bool]:
    """Validity (and, for tie-breaks with a gold label, correctness) of a schema-constrained call."""
    try:
        if role == "tiebreak":
            picked = json.loads(visible).get("intent")
            valid = picked in set(candidates) | {"unsure"} if candidates else isinstance(picked, str)
            return {"valid": valid, "correct": expected is not None and picked == expected}
        rows = _extract_json_array(visible)
        valid = isinstance(rows, list) and all(
            isinstance(r, dict) and {"index", "status", "corrected"} <= set(r) for r in rows
        )
        return {"valid": valid, "correct": valid}
    except Exception:
        return {"valid": False, "correct": False}


def paired_bootstrap(a: List[float], b: List[float], n: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Mean of (a - b) over paired items with a 95% percentile interval."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n)
    )
    return {
        "diff": sum(diffs) / len(diffs),
        "lo": means[int(0.025 * n)],
        "hi": means[int(0.975 * n) - 1],
    }


def gate_results(fit_ok: bool, failure_rate: float, json_valid: Optional[float], leak_rate: float) -> Dict[str, bool]:
    return {
        "fit": fit_ok,
        "reliability": failure_rate <= GATES["max_failure_rate"],
        "json": json_valid is None or json_valid >= GATES["min_json_valid"],
        "no_leak": leak_rate <= GATES["max_leak_rate"],
    }


def recommend(rows: List[dict]) -> Optional[str]:
    """Decision rule: among gate-passing models within QUALITY_TIE_MARGIN of the best quality, the lowest median time."""
    ok = [r for r in rows if all(r["gates"].values())]
    if not ok:
        return None
    best = max(r["quality"] for r in ok)
    close = [r for r in ok if best - r["quality"] <= QUALITY_TIE_MARGIN]
    return min(close, key=lambda r: r["median_seconds"])["model"]
