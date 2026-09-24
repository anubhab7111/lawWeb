"""
unified_rerank.py — one cross-encoder pass over everything retrieved for a query.

Statute sections, curated case-law summaries and judgment passages are retrieved by
different indices whose own scores are not comparable. Here every candidate is
scored against the query by the same cross-encoder, so the final ordering is on one
scale across sources:

  * long items are split into segments and keep their best segment's score, so a
    relevant sentence late in a long passage is not lost to input truncation;
  * scores are relative to the best item (cross-encoder probabilities are ordinal,
    not calibrated), items below `min_relative` of the best are dropped;
  * `per_source_cap` keeps one source (e.g. many passages of one judgment) from
    crowding out the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

SEGMENT_CHARS = 1400  # ~350 tokens: leaves room for the query inside a 512-token input
MAX_SEGMENTS = 4


@dataclass
class Candidate:
    source: str  # "statute" | "case_law" | "judgment" | ...
    key: str  # unique within source
    text: str
    header: str = ""  # e.g. "Indian Penal Code s.302 — Punishment for murder"
    payload: Any = None  # the original object, returned untouched
    score: float = 0.0
    segment_scores: List[float] = field(default_factory=list)


def segments(text: str, size: int = SEGMENT_CHARS, limit: int = MAX_SEGMENTS) -> List[str]:
    """Split on paragraph/sentence boundaries into <= limit pieces of ~size chars."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out: List[str] = []
    current = ""
    for part in text.replace("\n\n", "\n").split("\n"):
        pieces = [part] if len(part) <= size else [part[i : i + size] for i in range(0, len(part), size)]
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > size:
                out.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
    if current:
        out.append(current)
    if len(out) > limit:  # keep the start and spread the rest evenly
        step = (len(out) - 1) / (limit - 1)
        out = [out[round(i * step)] for i in range(limit)]
    return out


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x > -60 else 0.0


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    predict: Callable[[List[tuple]], Sequence[float]],
    top_k: int = 10,
    min_relative: float = 0.2,
    per_source_cap: Optional[Dict[str, int]] = None,
) -> List[Candidate]:
    """Score every candidate with `predict` (a cross-encoder's predict over
    (query, text) pairs) and return the best top_k across all sources."""
    pairs, owners = [], []
    for i, c in enumerate(candidates):
        for seg in segments(c.text) or [c.header]:
            body = f"{c.header}\n{seg}" if c.header else seg
            pairs.append((query, body))
            owners.append(i)
    if not pairs:
        return []
    raw = [float(s) for s in predict(pairs)]
    if any(s < 0 or s > 1 for s in raw):
        raw = [_sigmoid(s) for s in raw]
    best: Dict[int, List[float]] = {}
    for i, s in zip(owners, raw):
        best.setdefault(i, []).append(s)
    scored = []
    for i, scores in best.items():
        c = candidates[i]
        c.segment_scores = scores
        c.score = max(scores)
        scored.append(c)
    scored.sort(key=lambda c: -c.score)
    top = scored[0].score if scored else 0.0
    caps = dict(per_source_cap or {})
    out: List[Candidate] = []
    for c in scored:
        if top <= 0 or c.score / top < min_relative:
            break
        if c.source in caps:
            if caps[c.source] <= 0:
                continue
            caps[c.source] -= 1
        c.score = c.score / top
        out.append(c)
        if len(out) >= top_k:
            break
    return out
