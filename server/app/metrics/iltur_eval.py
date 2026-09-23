"""
Pure logic for the LLM-free IL-TUR retrieval evaluation (eval_iltur_retrieval.py):
turning ~7k-char case facts into retrieval windows, fusing per-window rankings,
and scoring the fused section ranking against IL-TUR's label set.

Scoring is by section *number* over IPC provisions: IL-TUR `lsi` labels are IPC
sections only (each label's statute text matches the IPC provision), so a label
counts as found if a retrieved IPC provision has that number.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

_ENTITY_RE = re.compile(r"<ENTITY>")
_LEADING_PARA_RE = re.compile(r"^\s*\d+\s*[.)]\s*")
RRF_K = 60


def clean_fact_sentence(sentence: str) -> str:
    """Drop paragraph numbering and turn IL-TUR's <ENTITY> masks into a
    neutral noun so retrieval queries read as prose."""
    sentence = _LEADING_PARA_RE.sub("", sentence)
    return _ENTITY_RE.sub("a person", sentence).strip()


def fact_windows(
    sentences: Sequence[str],
    window: int = 4,
    stride: int = 3,
    max_windows: int = 8,
) -> List[str]:
    """Overlapping sentence windows over a case's facts.

    Cases average ~7k characters, far past what one retrieval query can carry,
    so each window is retrieved separately and the rankings are fused. When a
    case has more windows than max_windows they are sampled evenly, keeping
    coverage of the whole narrative rather than only its opening.
    """
    cleaned = [c for c in (clean_fact_sentence(s) for s in sentences) if c]
    if not cleaned:
        return []
    starts = list(range(0, max(len(cleaned) - window, 0) + 1, stride))
    if starts[-1] + window < len(cleaned):
        starts.append(len(cleaned) - window)
    if len(starts) > max_windows:
        step = (len(starts) - 1) / (max_windows - 1)
        starts = [starts[round(i * step)] for i in range(max_windows)]
    return [" ".join(cleaned[s : s + window]) for s in starts]


def rrf_fuse(
    rankings: Iterable[Sequence[str]],
    k: int = RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[str]:
    """Reciprocal-rank fusion of rankings (per-window, or per-system when
    combining statute and precedent evidence), optionally weighted."""
    scores: Dict[str, float] = defaultdict(float)
    for i, ranking in enumerate(rankings):
        weight = weights[i] if weights is not None else 1.0
        for rank, item in enumerate(ranking, start=1):
            scores[item] += weight / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def case_metrics(ranked: Sequence[str], gold: Sequence[str], ks=(1, 3, 5, 10)) -> Dict:
    gold_set = set(gold)
    out: Dict = {}
    for k in ks:
        top = set(ranked[:k])
        found = len(top & gold_set)
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"recall@{k}"] = found / len(gold_set) if gold_set else 0.0
        out[f"tp@{k}"] = found
        out[f"pred@{k}"] = min(k, len(ranked))
    out["rr"] = next(
        (1.0 / r for r, item in enumerate(ranked, start=1) if item in gold_set), 0.0
    )
    return out


def aggregate(per_case: Sequence[Dict], gold_sizes: Sequence[int], ks=(1, 3, 5, 10)):
    """Mean Hit@k / Recall@k / MRR plus micro-F1@k (pooled TP/FP/FN)."""
    n = len(per_case)
    if not n:
        return {}
    agg: Dict = {"n": n, "mrr": sum(c["rr"] for c in per_case) / n}
    for k in ks:
        agg[f"hit@{k}"] = sum(c[f"hit@{k}"] for c in per_case) / n
        agg[f"recall@{k}"] = sum(c[f"recall@{k}"] for c in per_case) / n
        tp = sum(c[f"tp@{k}"] for c in per_case)
        pred = sum(c[f"pred@{k}"] for c in per_case)
        gold = sum(gold_sizes)
        precision = tp / pred if pred else 0.0
        recall = tp / gold if gold else 0.0
        agg[f"micro_f1@{k}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return agg


def label_coverage(labels: Sequence[str], indexed_numbers: Iterable[str]) -> Dict:
    """Which IL-TUR label sections have at least one indexed IPC chunk."""
    have = {n.upper() for n in indexed_numbers}
    missing = [label for label in labels if label.upper() not in have]
    return {
        "labels": len(labels),
        "covered": len(labels) - len(missing),
        "missing": missing,
    }
