"""
Pure logic for the LLM-free IL-TUR retrieval evaluation (eval_iltur_retrieval.py):
turning ~7k-char case facts into retrieval windows, fusing per-window rankings,
and scoring the fused section ranking against IL-TUR's label set.

Scoring is by section *number* over IPC provisions: IL-TUR `lsi` labels are IPC
sections only (each label's statute text matches the IPC provision), so a label
counts as found if a retrieved IPC provision has that number.
"""

from __future__ import annotations

import math
import random
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
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold_set, k)
    out["ap"] = average_precision(ranked, gold_set)
    return out


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k: rewards putting gold labels early, not just anywhere."""
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in gold_set)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    found, total = 0, 0.0
    for i, r in enumerate(ranked, start=1):
        if r in gold_set:
            found += 1
            total += found / i
    return total / len(gold_set) if gold_set else 0.0


def macro_f1(rankings: Sequence[Sequence[str]], golds: Sequence[Sequence[str]], k: int) -> float:
    """Macro-F1 over labels when each case predicts its top-k set — the metric
    IL-TUR's lsi task is scored with (there from a classifier's label sets), which
    weights every label equally, so a system that only predicts popular labels
    scores near zero."""
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    for ranked, gold in zip(rankings, golds):
        pred, gold_set = set(ranked[:k]), set(gold)
        for label in pred | gold_set:
            tp[label] += label in pred and label in gold_set
            fp[label] += label in pred and label not in gold_set
            fn[label] += label not in pred and label in gold_set
    labels = set(tp) | set(fp) | set(fn)
    f1 = [2 * tp[l] / (2 * tp[l] + fp[l] + fn[l]) if (2 * tp[l] + fp[l] + fn[l]) else 0.0 for l in labels]
    return sum(f1) / len(f1) if f1 else 0.0


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 1):
    """95% percentile-bootstrap interval of the mean."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def popularity_ranking(case_labels: Iterable[Sequence[str]], depth: int = 30) -> List[str]:
    """Labels ordered by how many cases carry them: the no-input baseline any system
    has to beat before its retrieval can be said to add anything."""
    counts: dict = defaultdict(int)
    for labels in case_labels:
        for label in set(labels):
            counts[label] += 1
    return [l for l, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:depth]


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
        agg[f"ndcg@{k}"] = sum(c[f"ndcg@{k}"] for c in per_case) / n
    agg["map"] = sum(c["ap"] for c in per_case) / n
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
