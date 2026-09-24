"""
Official-protocol scoring for IL-TUR `lsi`, as the IL-TUR leaderboard computes it.

The leaderboard's `evaluate_lsi` (spaces/Exploration-Lab/IL-TUR-Leaderboard,
eval_utils.py) builds a cases x 100 binary matrix over the label vocabulary and
calls `sklearn.metrics.f1_score(gold, pred, average="macro")`. Consequences
mirrored here:
  * labels are exact names: "Section 294(b)" and "Section 376(2)" are columns of
    their own, distinct from 294 and 376;
  * every one of the 100 columns counts, including labels never in the test set
    (a column with no gold and no prediction scores 0 under sklearn's default);
  * predictions are label *sets*, so a system's scores need a decision rule.

Because macro-F1 is the mean of per-column F1s and each column's F1 depends only on
that column's predictions, per-label thresholds can be tuned exactly, one label at a
time (`fit_label_thresholds`) — on dev, then applied unchanged to test.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

Scores = Mapping[str, float]


def gold_matrix(golds: Sequence[Sequence[str]], vocab: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(vocab)}
    m = np.zeros((len(golds), len(vocab)), dtype=np.int8)
    for row, labels in enumerate(golds):
        for label in labels:
            if label in index:
                m[row, index[label]] = 1
    return m


def score_matrix(scores: Sequence[Scores], vocab: Sequence[str], normalize: bool = True) -> np.ndarray:
    """cases x labels score matrix; `normalize` divides each case by its top score,
    so thresholds mean "at least this fraction of the best label's evidence"."""
    index = {label: i for i, label in enumerate(vocab)}
    m = np.zeros((len(scores), len(vocab)), dtype=np.float64)
    for row, case in enumerate(scores):
        for label, value in case.items():
            if label in index:
                m[row, index[label]] = value
        if normalize:
            top = m[row].max()
            if top > 0:
                m[row] /= top
    return m


def official_macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    """Exactly the leaderboard's computation (percent)."""
    return 100.0 * f1_score(gold, pred, average="macro", zero_division=0)


def per_label_f1(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return f1_score(gold, pred, average=None, zero_division=0)


def predict_topk(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros_like(scores, dtype=np.int8)
    top = np.argsort(-scores, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    pred[rows, top] = 1
    pred[scores <= 0] = 0
    return pred


def predict_threshold(scores: np.ndarray, thresholds, min_labels: int = 1) -> np.ndarray:
    """Labels whose score clears their threshold; at least `min_labels` per case
    (every IL-TUR case has at least one statute)."""
    pred = (scores >= np.asarray(thresholds)[None, :]).astype(np.int8)
    pred[scores <= 0] = 0
    if min_labels:
        best = np.argsort(-scores, axis=1)[:, :min_labels]
        rows = np.arange(scores.shape[0])[:, None]
        pred[rows, best] = np.where(scores[rows, best] > 0, 1, pred[rows, best])
    return pred


def fit_global_threshold(scores: np.ndarray, gold: np.ndarray, grid: Optional[Sequence[float]] = None) -> float:
    grid = grid if grid is not None else np.linspace(0.02, 1.0, 50)
    return max(grid, key=lambda t: official_macro_f1(gold, predict_threshold(scores, [t] * scores.shape[1])))


def fit_label_thresholds(
    scores: np.ndarray,
    gold: np.ndarray,
    fallback: float,
    min_positives: int = 5,
    shrink: float = 0.0,
) -> np.ndarray:
    """Per-label thresholds maximising each label's own dev F1.

    Labels with fewer than `min_positives` dev positives keep `fallback`; `shrink`
    pulls each fitted threshold toward `fallback` (0 = no shrinkage) to limit
    overfitting on labels with few examples.
    """
    thresholds = np.full(scores.shape[1], fallback, dtype=np.float64)
    for j in range(scores.shape[1]):
        y = gold[:, j]
        positives = int(y.sum())
        if positives < min_positives:
            continue
        s = scores[:, j]
        candidates = np.unique(s[s > 0])
        if len(candidates) > 200:
            candidates = np.quantile(s[s > 0], np.linspace(0, 1, 200))
        best_t, best_f1 = fallback, -1.0
        for t in candidates:
            p = s >= t
            tp = int((p & (y == 1)).sum())
            fp = int((p & (y == 0)).sum())
            fn = positives - tp
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        thresholds[j] = (1 - shrink) * best_t + shrink * fallback
    return thresholds


def labels_from_row(pred_row: np.ndarray, vocab: Sequence[str]) -> List[str]:
    return [vocab[i] for i in np.flatnonzero(pred_row)]


def evaluate_rules(
    dev_scores: np.ndarray,
    dev_gold: np.ndarray,
    test_scores: np.ndarray,
    test_gold: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Fit each decision rule on dev, report dev and test official macro-F1."""
    out: Dict[str, Dict[str, float]] = {}
    for k in (1, 2, 3, 4, 5):
        out[f"top{k}"] = {
            "dev": official_macro_f1(dev_gold, predict_topk(dev_scores, k)),
            "test": official_macro_f1(test_gold, predict_topk(test_scores, k)),
        }
    t = fit_global_threshold(dev_scores, dev_gold)
    n = dev_scores.shape[1]
    out["global_threshold"] = {
        "t": t,
        "dev": official_macro_f1(dev_gold, predict_threshold(dev_scores, [t] * n)),
        "test": official_macro_f1(test_gold, predict_threshold(test_scores, [t] * n)),
    }
    for shrink in (0.0, 0.3):
        th = fit_label_thresholds(dev_scores, dev_gold, fallback=t, shrink=shrink)
        out[f"label_thresholds_s{shrink}"] = {
            "dev": official_macro_f1(dev_gold, predict_threshold(dev_scores, th)),
            "test": official_macro_f1(test_gold, predict_threshold(test_scores, th)),
        }
    return out
