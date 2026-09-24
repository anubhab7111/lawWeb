import numpy as np
from sklearn.metrics import f1_score

from app.metrics import iltur_official as io

VOCAB = ["Section 34", "Section 294", "Section 294(b)", "Section 302"]


def test_matches_the_leaderboard_computation_including_absent_labels():
    golds = [["Section 302", "Section 34"], ["Section 294(b)"]]
    preds = [["Section 302"], ["Section 294"]]
    g, p = io.gold_matrix(golds, VOCAB), io.gold_matrix(preds, VOCAB)
    # leaderboard: plain sklearn macro over all vocab columns
    assert io.official_macro_f1(g, p) == 100 * f1_score(g, p, average="macro", zero_division=0)
    # 294 and 294(b) are different columns; 302 right, 34 missed, 294 wrong, 294(b) missed
    assert abs(io.official_macro_f1(g, p) - 25.0) < 1e-9


def test_score_matrix_normalizes_per_case():
    m = io.score_matrix([{"Section 34": 2.0, "Section 302": 4.0}], VOCAB)
    assert m[0].tolist() == [0.5, 0.0, 0.0, 1.0]


def test_threshold_rule_keeps_at_least_one_label():
    scores = np.array([[0.1, 0.0, 0.0, 0.2]])
    pred = io.predict_threshold(scores, [0.9] * 4)
    assert pred.tolist() == [[0, 0, 0, 1]]


def test_topk_never_predicts_zero_scores():
    scores = np.array([[0.5, 0.0, 0.0, 0.0]])
    assert io.predict_topk(scores, 3).tolist() == [[1, 0, 0, 0]]


def test_label_thresholds_are_fitted_per_column_and_improve_dev_f1():
    rng = np.random.default_rng(0)
    gold = (rng.random((400, 2)) < 0.3).astype(np.int8)
    # label 0 is separable at 0.6, label 1 at 0.3
    scores = np.where(gold == 1, [0.7, 0.4], [0.5, 0.2]) + rng.normal(0, 0.02, (400, 2))
    th = io.fit_label_thresholds(scores, gold, fallback=0.5)
    assert 0.5 < th[0] <= 0.7 and 0.2 < th[1] <= 0.4
    tuned = io.official_macro_f1(gold, io.predict_threshold(scores, th, min_labels=0))
    flat = io.official_macro_f1(gold, io.predict_threshold(scores, [0.5, 0.5], min_labels=0))
    assert tuned > flat and tuned > 99


def test_rare_labels_keep_the_fallback_threshold():
    gold = np.zeros((50, 1), dtype=np.int8)
    gold[:2] = 1
    th = io.fit_label_thresholds(np.ones((50, 1)), gold, fallback=0.42, min_positives=5)
    assert th[0] == 0.42
