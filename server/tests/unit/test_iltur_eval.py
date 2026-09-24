from app.metrics.iltur_eval import (
    aggregate,
    average_precision,
    bootstrap_ci,
    macro_f1,
    ndcg_at_k,
    popularity_ranking,
    case_metrics,
    clean_fact_sentence,
    fact_windows,
    label_coverage,
    rrf_fuse,
)
from app.metrics.iltur_loader import decode_labels

NAMES = ["Section 2", "Section 294(b)", "Section 302", "Section 304A", "Section 376(2)"]


def test_decode_labels_maps_class_ids_to_section_numbers():
    # regression: ids used to be read as section numbers (id 2 -> "Section 2")
    assert decode_labels([2, 3], NAMES) == ["302", "304A"]


def test_decode_labels_drops_subclause_and_dedupes():
    assert decode_labels([1, 4, 4], NAMES) == ["294", "376"]
    assert decode_labels(["Section 376(2)", "IPC_376"]) == ["376"]


def test_clean_fact_sentence():
    raw = "2.<ENTITY> was married to <ENTITY> 5 years ago."
    assert clean_fact_sentence(raw) == "a person was married to a person 5 years ago."


def test_fact_windows_cover_tail_and_respect_cap():
    sentences = [f"Sentence number {i}." for i in range(40)]
    windows = fact_windows(sentences, window=4, stride=3, max_windows=5)
    assert len(windows) == 5
    assert windows[0].startswith("Sentence number 0.")
    assert windows[-1].endswith("Sentence number 39.")


def test_fact_windows_short_and_empty():
    assert fact_windows([]) == []
    assert fact_windows(["Only one."]) == ["Only one."]


def test_rrf_prefers_consistently_ranked_items():
    fused = rrf_fuse([["302", "34"], ["34", "302"], ["34", "120B"]])
    assert fused[0] == "34"


def test_case_metrics_and_aggregate():
    m = case_metrics(["302", "34", "120B"], ["34", "506"], ks=(1, 3))
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["rr"] == 0.5
    assert m["recall@3"] == 0.5
    agg = aggregate([m], [2], ks=(1, 3))
    assert agg["hit@3"] == 1.0 and agg["mrr"] == 0.5
    assert 0.0 < agg["micro_f1@3"] < 1.0


def test_label_coverage():
    cov = label_coverage(["302", "34", "149"], ["302", "34A", "34"])
    assert cov["covered"] == 2 and cov["missing"] == ["149"]


def test_ndcg_rewards_early_placement():
    assert ndcg_at_k(["a", "x"], ["a"], 2) == 1.0
    assert 0 < ndcg_at_k(["x", "a"], ["a"], 2) < 1.0
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_average_precision():
    assert average_precision(["a", "b"], ["a", "b"]) == 1.0
    assert abs(average_precision(["x", "a"], ["a", "b"]) - (0.5 / 2)) < 1e-9


def test_macro_f1_punishes_only_predicting_popular_labels():
    golds = [["34"], ["34"], ["34"], ["482"]]
    always_popular = [["34"]] * 4
    perfect = [["34"], ["34"], ["34"], ["482"]]
    assert macro_f1(perfect, golds, 1) == 1.0
    # micro-style accuracy would be 75%, but the rare label is never found
    assert macro_f1(always_popular, golds, 1) < 0.5


def test_popularity_ranking_and_bootstrap_ci():
    assert popularity_ranking([["34", "302"], ["34"], ["302", "34"], ["420"]])[:2] == ["34", "302"]
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert lo < 0.5 < hi and hi - lo < 0.25
