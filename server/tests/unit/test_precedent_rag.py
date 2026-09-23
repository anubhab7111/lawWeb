import json

import faiss
import numpy as np

from app.config import get_settings
from app.tools import precedent_rag as pr


def test_vote_labels_weights_by_similarity_and_counts_case_once():
    sims = np.array([[0.9, 0.85, 0.3]])
    cases = np.array([[0, 0, 1]])  # two windows of case 0, one of case 1
    labels = [["302", "34"], ["420"]]
    votes = pr.vote_labels(sims, cases, labels, power=1.0)
    assert votes["302"] == votes["34"] == 0.9  # case 0 counted once, at its best window
    assert votes["420"] == 0.3


def test_vote_labels_excludes_cases_and_averages_query_windows():
    sims = np.array([[0.9], [0.9]])
    cases = np.array([[0], [1]])
    labels = [["302"], ["420"]]
    assert pr.vote_labels(sims, cases, labels, exclude={1}, power=1.0) == {"302": 0.45}


def test_rank_votes_orders_by_score_then_label():
    assert pr.rank_votes({"b": 1.0, "a": 1.0, "c": 2.0}) == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


def _tiny_index(tmp_path):
    vecs = np.eye(4, dtype=np.float32)  # 4 unit windows, 2 per case
    owner = np.array([0, 0, 1, 1], dtype=np.int32)
    index = faiss.IndexScalarQuantizer(4, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / pr.INDEX_FILE))
    np.save(tmp_path / pr.WINDOW_CASE_FILE, owner)
    (tmp_path / pr.META_FILE).write_text(
        json.dumps(
            {
                "embedding_model": get_settings().embedding_model,
                "case_ids": ["c0", "c1"],
                "case_labels": [["302"], ["420"]],
            }
        )
    )
    return pr.PrecedentIndex(tmp_path)


def test_index_round_trip_and_case_exclusion(tmp_path):
    index = _tiny_index(tmp_path)
    assert index.available
    query = np.array([[1, 0, 0, 0]], dtype=np.float32)  # matches a window of c0
    top = index.rank_sections(query, k=2, power=1.0)
    assert top[0][0] == "302"
    masked = index.rank_sections(query, k=2, power=1.0, exclude=index.case_index(["c0"]))
    assert all(label != "302" for label, _ in masked)


def test_unbuilt_index_is_unavailable(tmp_path):
    assert not pr.PrecedentIndex(tmp_path).available
