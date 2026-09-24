import json

import faiss
import numpy as np

from app.config import get_settings
from app.tools import judgment_rag as jr


def _chunks():
    def chunk(i, text, cites):
        return {
            "chunk_id": f"sc-x#{i:03d}", "doc_id": "sc-x", "year": 1990, "case_title": "A v. B",
            "citation": "[1990] 1 S.C.R. 1", "role": "body", "sections_cited": cites, "text": text,
        }

    return [
        chunk(0, "The accused was convicted of murder under the penal code and sentenced to life", ["IPC:302"]),
        chunk(1, "The tenant failed to pay rent and the landlord sought eviction from the premises", []),
        chunk(2, "Common intention requires a prearranged plan among the accused persons", ["IPC:34"]),
    ]


def _build(tmp_path):
    vecs = np.eye(3, 4, dtype=np.float32)
    index = faiss.IndexScalarQuantizer(4, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / jr.INDEX_FILE))
    con = jr.create_database(tmp_path / jr.DB_FILE)
    jr.insert_chunks(con, 0, _chunks())
    con.commit()
    con.close()
    (tmp_path / jr.META_FILE).write_text(json.dumps({"embedding_model": get_settings().embedding_model}))
    return jr.JudgmentIndex(tmp_path)


def test_fts_query_is_safe_and_deduplicated():
    q = jr.fts_query('He said "murder" -- OR NOT (the) murder; a')
    assert q.count('"murder"') == 1 and "--" not in q


def test_lexical_and_dense_hybrid_retrieval(tmp_path):
    index = _build(tmp_path)
    assert index.available
    vec = np.array([[0, 1, 0, 0]], dtype=np.float32)  # dense points at the rent chunk
    top = index.retrieve("landlord eviction rent", vec, k=2)
    assert top[0].chunk_id == "sc-x#001"
    assert set(top[0].sources) == {"dense", "lexical"}


def test_section_boost_pulls_in_citing_passages(tmp_path):
    index = _build(tmp_path)
    vec = np.array([[0, 1, 0, 0]], dtype=np.float32)
    plain = index.retrieve("unrelated words", vec, k=3)
    boosted = index.retrieve("unrelated words", vec, k=3, boost_sections=["IPC:34"])
    rank = lambda passages: [p.chunk_id for p in passages].index("sc-x#002") if any(p.chunk_id == "sc-x#002" for p in passages) else 99
    assert rank(boosted) <= rank(plain)
    assert "cites" in next(p for p in boosted if p.chunk_id == "sc-x#002").sources


def test_unbuilt_index_is_unavailable(tmp_path):
    assert not jr.JudgmentIndex(tmp_path).available
