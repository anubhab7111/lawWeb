import json
import sqlite3

import faiss
import numpy as np

from app.config import get_settings
from app.tools import judgment_rag as jr


def _chunk(i, doc, sections):
    return {
        "chunk_id": f"{doc}#{i:03d}", "doc_id": doc, "year": 1990, "case_title": "A v. B", "citation": "",
        "role": "body", "sections_cited": [], "doc_sections": sections, "text": f"text {i}",
    }


def test_doc_sections_round_trip(tmp_path):
    con = jr.create_database(tmp_path / jr.DB_FILE)
    jr.insert_chunks(con, 0, [{**_chunk(0, "sc-x", ["302", "34"]), "sections_cited": ["IPC:302"]}])
    con.commit()
    con.close()
    index = jr.JudgmentIndex(tmp_path)
    index._db = sqlite3.connect(tmp_path / jr.DB_FILE)
    assert index._fetch([0])[0].doc_sections == ["302", "34"]


def test_section_votes_count_a_judgment_once_and_normalise_by_breadth(tmp_path):
    # chunks 0,1 belong to one judgment citing 302 and 34; chunk 2 to another citing only 420
    vecs = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexScalarQuantizer(4, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / jr.INDEX_FILE))
    con = jr.create_database(tmp_path / jr.DB_FILE)
    jr.insert_chunks(
        con, 0, [_chunk(0, "sc-a", ["302", "34"]), _chunk(1, "sc-a", ["302", "34"]), _chunk(2, "sc-b", ["420"])]
    )
    con.commit()
    con.close()
    (tmp_path / jr.META_FILE).write_text(json.dumps({"embedding_model": get_settings().embedding_model}))

    idx = jr.JudgmentIndex(tmp_path)
    idx.load()
    query = np.array([[1, 0, 0, 0]], dtype=np.float32)
    raw = idx.section_votes(query, k=3, power=1.0, normalize=False)
    assert raw["302"] == raw["34"] > raw["420"]  # judgment sc-a votes once, at its best passage
    assert abs(raw["302"] - 1.0) < 1e-3
    normed = idx.section_votes(query, k=3, power=1.0, normalize=True)
    assert abs(normed["302"] - 1.0 / 2**0.5) < 1e-3

    # a judgment overlapping a tuning case is excluded from voting entirely
    excluded = idx.section_votes(query, k=3, power=1.0, normalize=False, exclude_docs={"sc-a"})
    assert "302" not in excluded and "34" not in excluded and "420" in excluded
