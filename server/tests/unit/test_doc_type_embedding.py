import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools import document_classifier as dc

REFS = (
    np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1]], dtype=np.float32),
    np.array(["Sale Deed", "Sale Deed", "Will / Testament", "Will / Testament"]),
)


def test_vector_picks_the_nearest_reference_type(monkeypatch):
    monkeypatch.setattr(dc, "_load_references", lambda: REFS)
    result = dc.DocumentClassifier().classify("I bequeath my house in Pune to my son.", [0.1, 1.0, 0.0])
    assert result.document_type == "Will / Testament" and type(result.document_type) is str
    assert result.matched_indicators == ["embedding: nearest reference documents"]


def test_without_a_vector_the_regex_still_classifies(monkeypatch):
    monkeypatch.setattr(dc, "_load_references", lambda: REFS)
    assert dc.DocumentClassifier().classify("LAST WILL AND TESTAMENT. I bequeath ...").document_type == "Will / Testament"


def test_classify_document_falls_back_to_regex_when_embedding_fails(monkeypatch):
    monkeypatch.setattr(dc, "_load_references", lambda: REFS)

    async def broken():
        raise RuntimeError("no model")

    monkeypatch.setattr("app.tools.base_legal_rag._get_shared_embeddings", broken)
    result = asyncio.run(dc.classify_document("FIRST INFORMATION REPORT under section 154 Cr.P.C."))
    assert result.document_type == "FIR"
