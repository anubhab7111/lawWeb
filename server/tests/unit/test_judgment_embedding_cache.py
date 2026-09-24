import gzip
import json

import numpy as np
import pytest

from app.config import get_settings
from app.ingest import build_judgment_index as bji


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWWEB_CORPUS_ROOT", str(tmp_path))
    get_settings.cache_clear()
    (tmp_path / "builds" / "judgments" / "selected").mkdir(parents=True)
    (tmp_path / "builds" / "judgments" / "emb").mkdir(parents=True)
    yield tmp_path / "builds" / "judgments"
    monkeypatch.undo()
    get_settings.cache_clear()


def _write_selected(root, year, ids):
    with gzip.open(root / "selected" / f"year={year}.jsonl.gz", "wt") as f:
        for i in ids:
            f.write(json.dumps({"chunk_id": i, "doc_id": "d", "year": year, "case_title": "t", "text": "x"}) + "\n")


def test_vectors_round_trip_keyed_by_chunk_id(tmp_path):
    path = tmp_path / "year=2000.npz"
    vecs = np.arange(6, dtype=np.float16).reshape(3, 2)
    bji._save_vectors(path, ["a", "b", "c"], vecs)
    loaded = bji._load_vectors(path)
    assert list(loaded) == ["a", "b", "c"]
    assert np.array_equal(loaded["b"], vecs[1])
    assert bji._load_vectors(tmp_path / "missing.npz") == {}


def test_legacy_positional_vectors_migrate_to_id_keyed_cache(corpus):
    _write_selected(corpus, 1999, ["c0", "c1"])
    np.save(corpus / "emb" / "year=1999.npy", np.array([[1, 0], [0, 1]], dtype=np.float16))
    bji.migrate_embeddings([1999])
    assert not (corpus / "emb" / "year=1999.npy").exists()
    cached = bji._load_vectors(corpus / "emb" / "year=1999.npz")
    assert np.array_equal(cached["c1"], np.array([0, 1], dtype=np.float16))


def test_migration_drops_vectors_that_no_longer_match_their_selection(corpus):
    _write_selected(corpus, 1999, ["only-one"])
    np.save(corpus / "emb" / "year=1999.npy", np.zeros((2, 2), dtype=np.float16))
    bji.migrate_embeddings([1999])
    assert not (corpus / "emb" / "year=1999.npz").exists()
