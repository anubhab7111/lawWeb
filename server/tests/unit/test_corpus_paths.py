import asyncio
import pickle

import pytest

from app.config import get_settings
from app.ingest import paths
from app.tools.unified_legal_rag import UnifiedLegalRAGSystem


@pytest.fixture
def corpus_env(monkeypatch):
    def set_root(value):
        monkeypatch.setenv("LAWWEB_CORPUS_ROOT", str(value))
        get_settings.cache_clear()

    yield set_root
    monkeypatch.undo()
    get_settings.cache_clear()


def test_unset_corpus_uses_legacy_layout(corpus_env):
    corpus_env("")
    assert paths.statutes_dir() == paths.LEGACY_DATA_DIR
    assert paths.case_law_dir() == paths.LEGACY_DATA_DIR / "case_law"
    assert not paths.corpus_available()


def test_corpus_dirs_win_once_they_exist(corpus_env, tmp_path):
    corpus_env(tmp_path)
    assert paths.statutes_dir() == paths.LEGACY_DATA_DIR  # not moved yet
    (tmp_path / "statutes").mkdir()
    (tmp_path / "case_law" / "curated").mkdir(parents=True)
    assert paths.statutes_dir() == tmp_path / "statutes"
    assert paths.case_law_dir() == tmp_path / "case_law" / "curated"


def test_require_corpus_names_the_missing_mount(corpus_env, tmp_path):
    corpus_env(tmp_path / "not-mounted")
    with pytest.raises(paths.CorpusUnavailable, match="not-mounted"):
        paths.require_corpus()


def _system_with_index(tmp_path, fingerprint, sources_dir):
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    meta = {
        "pdf_fingerprint": fingerprint,
        "embedding_model": get_settings().embedding_model,
    }
    (faiss_dir / "meta.pkl").write_bytes(pickle.dumps(meta))
    (faiss_dir / "sections.json").write_text("{}")
    system = UnifiedLegalRAGSystem()
    system._faiss_dir = faiss_dir
    system._meta_path = faiss_dir / "meta.pkl"
    system._cache_path = faiss_dir / "sections.json"
    system._bare_acts_dir = sources_dir
    return system


def test_unmounted_sources_never_trigger_a_rebuild(tmp_path):
    system = _system_with_index(tmp_path, {"a.pdf": "1:x"}, tmp_path / "missing")
    assert asyncio.run(system._should_rebuild()) is False


def test_changed_sources_still_trigger_a_rebuild(tmp_path):
    sources = tmp_path / "bare_acts"
    sources.mkdir()
    (sources / "a.pdf").write_bytes(b"%PDF changed")
    system = _system_with_index(tmp_path, {"a.pdf": "1:x"}, sources)
    assert asyncio.run(system._should_rebuild()) is True
