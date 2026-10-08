import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.unified_legal_rag import _GenericDomainParser


def _rag(doc_ids, chunk_ids):
    rag = _GenericDomainParser.__new__(_GenericDomainParser)
    rag._domain = "unified"
    docs = {i: SimpleNamespace(metadata={"chunk_id": cid}) for i, cid in enumerate(doc_ids)}
    rag.vector_store = SimpleNamespace(docstore=SimpleNamespace(_dict=docs))
    rag._chunks = {cid: object() for cid in chunk_ids}
    rag._cache_path = Path("sections.json")
    return rag


def test_warns_when_indexed_chunks_are_missing_from_the_chunk_cache(capsys):
    _rag(["CRI_BNS_85", "CRI_BNS_303"], ["CRI_BNSBNS_85"])._warn_if_chunk_cache_mismatch()
    assert "WARNING: 2/2 indexed chunks are missing" in capsys.readouterr().out


def test_silent_when_cache_matches_the_index(capsys):
    _rag(["CRI_BNS_85"], ["CRI_BNS_85"])._warn_if_chunk_cache_mismatch()
    assert capsys.readouterr().out == ""
