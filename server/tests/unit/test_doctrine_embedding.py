import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools import legal_query_parser as lqp

ONTOLOGY = {"doctrines": [
    {"doctrine": "retrenchment", "aliases": ["layoff compensation"], "domain": "labour",
     "sections": [["Industrial Disputes Act", "25F"]], "landmark_cases": []},
    {"doctrine": "forgery", "aliases": ["forged document"], "domain": "criminal",
     "sections": [["Indian Penal Code", "463"]], "landmark_cases": []},
], "concordance": []}


class FakeEmbeddings:
    """Doctrine texts embed to axes; the query embeds to `query_vec`."""

    def __init__(self, query_vec):
        self.query_vec = query_vec
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]][: len(texts)]

    def embed_query(self, text):
        return self.query_vec


def _parse(monkeypatch, query, query_vec):
    emb = FakeEmbeddings(query_vec)

    async def shared():
        return emb

    monkeypatch.setattr(lqp, "_load_ontology", lambda: ONTOLOGY)
    monkeypatch.setattr(lqp, "_doctrine_vectors", None)
    monkeypatch.setattr("app.tools.base_legal_rag._get_shared_embeddings", shared)
    return asyncio.run(lqp.parse_legal_query_embedding(query)), emb


def test_close_doctrine_is_pinned(monkeypatch):
    parsed, _ = _parse(monkeypatch, "What must an employer pay when letting workers go?", [0.9, 0.1, 0.0])
    assert parsed.doctrines == ["retrenchment"]
    assert ("Industrial Disputes Act", "25F") in parsed.pinned_sections
    assert parsed.query_type == "doctrine"


def test_distant_doctrine_is_ignored(monkeypatch):
    parsed, _ = _parse(monkeypatch, "Something unrelated", [0.5, 0.5, 0.7])
    assert parsed.doctrines == []
    assert parsed.pinned_sections == []


def test_ontology_alias_hit_skips_the_embedding(monkeypatch):
    parsed, emb = _parse(monkeypatch, "Is a layoff compensation mandatory?", [0.0, 1.0, 0.0])
    assert emb.document_calls == 0
    assert parsed.doctrines == ["retrenchment"]
