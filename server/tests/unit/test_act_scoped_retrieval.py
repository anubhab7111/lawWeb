import asyncio
from types import SimpleNamespace

from app.tools.base_legal_rag import LegalChunk
from app.tools.unified_legal_rag import UnifiedLegalRAGSystem


def _chunk(cid, act, section):
    return LegalChunk(
        chunk_id=cid, domain="criminal", act_name=act, section_number=section,
        title="t", text=f"{act} section {section} text", source_file="x.pdf", domains=["criminal"],
    )


class FakeStore:
    """Returns every chunk in a fixed order, applying FAISS-style metadata filtering."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def similarity_search_with_score(self, query, k=4, filter=None, fetch_k=20):
        self.calls.append({"k": k, "fetch_k": fetch_k, "filtered": filter is not None})
        docs = [
            (SimpleNamespace(metadata={"chunk_id": c.chunk_id, "act_name": c.act_name, "domains": "criminal"}), 0.5)
            for c in self.chunks
        ]
        if filter:
            docs = [d for d in docs if filter(d[0].metadata)]
        return docs[:k]


def _rag(chunks):
    rag = UnifiedLegalRAGSystem()
    rag.initialized = True
    rag.vector_store = FakeStore(chunks)
    rag._chunks = {c.chunk_id: c for c in chunks}
    rag._bm25 = None
    return rag


CHUNKS = [
    _chunk("a", "Bharatiya Nyaya Sanhita BNS", "103"),
    _chunk("b", "Indian Penal Code", "302"),
    _chunk("c", "Code of Criminal Procedure", "302"),
    _chunk("d", "Indian Penal Code", "34"),
]


def test_scoping_by_act_excludes_other_acts_and_overfetches():
    rag = _rag(CHUNKS)
    result = asyncio.run(rag.retrieve("murder", k=10, use_reranker=False, min_score=0.0, acts=["Indian Penal Code"]))
    assert {c.section_number for c in result.chunks} == {"302", "34"}
    assert all(c.act_name == "Indian Penal Code" for c in result.chunks)
    call = rag.vector_store.calls[0]
    assert call["filtered"] and call["fetch_k"] >= 30 * 60


def test_unscoped_retrieval_is_unchanged():
    rag = _rag(CHUNKS)
    result = asyncio.run(rag.retrieve("murder", k=10, use_reranker=False, min_score=0.0))
    assert len(result.chunks) == 4
    assert rag.vector_store.calls[0]["filtered"] is False
