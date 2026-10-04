import asyncio

import app.tools.base_legal_rag as b
from app.tools.base_legal_rag import LegalChunk, compress_chunks_for_context


class CountingReranker:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, batch_size=16):
        self.pairs += pairs
        return [float(text.lower().count("anticipatory")) for _q, text in pairs]


def chunk(text):
    return LegalChunk(chunk_id="c1", domain="criminal", act_name="Code of Criminal Procedure", section_number="438",
                      title="Direction for grant of bail", text=text, source_file="crpc.pdf")


def test_long_section_scores_few_sentences_and_keeps_the_relevant_one(monkeypatch):
    sentences = ["Section opening rule applies here."]
    sentences += [f"Filler clause number {i} about procedure." for i in range(30)]
    sentences += ["A person may seek anticipatory bail from the Court of Session."]
    reranker = CountingReranker()

    async def fake_reranker():
        return reranker

    monkeypatch.setattr(b, "_get_shared_reranker", fake_reranker)
    out = asyncio.run(compress_chunks_for_context("When can anticipatory bail be granted?", [chunk(" ".join(sentences))]))
    assert len(reranker.pairs) <= 8
    assert out[0].text.startswith("Section opening rule") and "anticipatory bail" in out[0].text
