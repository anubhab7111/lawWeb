import asyncio

from app import tool_dispatch as td
from app.tools.case_law_rag import CaseRecord
from app.tools.judgment_rag import JudgmentPassage


def case(cid, name, summary):
    return CaseRecord(case_id=cid, case_name=name, citation="(2014) 8 SCC 273", court="Supreme Court",
                      bench_size=2, court_rank=1, date="2014-07-02", status="good law", summary=summary, text=summary)


def passage(cid, title, text):
    return JudgmentPassage(cid, cid.split("#")[0], 1995, title, "[1995] 1 S.C.R. 1", "body", [], [], text)


class KeywordReranker:
    def predict(self, pairs, batch_size=16):
        return [0.05 + 0.3 * text.lower().count("arrest") for _q, text in pairs]


def run(cases, passages, reranker):
    async def fake_get():
        return reranker

    import app.tools.base_legal_rag as b

    saved = b._get_shared_reranker
    b._get_shared_reranker = fake_get
    try:
        return asyncio.run(td._reranked_case_law_text("arrest without warrant", cases, passages))
    finally:
        b._get_shared_reranker = saved


def test_most_relevant_authority_wins_whichever_index_found_it():
    cases = [case("arnesh", "Arnesh Kumar v. State of Bihar", "guidelines on arrest arrest arrest")]
    passages = [passage("sc-1#001", "A v. B", "land dispute"), passage("sc-2#004", "C v. D", "arrest was illegal arrest")]
    text = run(cases, passages, KeywordReranker())
    assert text.index("Arnesh Kumar") < text.index("C v. D")
    assert "A v. B" not in text  # below the relative cut-off


def test_judgment_passages_are_capped_and_cited():
    passages = [passage(f"sc-{i}#001", f"Case {i}", "arrest " * (i + 1)) for i in range(6)]
    text = run([], passages, KeywordReranker())
    assert text.count("Supreme Court of India") == 4
    assert "[1995] 1 S.C.R. 1" in text


def test_without_a_reranker_landmark_cases_come_first():
    cases = [case("arnesh", "Arnesh Kumar v. State of Bihar", "guidelines")]
    passages = [passage("sc-2#004", "C v. D", "arrest")]
    text = run(cases, passages, None)
    assert text.index("Arnesh Kumar") < text.index("C v. D")
