import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app.chatbot as cb
from app.tools import legal_defect_analyzer as lda
from app.tools.document_classifier import DocumentClassification
from app.tools.indian_law_rag import IndianLawContext
from app.tools.statutory_validator import StatutoryValidationResult


class StreamingLLM:
    def __init__(self, words=None, fail=False):
        self.words = words or ["The ", "rent ", "clause ", "is ", "present."]
        self.fail = fail

    async def astream(self, messages):
        if self.fail:
            raise RuntimeError("model unavailable")
        yield type("Chunk", (), {"content": "thinking</think>"})()
        for w in self.words:
            yield type("Chunk", (), {"content": w})()


CLASSIFICATION = DocumentClassification(document_type="Rent Agreement", confidence=0.9, jurisdiction_hints=["Delhi"])
VALIDATION = StatutoryValidationResult(document_type="Rent Agreement", total_checks=4, passed=3, failed=1,
                                       compliance_score=0.75)
LAW = IndianLawContext(document_type="Rent Agreement", applicable_acts=["Transfer of Property Act, 1882"],
                       applicable_sections=["Section 105"], precedent_notes=["Some case"])


def _stream(llm):
    async def go():
        queue = asyncio.Queue()
        token = cb._stream_queue_var.set(queue)
        try:
            result = await lda.LegalDefectAnalyzer(llm).analyze_defects(
                CLASSIFICATION, VALIDATION, LAW, document_text="rent is Rs. 28,000",
                think_output="checklist", stream=True,
            )
        finally:
            cb._stream_queue_var.reset(token)
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return result, items

    cb._llm_breaker.record_success()
    return asyncio.run(go())


def test_streamed_report_equals_the_final_report():
    result, items = _stream(StreamingLLM())
    assert all(isinstance(i, str) for i in items)
    streamed = "".join(items)
    assert streamed == result["formatted_response"]
    assert "The rent clause is present." in streamed
    assert streamed.index("## ⚖️ Legal Analysis") < streamed.index("The rent") < streamed.index("Transfer of Property")


def test_failed_analysis_streams_no_give_up_note_and_returns_the_fallback():
    result, items = _stream(StreamingLLM(fail=True))
    streamed = "".join(items)
    assert cb._INCOMPLETE_GENERATION_NOTE not in streamed
    assert "Based on statutory requirements" in result["llm_analysis"]
    assert streamed != result["formatted_response"]  # the handler's replace event fixes this


def test_non_streaming_report_is_unchanged_by_the_split():
    analyzer = lda.LegalDefectAnalyzer(None)
    full = analyzer._format_final_response(CLASSIFICATION, VALIDATION, LAW, "ANALYSIS", {"think": "checklist"})
    prefix = analyzer._format_prefix(CLASSIFICATION, VALIDATION, {"think": "checklist"})
    assert full == f"{prefix}\nANALYSIS\n{analyzer._format_suffix(LAW)}"
    assert prefix.endswith("## ⚖️ Legal Analysis")
    assert "\nANALYSIS\n\n## 📚 Applicable Indian Law" in full


def _run_handler(monkeypatch, llm):
    from types import SimpleNamespace

    import app.tools.civil_rag as civil
    import app.tools.criminal_rag as criminal

    async def classify(text):
        return CLASSIFICATION

    class LawRag:
        async def retrieve_context(self, **kwargs):
            return LAW

    class NoTool:
        async def initialize(self):
            return False

    async def cached_think(self, classification):
        return "checklist"

    monkeypatch.setattr(cb, "classify_document", classify)
    monkeypatch.setattr(cb, "get_statutory_validator", lambda: SimpleNamespace(validate=lambda t, d: VALIDATION))
    monkeypatch.setattr(cb, "get_indian_kanoon_tool", lambda: NoTool())
    monkeypatch.setattr(criminal, "get_criminal_rag_system", lambda: NoTool())
    monkeypatch.setattr(civil, "get_civil_rag_system", lambda: NoTool())
    monkeypatch.setattr(cb, "get_indian_law_rag", lambda *a, **k: LawRag())
    monkeypatch.setattr(cb, "get_llm", lambda: llm)
    monkeypatch.setattr(lda.LegalDefectAnalyzer, "think", cached_think)
    cb._llm_breaker.record_success()

    async def go():
        queue = asyncio.Queue()
        token = cb._stream_queue_var.set(queue)
        try:
            state = await cb._handle_document_validation(
                {"document_content": "rent is Rs. 28,000", "messages": [], "current_input": "validate"}
            )
        finally:
            cb._stream_queue_var.reset(token)
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return state, items

    return asyncio.run(go())


def test_handler_streams_status_then_report_and_ends_on_the_exact_report(monkeypatch):
    state, items = _run_handler(monkeypatch, StreamingLLM())
    events = [i for i in items if isinstance(i, dict)]
    assert [e["stage"] for e in events if e["type"] == "status"] == ["classify", "retrieval", "generation"]
    assert events[-1] == {"type": "replace", "content": state["response"]}
    assert "".join(i for i in items if isinstance(i, str)) == state["response"]


def test_handler_replaces_the_stream_when_the_analysis_falls_back(monkeypatch):
    state, items = _run_handler(monkeypatch, StreamingLLM(fail=True))
    last = [i for i in items if isinstance(i, dict)][-1]
    assert last == {"type": "replace", "content": state["response"]}
    assert "Based on statutory requirements" in state["response"]
