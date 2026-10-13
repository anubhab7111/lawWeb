import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace

import pytest
from app.tools import legal_defect_analyzer as lda
from app.tools.document_classifier import DocumentClassification


@pytest.fixture(autouse=True)
def no_case_index(monkeypatch):
    """THINK output is filtered through the case-law index; stub it."""
    import app.tools.case_law_rag as clr

    monkeypatch.setattr(
        clr,
        "get_case_law_rag_system",
        lambda: SimpleNamespace(initialized=True, find_case=lambda name: None),
    )


def _analyzer(outputs):
    analyzer = lda.LegalDefectAnalyzer(llm=None)
    calls = []

    async def fake_step(classification):
        calls.append(classification.document_type)
        out = outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    analyzer._step_think = fake_step
    return analyzer, calls


def test_think_runs_once_per_document_kind(monkeypatch):
    monkeypatch.setattr(lda, "_THINK_CACHE", {})
    analyzer, calls = _analyzer(
        ["rent checklist. Void per Fake v. Party (2018).", "fir checklist"]
    )
    rent = DocumentClassification(
        document_type="Rent Agreement", confidence=0.9, jurisdiction_hints=["Delhi"]
    )
    fir = DocumentClassification(document_type="FIR", confidence=0.9)

    assert asyncio.run(analyzer.think(rent)) == "rent checklist."
    assert asyncio.run(analyzer.think(rent)) == "rent checklist."
    assert asyncio.run(analyzer.think(fir)) == "fir checklist"
    assert calls == ["Rent Agreement", "FIR"]


def test_think_fallback_is_not_cached(monkeypatch):
    monkeypatch.setattr(lda, "_THINK_CACHE", {})
    analyzer, calls = _analyzer([RuntimeError("timeout"), "fir checklist"])
    fir = DocumentClassification(document_type="FIR", confidence=0.9)

    assert asyncio.run(analyzer.think(fir)) == analyzer._fallback_think(fir)
    assert asyncio.run(analyzer.think(fir)) == "fir checklist"
    assert len(calls) == 2


def test_degenerate_checklist_detection():
    runaway = "**THOUGHT 1 — Governing Law:**\nIndian Contract Act, 1872 (Section " + ", ".join(
        str(n) for n in range(10, 200)
    )
    assert lda.is_degenerate_checklist(runaway)
    assert lda.is_degenerate_checklist("word " * 700)
    ok = (
        "**THOUGHT 1 — Governing Law:**\nIndian Contract Act, 1872 (Section 1, 2, 17); Registration Act, 1908 (Section 17).\n"
        "**THOUGHT 2 — Mandatory Elements Checklist:**\n- Parties — Section 10 — void if missing"
    )
    assert not lda.is_degenerate_checklist(ok)


def test_runaway_think_output_falls_back_and_is_not_cached(monkeypatch):
    monkeypatch.setattr(lda, "_THINK_CACHE", {})
    import app.chatbot as cb

    monkeypatch.setattr(cb, "get_think_llm", lambda: object())
    outputs = ["Section " + ", ".join(str(n) for n in range(10, 400)), "rent checklist"]
    analyzer = lda.LegalDefectAnalyzer(llm=None)

    async def fake_invoke(prompt, stream=False, llm=None):
        return outputs.pop(0)

    analyzer._invoke_llm = fake_invoke
    rent = DocumentClassification(document_type="Rent Agreement", confidence=0.9)

    assert asyncio.run(analyzer.think(rent)) == analyzer._fallback_think(rent)
    assert lda._THINK_CACHE == {}
    assert asyncio.run(analyzer.think(rent)) == "rent checklist"


def test_think_prompt_limits_section_lists():
    from app.prompts import REACT_THINK_PROMPT

    assert "at most three key section numbers" in REACT_THINK_PROMPT
