import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools import legal_defect_analyzer as lda
from app.tools.document_classifier import DocumentClassification


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
    analyzer, calls = _analyzer(["rent checklist", "fir checklist"])
    rent = DocumentClassification(document_type="Rent Agreement", confidence=0.9, jurisdiction_hints=["Delhi"])
    fir = DocumentClassification(document_type="FIR", confidence=0.9)

    assert asyncio.run(analyzer.think(rent)) == "rent checklist"
    assert asyncio.run(analyzer.think(rent)) == "rent checklist"
    assert asyncio.run(analyzer.think(fir)) == "fir checklist"
    assert calls == ["Rent Agreement", "FIR"]


def test_think_fallback_is_not_cached(monkeypatch):
    monkeypatch.setattr(lda, "_THINK_CACHE", {})
    analyzer, calls = _analyzer([RuntimeError("timeout"), "fir checklist"])
    fir = DocumentClassification(document_type="FIR", confidence=0.9)

    assert asyncio.run(analyzer.think(fir)) == analyzer._fallback_think(fir)
    assert asyncio.run(analyzer.think(fir)) == "fir checklist"
    assert len(calls) == 2
