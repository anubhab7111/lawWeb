import asyncio

import pytest

from app.tools import legal_retrieval as lr


@pytest.fixture(autouse=True)
def no_classifier(monkeypatch):
    """Precedent-path tests run as if the statute classifier weren't exported."""
    monkeypatch.setattr(lr, "_get_classifier", lambda: None)

NARRATIVE = " ".join(
    [
        "The prosecution case in brief is that the deceased was married to the accused five years ago.",
        "After the marriage she was treated with cruelty by her husband and in laws over dowry demands.",
        "On the night of the incident the neighbours heard shouts from the matrimonial home.",
        "The deceased was found with burn injuries and was taken to the hospital by the informant.",
        "Before she died she made a statement to the doctor implicating her husband and mother in law.",
        "The trial court convicted the accused and the High Court affirmed the conviction on appeal.",
        "The accused now contends before this Court that the evidence on record does not support the finding.",
    ]
)


def test_short_questions_are_not_fact_patterns():
    assert not lr.looks_like_fact_pattern("What is the punishment for murder under the IPC?")
    assert not lr.looks_like_fact_pattern("Section 302. " * 10)


def test_long_narratives_are_fact_patterns():
    assert lr.looks_like_fact_pattern(NARRATIVE)
    assert len(lr.split_sentences(NARRATIVE)) == 7


def test_precedent_pins_are_empty_for_questions_and_when_index_missing(monkeypatch):
    assert asyncio.run(lr.precedent_pins("What is bail?")) == []

    class Missing:
        available = False

    monkeypatch.setattr("app.tools.precedent_rag.get_precedent_index", lambda: Missing())
    assert asyncio.run(lr.precedent_pins(NARRATIVE)) == []


def test_precedent_pins_map_votes_to_ipc_and_cap_the_count(monkeypatch):
    class Ready:
        available = True

    async def fake_votes(sentences, **kwargs):
        assert len(sentences) == 7
        return [("304B", 0.9), ("498A", 0.8), ("34", 0.5), ("306", 0.4), ("302", 0.1)]

    monkeypatch.setattr("app.tools.precedent_rag.get_precedent_index", lambda: Ready())
    monkeypatch.setattr("app.tools.precedent_rag.retrieve_precedent_sections", fake_votes)
    pins = asyncio.run(lr.precedent_pins(NARRATIVE, top=3))
    assert pins == [("Indian Penal Code", "304B"), ("Indian Penal Code", "498A"), ("Indian Penal Code", "34")]


def test_classifier_decides_when_exported(monkeypatch):
    class FakeClassifier:
        def predict(self, sentences):
            return {"Section 304B": 0.9, "Section 482": 0.7, "Section 4": 0.8, "Section 313": 0.45, "Section 302": 0.1}

        def decide(self, probs, max_labels=6):
            return [(l, p) for l, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= 0.4][:max_labels]

    monkeypatch.setattr(lr, "_get_classifier", lambda: FakeClassifier())
    pins = asyncio.run(lr.precedent_pins(NARRATIVE))
    # 482 is pinned as the CrPC provision judgments mean by it, not IPC 482; the
    # jurisdiction label (4) and a label below 0.5 (313) are not pinned in chat
    assert pins == [("Indian Penal Code", "304B"), ("Code of Criminal Procedure", "482")]


def test_label_to_pin_uses_the_act_judgments_cite():
    assert lr.label_to_pin("Section 302") == ("Indian Penal Code", "302")
    assert lr.label_to_pin("Section 438") == ("Code of Criminal Procedure", "438")
    assert lr.label_to_pin("Section 294(b)") == ("Indian Penal Code", "294")


def test_a_failing_precedent_lookup_never_breaks_retrieval(monkeypatch):
    class Ready:
        available = True

    async def boom(sentences, **kwargs):
        raise RuntimeError("index corrupt")

    monkeypatch.setattr("app.tools.precedent_rag.get_precedent_index", lambda: Ready())
    monkeypatch.setattr("app.tools.precedent_rag.retrieve_precedent_sections", boom)
    assert asyncio.run(lr.precedent_pins(NARRATIVE)) == []
