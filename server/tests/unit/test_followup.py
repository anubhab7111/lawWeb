import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import chatbot as cb
from app.prompts import CRIME_DETAILS_PREFIX, LAWYER_DETAILS_PREFIX
from app.tools import followup


@pytest.fixture(autouse=True)
def places(monkeypatch):
    monkeypatch.setattr(followup, "_places", lambda: ["navi mumbai", "mumbai", "delhi", "pune", "kerala"])


def _state(text, messages=None, **extra):
    return {"current_input": text, "messages": messages or [], "trace": {}, **extra}


def test_lawyer_request_without_place_or_matter_needs_both():
    assert followup.missing_lawyer_details("Is there any lawyer available for a free legal aid consultation?") == [
        "location", "matter"]


def test_complete_lawyer_request_needs_nothing():
    assert followup.missing_lawyer_details("Need a criminal lawyer in Pune for my brother's bail") == []
    assert followup.find_location("I need a divorce lawyer in Navi Mumbai") == "navi mumbai"


def test_lawyer_request_asks_before_searching():
    out = asyncio.run(cb.handle_find_lawyer(_state("Can you recommend a good lawyer?")))
    assert out["response"].startswith(LAWYER_DETAILS_PREFIX)
    assert "city or district" in out["response"] and "matter" in out["response"]


def _crime_type(monkeypatch, crime):
    async def fake(_):
        return crime

    monkeypatch.setattr(cb, "classify_crime_type", fake)


def test_short_crime_report_asks_what_when_where(monkeypatch):
    _crime_type(monkeypatch, "fraud")
    out = asyncio.run(cb.handle_crime_report(_state("I was cheated")))
    assert out["response"].startswith(CRIME_DETAILS_PREFIX)


def test_short_serious_report_is_answered_with_helplines_not_asked(monkeypatch):
    _crime_type(monkeypatch, "rape")

    class Sections:
        context_text, succeeded = "", False

    async def no_sections(*args, **kwargs):
        return Sections()

    async def answer(*args, **kwargs):
        return "Guidance."

    monkeypatch.setitem(cb.RAG_TOOL_REGISTRY, "crime_sections", no_sections)
    monkeypatch.setattr(cb, "invoke_llm_safely", answer)
    out = asyncio.run(cb.handle_crime_report(_state("I got raped in market, please help")))
    assert not out["response"].startswith(CRIME_DETAILS_PREFIX)
    assert "112" in out["response"] and "181" in out["response"]


def test_reply_to_a_followup_resumes_the_same_flow_with_both_messages(monkeypatch):
    async def no_hint(_):
        return None

    monkeypatch.setattr(cb, "classify_domain_hint_embedding", no_hint)
    history = [
        {"role": "user", "content": "Can you recommend a good lawyer?"},
        {"role": "assistant", "content": LAWYER_DETAILS_PREFIX + "which city?"},
    ]
    out = asyncio.run(cb.classify_intent(_state("Pune, it's a property dispute", history)))
    assert out["intent"] == "find_lawyer" and out["followup_answered"]
    assert out["lawyer_query"] == "Can you recommend a good lawyer? Pune, it's a property dispute"


def test_ordinary_turns_are_not_treated_as_followup_replies():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]
    assert cb._resume_after_followup(history, "find me a lawyer") is None
