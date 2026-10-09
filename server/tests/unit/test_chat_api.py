"""HTTP-level checks for the chat endpoints: rate limiting, the busy (503)
mapping and SSE relaying of the workflow's progress events. The chatbot itself
is faked; guests never touch the database."""

import json

import pytest
from fastapi.testclient import TestClient

from app.chatbot import ChatBusyError
from app.config import get_settings
from app.deps import rate_limit


class FakeBot:
    def __init__(self, chat_exc=None, events=None, stream_exc=None):
        self.chat_exc = chat_exc
        self.events = events or []
        self.stream_exc = stream_exc

    def has_session(self, session_id):
        return True

    async def chat(self, message, session_id, **kwargs):
        if self.chat_exc:
            raise self.chat_exc
        return {"response": "ok", "response_en": "ok", "query_en": message,
                "language": "en", "intent": "general_query"}

    async def stream_chat(self, message, session_id, **kwargs):
        self.stream_calls = getattr(self, "stream_calls", []) + [(message, kwargs)]
        if self.stream_exc:
            raise self.stream_exc
        for event in self.events:
            yield event


@pytest.fixture()
def client_for(monkeypatch):
    from app.main import app
    from app.routers import chat as chat_router

    rate_limit.reset_rate_limits()
    get_settings.cache_clear()

    def make(bot):
        monkeypatch.setattr(chat_router, "get_chatbot", lambda: bot)
        return TestClient(app)  # not a context manager: no lifespan/warmup

    yield make
    rate_limit.reset_rate_limits()
    get_settings.cache_clear()


def test_chat_endpoint_is_rate_limited_per_caller(client_for, monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    client = client_for(FakeBot())
    body = {"message": "hello"}
    assert client.post("/api/chat", json=body).status_code == 200
    assert client.post("/api/chat", json=body).status_code == 200
    third = client.post("/api/chat", json=body)
    assert third.status_code == 429
    assert int(third.headers["Retry-After"]) >= 1


def test_rate_limit_can_be_disabled(client_for, monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "0")
    get_settings.cache_clear()
    client = client_for(FakeBot())
    assert all(client.post("/api/chat", json={"message": "hi"}).status_code == 200
               for _ in range(5))


def test_busy_chatbot_returns_503_not_500(client_for):
    client = client_for(FakeBot(chat_exc=ChatBusyError("The assistant is busy right now.")))
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 503
    assert "busy" in resp.json()["detail"]


def _sse(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_stream_relays_progress_events_and_done(client_for):
    events = [
        {"type": "status", "stage": "retrieval", "label": "Searching…"},
        {"type": "token", "content": "Draft "},
        {"type": "reset"},
        {"type": "token", "content": "Final"},
        {"type": "replace", "content": "Final answer"},
        {"type": "done", "session_id": "s", "intent": "general_query",
         "response": "Final answer", "response_en": "Final answer",
         "query_en": "q", "language": "en", "trace": {"routing": {"intent": "general_query"}}},
    ]
    resp = client_for(FakeBot(events=events)).post("/api/chat/stream", json={"message": "q"})
    assert resp.status_code == 200
    got = _sse(resp)
    assert [e["type"] for e in got] == ["status", "token", "reset", "token", "replace", "done"]
    assert got[-1]["trace"]["routing"]["intent"] == "general_query"


def test_stream_busy_becomes_an_error_event_with_a_useful_message(client_for):
    resp = client_for(FakeBot(stream_exc=ChatBusyError("The assistant is busy right now."))
                      ).post("/api/chat/stream", json={"message": "q"})
    got = _sse(resp)
    assert got == [{"type": "error", "content": "The assistant is busy right now."}]


def test_validate_document_stream_sends_the_document_with_validation_intent(client_for):
    events = [
        {"type": "status", "stage": "classify", "label": "Reading the document…"},
        {"type": "token", "content": "## 📄 Document Classification"},
        {"type": "replace", "content": "Full report"},
        {"type": "done", "session_id": "s", "intent": "document_analysis",
         "response": "Full report", "response_en": "Full report", "query_en": "q", "language": "en"},
    ]
    bot = FakeBot(events=events)
    text = "This rent agreement is made between the landlord and the tenant."
    resp = client_for(bot).post("/api/chat/validate-document/stream",
                                data={"document_text": text, "message": "Is this valid?"})
    assert resp.status_code == 200
    assert [e["type"] for e in _sse(resp)] == ["status", "token", "replace", "done"]
    message, kwargs = bot.stream_calls[0]
    assert "validate" in message.lower()
    assert kwargs["document_content"] == text


def test_validate_document_stream_rejects_an_empty_document(client_for):
    resp = client_for(FakeBot()).post("/api/chat/validate-document/stream", data={"document_text": " "})
    assert resp.status_code == 422


def test_app_lifespan_initialises_the_checkpointer_and_registers_the_cleanup_job(monkeypatch):
    """Boot the real lifespan (warmup stubbed: it loads GPU models) against the
    scratch database."""
    from app import checkpointing, main
    from app.scheduler import get_scheduler

    async def no_warmup():
        return None

    monkeypatch.setattr(main, "_warmup", no_warmup)
    monkeypatch.setattr(checkpointing, "_saver", None)
    monkeypatch.setattr(checkpointing, "_pool", None)
    get_settings.cache_clear()

    with TestClient(main.app) as client:
        assert checkpointing.uses_postgres() is True
        assert get_scheduler().get_job("cleanup_stale_chat_threads") is not None
        assert client.get("/api/chat/health").status_code == 200
    assert checkpointing.uses_postgres() is False  # pool closed on shutdown
    get_settings.cache_clear()
