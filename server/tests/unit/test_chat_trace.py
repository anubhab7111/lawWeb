"""chat_messages.trace: migration, persistence (assistant rows only) and the
history response. Runs against the scratch database (see conftest)."""

import asyncio
import uuid

from sqlalchemy import text
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.init_db import init_db
from app.db.migrations import ensure_chat_messages_trace_column
from app.db.models import ChatMessage, MessageRole, User
from app.routers import chat as chat_router

TRACE = {
    "routing": {"intent": "general_query", "secondary": ["find_lawyer"]},
    "retrieval": {"grade": "good", "sections": {"438", "437"}},  # a set: JSONB would reject it
    "grounding": {"verified": True, "score": 0.9},
}


def _user(s):
    u = User(name="T", email=f"{uuid.uuid4()}@example.com", password="x")
    s.add(u)
    s.commit()
    s.refresh(u)
    return u


def test_migration_is_idempotent_and_creates_a_nullable_jsonb_column():
    init_db(embed=False)
    engine = get_engine()
    ensure_chat_messages_trace_column(engine)
    ensure_chat_messages_trace_column(engine)  # second run must be a no-op
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'chat_messages' AND column_name = 'trace'"
        )).one()
    assert row == ("jsonb", "YES")


def test_trace_is_stored_on_the_assistant_row_only():
    init_db(embed=False)
    sid = str(uuid.uuid4())
    with Session(get_engine()) as s:
        user = _user(s)
        chat_router._persist_turn_sync(
            s, user, sid,
            user_message="Is bail available?", assistant_message="Yes, under s.437.",
            trace=TRACE,
        )
        rows = s.exec(
            select(ChatMessage).where(ChatMessage.session_id == sid).order_by(ChatMessage.created_at)
        ).all()
    assert [r.role for r in rows] == [MessageRole.user, MessageRole.assistant]
    assert rows[0].trace is None
    stored = rows[1].trace
    assert stored["routing"]["intent"] == "general_query"
    assert stored["grounding"]["score"] == 0.9
    assert sorted(stored["retrieval"]["sections"]) == ["437", "438"]  # set became a list


def test_turn_without_a_trace_still_persists():
    init_db(embed=False)
    sid = str(uuid.uuid4())
    with Session(get_engine()) as s:
        user = _user(s)
        chat_router._persist_turn_sync(
            s, user, sid, user_message="hi", assistant_message="hello"
        )
        rows = s.exec(select(ChatMessage).where(ChatMessage.session_id == sid)).all()
    assert len(rows) == 2 and all(r.trace is None for r in rows)


def test_persist_chat_result_forwards_the_trace(monkeypatch):
    captured = {}

    async def fake_persist(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(chat_router, "_persist_turn", fake_persist)
    result = {"language": "en", "query_en": "q", "response_en": "a", "response": "a", "trace": TRACE}
    asyncio.run(chat_router._persist_chat_result(None, "user", "sid", "q", result))
    assert captured["trace"] is TRACE


def test_history_endpoint_returns_the_trace():
    init_db(embed=False)
    sid = str(uuid.uuid4())
    with Session(get_engine()) as s:
        user = _user(s)
        chat_router._persist_turn_sync(
            s, user, sid, user_message="q", assistant_message="a", trace=TRACE
        )
        out = chat_router.get_session_history(sid, user=user, session=s)
    by_role = {m["role"]: m for m in out["messages"]}
    assert by_role["user"]["trace"] is None
    assert by_role["assistant"]["trace"]["grounding"]["verified"] is True
