"""Conversation memory via LangGraph checkpoints: multi-turn context, survival
across a 'restart', account isolation, and — the design's key property — that
only plain message history is ever checkpointed. Postgres tests use the scratch
database (see conftest)."""

import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app import chatbot as cb
from app import checkpointing
from app.config import get_settings

from test_chatbot import FakeLLM, Rig, _classification, _report, _statute, run  # noqa: F401
from test_chatbot import env  # noqa: F401  (autouse fixture)


def _rig(monkeypatch, answers=("Answer one.", "Answer two.", "Answer three."), saver=None):
    rig = Rig(monkeypatch, FakeLLM(list(answers)), _classification("general_query"),
              [_statute()], [_report(1.0)])
    if saver is not None:
        rig.bot = cb.LegalChatbot(saver)
    return rig


def test_later_turns_see_earlier_ones(monkeypatch):
    rig = _rig(monkeypatch)

    async def scenario():
        await rig.bot.chat("What is anticipatory bail?", "s")
        await rig.bot.chat("And for economic offences?", "s")
        return await rig.bot.get_session_history("s")

    history = run(scenario())
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "What is anticipatory bail?"
    # the second turn's prompt carried the first exchange as context
    assert "What is anticipatory bail?" in rig.llm.prompts[1]


def test_history_survives_a_restart(monkeypatch):
    saver = InMemorySaver()  # stands in for the durable store
    first = _rig(monkeypatch, saver=saver)
    run(first.bot.chat("What is anticipatory bail?", "s"))

    second = _rig(monkeypatch, saver=saver)  # a fresh process, same store
    run(second.bot.chat("And for economic offences?", "s"))
    assert "What is anticipatory bail?" in second.llm.prompts[0]


def test_accounts_do_not_share_a_thread_even_with_the_same_session_id(monkeypatch):
    rig = _rig(monkeypatch)

    async def scenario():
        await rig.bot.chat("alice's private question", "user:1:shared-id")
        await rig.bot.chat("bob asks something", "user:2:shared-id")
        return (await rig.bot.get_session_history("user:1:shared-id"),
                await rig.bot.get_session_history("user:2:shared-id"))

    alice, bob = run(scenario())
    assert alice[0]["content"] == "alice's private question"
    assert all("alice" not in m["content"] for m in bob)
    assert "alice's private question" not in rig.llm.prompts[1]


def test_only_message_history_is_checkpointed(monkeypatch):
    """The turn graph's state (tool results holding retriever objects, sets,
    uploaded text) must never reach the checkpointer: LangGraph's serializer
    fails on arbitrary objects and flattens exceptions."""
    saver = InMemorySaver()
    rig = _rig(monkeypatch, saver=saver)
    run(rig.bot.chat("What is anticipatory bail?", "s", document_content="SECRET DOC " * 50))

    tup = saver.get_tuple({"configurable": {"thread_id": "s"}})
    channels = {k for k in tup.checkpoint["channel_values"] if not k.startswith(("branch:", "__"))}
    assert channels == {"messages"}
    # and the turn graph did not inherit the saver as a nested checkpoint
    namespaces = {ns for thread in saver.storage.values() for ns in thread}
    assert namespaces == {""}
    # nothing about the uploaded document leaked into memory
    assert "SECRET DOC" not in repr(tup.checkpoint["channel_values"])


def test_seed_session_primes_an_empty_thread_only(monkeypatch):
    rig = _rig(monkeypatch)
    seeded = [{"role": "user", "content": "old q"}, {"role": "assistant", "content": "old a"}]

    async def scenario():
        assert await rig.bot.has_session("s") is False
        await rig.bot.seed_session("s", seeded)
        assert await rig.bot.get_session_history("s") == seeded
        await rig.bot.seed_session("s", [{"role": "user", "content": "stale db read"}])
        return await rig.bot.get_session_history("s")

    assert run(scenario()) == seeded  # a live thread is never clobbered


def test_history_is_capped_at_twenty_messages(monkeypatch):
    rig = _rig(monkeypatch, answers=("A.",))

    async def scenario():
        for i in range(15):
            await rig.bot.chat(f"question {i}", "s")
        return await rig.bot.get_session_history("s")

    history = run(scenario())
    assert len(history) == 20
    assert history[-2]["content"] == "question 14"


def test_thread_id_is_stamped_with_activity_time_for_cleanup(monkeypatch):
    saver = InMemorySaver()
    rig = _rig(monkeypatch, saver=saver)
    run(rig.bot.chat("hi there friend", "s"))
    tup = saver.get_tuple({"configurable": {"thread_id": "s"}})
    assert tup.metadata["last_active"].endswith("+00:00")


# --------------------------------------------------------------------------
# Postgres (scratch database)
# --------------------------------------------------------------------------


def _db_url():
    return get_settings().database_url or os.environ["DATABASE_URL"]


def test_postgres_checkpointer_persists_across_a_restart_and_reaps_idle_threads(monkeypatch):
    async def scenario():
        pool, saver = await checkpointing.open_postgres_saver(_db_url())
        tid_active, tid_idle = f"guest:{uuid.uuid4()}", f"guest:{uuid.uuid4()}"
        try:
            first = _rig(monkeypatch, saver=saver)
            await first.bot.chat("What is anticipatory bail?", tid_active)
            await first.bot.chat("An old conversation", tid_idle)
        finally:
            await pool.close()

        # "restart": a brand new pool, saver and chatbot on the same database
        pool2, saver2 = await checkpointing.open_postgres_saver(_db_url())
        try:
            second = _rig(monkeypatch, saver=saver2)
            history = await second.bot.get_session_history(tid_active)
            assert [m["role"] for m in history] == ["user", "assistant"]
            await second.bot.chat("And for economic offences?", tid_active)
            assert "What is anticipatory bail?" in second.llm.prompts[0]

            # age one thread past retention by rewriting its activity stamp
            async with pool2.connection() as conn:
                await conn.execute(
                    "UPDATE checkpoints SET metadata = jsonb_set(metadata, '{last_active}', "
                    "to_jsonb((now() - interval '30 days')::text)) WHERE thread_id = %s",
                    (tid_idle,),
                )
            deleted = await checkpointing.delete_stale_threads(
                7, saver=saver2, pool=pool2
            )
            assert deleted >= 1
            assert await second.bot.has_session(tid_idle) is False
            assert await second.bot.has_session(tid_active) is True

            await second.bot.clear_session(tid_active)
            assert await second.bot.has_session(tid_active) is False
        finally:
            await pool2.close()

    run(scenario())


def test_init_checkpointer_falls_back_to_memory_when_postgres_is_unreachable(monkeypatch):
    monkeypatch.setattr(checkpointing, "_saver", None)
    monkeypatch.setattr(checkpointing, "_pool", None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:x@127.0.0.1:1/none")
    get_settings.cache_clear()

    async def fake_open(url):
        raise ConnectionError("no database")

    monkeypatch.setattr(checkpointing, "open_postgres_saver", fake_open)
    saver = run(checkpointing.init_checkpointer())
    assert isinstance(saver, InMemorySaver)
    assert checkpointing.uses_postgres() is False
    get_settings.cache_clear()


def test_conninfo_strips_sqlalchemy_driver_and_prisma_params():
    url = "postgresql+psycopg://u:p@localhost:5432/db?schema=public"
    assert checkpointing._conninfo(url) == "postgresql://u:p@localhost:5432/db"


def test_init_checkpointer_uses_postgres_when_reachable(monkeypatch):
    monkeypatch.setattr(checkpointing, "_saver", None)
    monkeypatch.setattr(checkpointing, "_pool", None)
    get_settings.cache_clear()

    async def scenario():
        saver = await checkpointing.init_checkpointer()
        try:
            assert checkpointing.uses_postgres() is True
            assert type(saver).__name__ == "AsyncPostgresSaver"
        finally:
            await checkpointing.close_checkpointer()

    run(scenario())
    assert checkpointing.uses_postgres() is False


def test_authenticated_thread_is_reseeded_from_the_db_transcript(monkeypatch):
    from sqlmodel import Session

    from app.db.engine import get_engine
    from app.db.init_db import init_db
    from app.db.models import User
    from app.routers import chat as chat_router

    init_db(embed=False)
    sid = str(uuid.uuid4())
    rig = _rig(monkeypatch)

    with Session(get_engine()) as s:
        owner = User(name="O", email=f"{uuid.uuid4()}@example.com", password="x")
        other = User(name="X", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add_all([owner, other])
        s.commit()
        s.refresh(owner)
        s.refresh(other)
        chat_router._persist_turn_sync(
            s, owner, sid, user_message="stored question", assistant_message="stored answer"
        )

        async def scenario():
            # another account presenting the same session id gets nothing
            await chat_router._seed_from_db_if_needed(s, rig.bot, other, sid)
            assert await rig.bot.has_session(chat_router._memory_key(other, sid)) is False
            # the owner's empty thread is primed from chat_messages...
            await chat_router._seed_from_db_if_needed(s, rig.bot, owner, sid)
            key = chat_router._memory_key(owner, sid)
            history = await rig.bot.get_session_history(key)
            # ...and a guest never touches the database
            await chat_router._seed_from_db_if_needed(s, rig.bot, None, sid)
            return history

        history = run(scenario())
    assert history == [
        {"role": "user", "content": "stored question"},
        {"role": "assistant", "content": "stored answer"},
    ]
