"""Conversation-memory checkpointer for the chatbot.

The live conversation window (last 20 English messages per session) is kept in
LangGraph checkpoints keyed by thread_id == the chatbot's memory key
(`user:<id>:<session>` / `guest:<session>`), so it survives restarts and does
not depend on which worker served the previous turn. `chat_messages` remains the
durable, user-visible transcript (sidebar, history endpoint); a thread with no
checkpoint is re-seeded from it (see app.routers.chat._seed_from_db_if_needed).

Postgres when reachable; if it is not, chat degrades to an in-process
InMemorySaver (with the old TTL/size eviction in LegalChatbot) rather than
failing — guest chat never needed the database before, and shouldn't now.
"""

import logging
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings

logger = logging.getLogger(__name__)

_saver: Optional[BaseCheckpointSaver] = None
_pool: Any = None


def _conninfo(database_url: str) -> str:
    """libpq connection string from the app's DATABASE_URL. Prisma-style URLs
    carry a ?schema=... parameter that libpq rejects."""
    return database_url.split("?", 1)[0].replace("postgresql+psycopg://", "postgresql://", 1)


async def open_postgres_saver(database_url: str) -> tuple:
    """Open a pool and an AsyncPostgresSaver on it, creating the checkpoint
    tables if needed (idempotent). Returns (pool, saver). The autocommit /
    prepare_threshold / dict_row settings are what the saver requires."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=_conninfo(database_url),
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open(wait=True, timeout=10)
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
    except Exception:
        await pool.close()
        raise
    return pool, saver


async def init_checkpointer() -> BaseCheckpointSaver:
    """Called once from the app lifespan, before the chatbot is first built."""
    global _saver, _pool
    if _saver is not None:
        return _saver
    url = get_settings().database_url
    if url:
        try:
            _pool, _saver = await open_postgres_saver(url)
            logger.info("Chat memory: Postgres checkpointer ready")
            return _saver
        except Exception:
            logger.exception(
                "Chat memory: Postgres checkpointer unavailable — falling back to "
                "in-memory sessions (lost on restart)"
            )
    _saver = InMemorySaver()
    return _saver


def get_checkpointer() -> BaseCheckpointSaver:
    """The saver for the running app; an InMemorySaver if init never ran
    (scripts, tests)."""
    global _saver
    if _saver is None:
        _saver = InMemorySaver()
    return _saver


def uses_postgres() -> bool:
    return _pool is not None


async def close_checkpointer() -> None:
    global _saver, _pool
    if _pool is not None:
        await _pool.close()
    _saver, _pool = None, None


async def delete_stale_threads(
    max_age_days: int,
    limit: int = 200,
    saver: Optional[BaseCheckpointSaver] = None,
    pool: Any = None,
) -> int:
    """Delete conversation threads idle for more than max_age_days. Every run
    stamps `last_active` into the checkpoint metadata (see LegalChatbot._config),
    which is what idleness is judged by. Returns how many threads were deleted;
    0 without touching anything when the app is on the in-memory fallback."""
    saver = saver or _saver
    pool = pool or _pool
    if saver is None or pool is None:
        return 0
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT thread_id FROM checkpoints
            GROUP BY thread_id
            HAVING max((metadata->>'last_active')::timestamptz)
                   < now() - make_interval(days => %s)
            LIMIT %s
            """,
            (max_age_days, limit),
        )
        stale = [row["thread_id"] for row in await cur.fetchall()]
    for thread_id in stale:
        await saver.adelete_thread(thread_id)
    return len(stale)
