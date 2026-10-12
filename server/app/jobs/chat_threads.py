"""Conversation-thread retention."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


def register(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        cleanup_stale_chat_threads,
        "interval",
        hours=6,
        id="cleanup_stale_chat_threads",
        replace_existing=True,
    )


async def cleanup_stale_chat_threads() -> None:
    from app.checkpointing import delete_stale_threads
    from app.config import get_settings

    deleted = await delete_stale_threads(get_settings().chat_thread_retention_days)
    if deleted:
        logger.info("Deleted %s idle chat threads", deleted)
