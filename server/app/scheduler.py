"""In-process job scheduler (APScheduler), started from app.main's lifespan."""

from functools import lru_cache

from app.db.engine import get_engine
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler


@lru_cache()
def get_scheduler() -> AsyncIOScheduler:
    jobstores = {"default": SQLAlchemyJobStore(engine=get_engine())}
    return AsyncIOScheduler(jobstores=jobstores, timezone="UTC")


def register_jobs(scheduler: AsyncIOScheduler) -> None:
    """Each feature module registers its own jobs here."""
    from app.jobs import calendar_sync, chat_threads, hearing_reminders

    hearing_reminders.register(scheduler)
    calendar_sync.register(scheduler)
    chat_threads.register(scheduler)
