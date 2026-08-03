"""
In-process job scheduler (APScheduler), started from app.main's lifespan.

Uses AsyncIOScheduler (not BackgroundScheduler) because job bodies need to
await HTTP calls (case-data provider, notification dispatch) on the same
event loop uvicorn already runs — no nested event loop, no extra thread.
The jobstore is a SQLAlchemy jobstore against the app's own Postgres engine,
so scheduled jobs survive restarts; combined with add_job(..., id=...,
replace_existing=True) in register_jobs(), a restart replaces existing job
rows instead of accumulating duplicates.

Operational invariant: this only works correctly with a single uvicorn
worker (already required today for in-memory chat sessions — see
app.chatbot). Running >1 worker would double-fire every scheduled job, since
there is no distributed lock here.
"""

from functools import lru_cache

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.db.engine import get_engine


@lru_cache()
def get_scheduler() -> AsyncIOScheduler:
    jobstores = {"default": SQLAlchemyJobStore(engine=get_engine())}
    return AsyncIOScheduler(jobstores=jobstores, timezone="UTC")


def register_jobs(scheduler: AsyncIOScheduler) -> None:
    """Each feature module registers its own jobs here. Kept as a single
    import point so app.main doesn't need to know what jobs exist."""
    from app.jobs import hearing_reminders

    hearing_reminders.register(scheduler)
