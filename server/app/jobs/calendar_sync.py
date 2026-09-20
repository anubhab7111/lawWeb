"""Hourly push of new LawWeb events to every connected external calendar."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import CalendarConnection
from app.services import calendar_oauth
from app.services.calendar_sync import sync_connection


def register(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        run_calendar_sync, "interval", hours=1, id="calendar_sync", replace_existing=True
    )


def _sync_all() -> None:
    with Session(get_engine()) as session:
        for connection in session.exec(select(CalendarConnection)).all():
            if not calendar_oauth.is_configured(connection.provider):
                continue
            try:
                sync_connection(session, connection)
            except Exception as e:
                session.rollback()
                print(f"[CalendarSync] sync failed for {connection.user_id}/{connection.provider}: {e}")


async def run_calendar_sync() -> None:
    import asyncio

    await asyncio.get_event_loop().run_in_executor(None, _sync_all)
