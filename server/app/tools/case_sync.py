"""
Shared case-sync logic used by both the on-demand routes in
app/routers/cases.py (save / manual re-sync) and the scheduled job in
app/jobs/_case_sync.py. Kept out of the router module so the scheduler
doesn't need to import FastAPI router internals.
"""

from datetime import datetime, timezone
from typing import List

from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session, select

from app.db.models import CalendarEvent, CaseAiSummary, CaseEvent, SavedCase
from app.tools.case_data_provider import CaseDataProviderError, get_case_data_provider
from app.tools.case_summarizer import summarize_case_event


def _event_key(event_type, title, event_date, source_id):
    if source_id:
        return ("id", source_id)
    minute = event_date.replace(second=0, microsecond=0) if event_date else None
    return (event_type, title, minute)


async def sync_case_events(session: Session, case: SavedCase) -> List[CaseEvent]:
    """Fetches history from the provider, inserts any new events, generates
    an AI summary for new orders, and returns the newly-inserted events (so
    callers — e.g. the hearing-reminders job — can act on new orders)."""
    provider = get_case_data_provider()
    if not case.cnr:
        return []

    try:
        history = await provider.fetch_case_history(case.cnr)
    except CaseDataProviderError as e:
        print(f"[CaseSync] sync failed for case {case.id}: {e}")
        return []

    # The session is plain blocking SQLAlchemy, so every DB step runs in the
    # thread pool (one step at a time); only the LLM summary is awaited here.
    new_events, order_jobs = await run_in_threadpool(_insert_new_events, session, case, history)

    for event_id, record in order_jobs:
        summary_text = await summarize_case_event(
            case.title or case.cnr, record.title, record.detail
        )
        if summary_text:
            await run_in_threadpool(_add_summary, session, case.id, event_id, summary_text)

    await run_in_threadpool(_finish_sync, session, case, new_events)
    return new_events


def _insert_new_events(session: Session, case: SavedCase, history):
    existing = session.exec(
        select(CaseEvent).where(CaseEvent.saved_case_id == case.id)
    ).all()
    existing_keys = {
        _event_key(e.event_type, e.title, e.event_date, (e.raw_payload or {}).get("source_id"))
        for e in existing
    }
    legacy_keys = {
        ("tt", e.event_type, e.title) for e in existing if not (e.raw_payload or {}).get("source_id")
    }
    new_events: List[CaseEvent] = []
    order_jobs = []
    for record in history:
        key = _event_key(record.event_type, record.title, record.event_date, record.source_id)
        if key in existing_keys or ("tt", record.event_type, record.title) in legacy_keys:
            continue
        existing_keys.add(key)
        event = CaseEvent(
            saved_case_id=case.id,
            event_type=record.event_type,
            event_date=record.event_date,
            title=record.title,
            detail=record.detail,
            source_url=record.source_url,
            raw_payload={**record.raw, **({"source_id": record.source_id} if record.source_id else {})},
        )
        session.add(event)
        session.flush()
        new_events.append(event)

        # A new hearing auto-populates the Personal Legal Calendar, keyed by
        # related_case_event_id (unique) so re-syncs can't duplicate it.
        if record.event_type == "hearing" and event.event_date:
            session.add(
                CalendarEvent(
                    user_id=case.user_id,
                    title=f"Hearing: {case.title or case.cnr}",
                    event_type="hearing",
                    start_at=event.event_date,
                    related_case_id=case.id,
                    related_case_event_id=event.id,
                )
            )
        if record.event_type == "order":
            order_jobs.append((event.id, record))
    return new_events, order_jobs


def _add_summary(session: Session, case_id: str, event_id: str, text: str) -> None:
    session.add(CaseAiSummary(saved_case_id=case_id, summary_text=text, source_event_id=event_id))
    session.flush()


def _finish_sync(session: Session, case: SavedCase, events: List[CaseEvent]) -> None:
    case.last_synced_at = datetime.now(timezone.utc)
    session.add(case)
    session.commit()
    # Load what callers read next (event fields) while still off the event loop.
    session.refresh(case)
    for event in events:
        session.refresh(event)
