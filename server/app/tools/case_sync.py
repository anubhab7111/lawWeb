"""
Shared case-sync logic used by both the on-demand routes in
app/routers/cases.py (save / manual re-sync) and the scheduled job in
app/jobs/_case_sync.py. Kept out of the router module so the scheduler
doesn't need to import FastAPI router internals.
"""

from datetime import datetime, timezone
from typing import List

from sqlmodel import Session, select

from app.db.models import CalendarEvent, CaseAiSummary, CaseEvent, SavedCase
from app.tools.case_data_provider import CaseDataProviderError, get_case_data_provider
from app.tools.case_summarizer import summarize_case_event


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

    existing = session.exec(
        select(CaseEvent).where(CaseEvent.saved_case_id == case.id)
    ).all()
    existing_keys = {(e.event_type, e.title, e.event_date) for e in existing}

    new_events: List[CaseEvent] = []
    for record in history:
        key = (record.event_type, record.title, record.event_date)
        if key in existing_keys:
            continue
        event = CaseEvent(
            saved_case_id=case.id,
            event_type=record.event_type,
            event_date=record.event_date,
            title=record.title,
            detail=record.detail,
            source_url=record.source_url,
            raw_payload=record.raw,
        )
        session.add(event)
        session.flush()
        new_events.append(event)

        # Personal Legal Calendar (Phase 4.1): a new hearing case_event
        # auto-populates the calendar, same as a confirmed booking does in
        # app/routers/bookings.py. Keyed by related_case_event_id (unique)
        # so a future update-in-place on the source event could re-sync
        # this row instead of creating a duplicate.
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
            summary_text = await summarize_case_event(
                case.title or case.cnr, record.title, record.detail
            )
            if summary_text:
                session.add(
                    CaseAiSummary(
                        saved_case_id=case.id,
                        summary_text=summary_text,
                        source_event_id=event.id,
                    )
                )

    case.last_synced_at = datetime.now(timezone.utc)
    session.add(case)
    session.commit()
    return new_events
