"""
Job bodies for app/jobs/hearing_reminders.py. Split out so the scheduler
module only imports plain functions, not FastAPI router internals.
"""

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import CaseEvent, Notification, SavedCase
from app.services.notification_dispatch import send_notification_async
from app.tools.case_sync import sync_case_events

_STALE_AFTER = timedelta(hours=6)

# (window label, timedelta before the hearing, notification type)
_REMINDER_WINDOWS = [
    ("7d", timedelta(days=7), "hearing_reminder_7d"),
    ("1d", timedelta(days=1), "hearing_reminder_1d"),
    ("2h", timedelta(hours=2), "hearing_reminder_2h"),
]


async def run_poll_cause_lists_and_sync_cases() -> None:
    with Session(get_engine()) as session:
        cutoff = datetime.now(timezone.utc) - _STALE_AFTER
        stale_cases = session.exec(
            select(SavedCase).where(
                (SavedCase.last_synced_at.is_(None)) | (SavedCase.last_synced_at < cutoff)
            )
        ).all()

        for case in stale_cases:
            try:
                new_events = await sync_case_events(session, case)
            except Exception as e:
                print(f"[HearingReminders] sync failed for case {case.id}: {e}")
                continue

            for event in new_events:
                if event.event_type == "order":
                    await send_notification_async(
                        user_id=case.user_id,
                        type_="new_order",
                        title=f"New order in {case.title or case.cnr}",
                        body=event.title,
                        related_case_id=case.id,
                        channels=["in_app", "email"],
                    )


async def run_send_hearing_reminders() -> None:
    with Session(get_engine()) as session:
        now = datetime.now(timezone.utc)

        for _label, delta, notif_type in _REMINDER_WINDOWS:
            target = now + delta
            # Hearings falling within a 15-minute tick window of the target
            # offset — matches the job's own 15-minute interval so no hearing
            # is skipped between two consecutive runs.
            window_start = target - timedelta(minutes=15)
            window_end = target

            upcoming = session.exec(
                select(CaseEvent).where(
                    CaseEvent.event_type == "hearing",
                    CaseEvent.event_date >= window_start,
                    CaseEvent.event_date <= window_end,
                )
            ).all()

            for event in upcoming:
                # Dedup per hearing EVENT (not per case): a case with multiple
                # hearings must get a reminder for each, so key on the event id.
                already_sent = session.exec(
                    select(Notification).where(
                        Notification.related_case_event_id == event.id,
                        Notification.type == notif_type,
                    )
                ).first()
                if already_sent is not None:
                    continue

                case = session.get(SavedCase, event.saved_case_id)
                if case is None:
                    continue

                await send_notification_async(
                    user_id=case.user_id,
                    type_=notif_type,
                    title=f"Upcoming hearing: {case.title or case.cnr}",
                    body=f"Hearing scheduled for {event.event_date.isoformat() if event.event_date else 'soon'}",
                    related_case_id=case.id,
                    related_case_event_id=event.id,
                    channels=["in_app", "email"],
                )
