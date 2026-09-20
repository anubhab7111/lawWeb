"""
Push a user's LawWeb calendar events to a connected Google / Outlook calendar.
Events already carrying that provider's id are skipped (LawWeb events can't be
edited after creation, so there is nothing to update).
"""

from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlmodel import Session, select

from app.db.models import CalendarConnection, CalendarEvent
from app.services import calendar_oauth

_ID_FIELD = {"google": "google_calendar_event_id", "outlook": "outlook_event_id"}


def provider_token(connection: CalendarConnection) -> str:
    return calendar_oauth.access_token(
        connection.provider, calendar_oauth.decrypt_token(connection.refresh_token_enc)
    )


def sync_connection(session: Session, connection: CalendarConnection) -> Dict[str, int]:
    token = provider_token(connection)
    field = _ID_FIELD[connection.provider]
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    events = session.exec(
        select(CalendarEvent).where(
            CalendarEvent.user_id == connection.user_id, CalendarEvent.start_at >= cutoff
        )
    ).all()

    pushed = failed = 0
    for event in events:
        if getattr(event, field):
            continue
        try:
            remote_id = calendar_oauth.push_event(
                connection.provider, token, event.title, event.start_at, event.end_at, None
            )
        except calendar_oauth.CalendarProviderError as e:
            print(f"[CalendarSync] {e}")
            failed += 1
            continue
        setattr(event, field, remote_id)
        session.add(event)
        pushed += 1

    connection.last_synced_at = datetime.now(timezone.utc)
    session.add(connection)
    session.commit()
    return {"pushed": pushed, "failed": failed}
