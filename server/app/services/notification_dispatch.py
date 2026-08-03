"""
Shared notification spine for Hearing Reminders and Smart Notifications.
Every producer (case sync job, vault upload, cause-list refresh, calendar
reminders) calls send_notification() rather than writing to email/push/DB
directly — see the implementation plan's Phase 2/4 for the full list of
producers this generalizes to.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Session, select

from app.db.models import Notification, NotificationPreference
from app.services.email_client import send_email
from app.services.fcm_client import send_push


def _channel_enabled(prefs: Optional[NotificationPreference], channel: str, type_: str) -> bool:
    if prefs is None:
        return channel == "in_app"  # safe default with no preferences row yet
    override = prefs.type_overrides.get(type_, {})
    if channel in override:
        return bool(override[channel])
    return {
        "email": prefs.email_enabled,
        "push": prefs.push_enabled,
        "sms": prefs.sms_enabled,
        "in_app": True,
    }.get(channel, False)


def send_notification(
    session: Session,
    user_id: str,
    type_: str,
    title: str,
    body: str,
    related_case_id: Optional[str] = None,
    related_case_event_id: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> List[Notification]:
    """Writes one `notifications` row per channel, then dispatches
    synchronously. Called from request handlers and scheduled jobs alike —
    kept synchronous (no async I/O) since email/FCM here are best-effort and
    a slow SMTP call blocking a background job tick is an acceptable
    tradeoff at this scale (see plan's reliability risk note)."""
    from app.db.models import User

    user = session.get(User, user_id)
    if user is None:
        return []

    prefs = session.exec(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    ).first()

    channels = channels or ["in_app", "email"]
    created: List[Notification] = []

    for channel in channels:
        if not _channel_enabled(prefs, channel, type_):
            continue

        notification = Notification(
            user_id=user_id,
            type=type_,
            title=title,
            body=body,
            related_case_id=related_case_id,
            related_case_event_id=related_case_event_id,
            channel=channel,
            status="pending",
        )
        session.add(notification)
        session.flush()

        sent = False
        if channel == "in_app":
            sent = True  # the feed row itself is the delivery
        elif channel == "email":
            sent = send_email(user.email, title, body)
        elif channel == "push" and prefs and prefs.fcm_token:
            sent = send_push(prefs.fcm_token, title, body)

        notification.status = "sent" if sent else "failed"
        if sent:
            notification.sent_at = datetime.now(timezone.utc)
        created.append(notification)

    session.commit()
    return created


async def send_notification_async(**kwargs) -> List[Notification]:
    """Awaitable wrapper around send_notification() for callers running on
    the shared FastAPI/APScheduler event loop (app/jobs/_case_sync.py).
    send_email() is a blocking smtplib call with a 10s timeout — calling
    send_notification() directly from an `async def` job body would freeze
    every concurrent HTTP request for up to that long once SMTP is actually
    configured. Sync request handlers (e.g. app/routers/vault.py's
    share_document) can keep calling send_notification() directly — FastAPI
    already runs those in its own thread pool.

    Opens its OWN Session inside the worker thread rather than reusing the
    caller's: a SQLAlchemy Session isn't thread-safe, and the caller's session
    lives on the event-loop thread. Notifications are best-effort and
    self-committing, so a separate transaction is correct here."""
    from app.db.engine import get_engine

    def _run() -> List[Notification]:
        with Session(get_engine()) as session:
            return send_notification(session, **kwargs)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
