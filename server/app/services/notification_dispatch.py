"""Shared notification spine for Hearing Reminders and Smart Notifications."""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from app.config import get_settings
from app.db.models import Notification, NotificationPreference
from app.services.email_client import send_email
from app.services.fcm_client import PushTokenInvalid, push_configured, send_push
from sqlmodel import Session, select


def _channel_enabled(
    prefs: Optional[NotificationPreference], channel: str, type_: str
) -> bool:
    if prefs is None:
        return channel == "in_app"
    override = prefs.type_overrides.get(type_, {})
    if channel in override:
        return bool(override[channel])
    return {
        "email": prefs.email_enabled,
        "push": prefs.push_enabled,
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
    """Writes one `notifications` row per channel, then dispatches synchronously."""
    from app.db.models import User

    user = session.get(User, user_id)
    if user is None:
        return []
    settings = get_settings()

    prefs = session.exec(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    ).first()

    channels = list(channels or ["in_app", "email"])
    if "push" not in channels and prefs and prefs.push_enabled:
        channels.append("push")
    created: List[Notification] = []

    for channel in channels:
        if not _channel_enabled(prefs, channel, type_):
            continue
        if channel == "email" and not settings.smtp_host:
            print(
                f"[Notify] SMTP not configured — skipping email '{title}' for {user.email}"
            )
            continue
        if channel == "push" and not (prefs and prefs.fcm_token and push_configured()):
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
            sent = True
        elif channel == "email":
            sent = send_email(user.email, title, body)
        elif channel == "push" and prefs and prefs.fcm_token:
            try:
                sent = send_push(prefs.fcm_token, title, body)
            except PushTokenInvalid:
                prefs.fcm_token = None
                session.add(prefs)
                sent = False

        notification.status = "sent" if sent else "failed"
        if sent:
            notification.sent_at = datetime.now(timezone.utc)
        created.append(notification)

    session.commit()
    return created


async def send_notification_async(**kwargs) -> List[Notification]:
    """Awaitable wrapper around send_notification() for callers running on the shared FastAPI/APScheduler event loop (app/jobs/_case_sync.py)."""
    from app.db.engine import get_engine

    def _run() -> List[Notification]:
        with Session(get_engine()) as session:
            return send_notification(session, **kwargs)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
