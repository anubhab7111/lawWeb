"""
Shared notification spine for Hearing Reminders and Smart Notifications.
Every producer (case sync job, vault upload, cause-list refresh, calendar
reminders) calls send_notification() rather than writing to email/push/DB
directly — see the implementation plan's Phase 2/4 for the full list of
producers this generalizes to.
"""

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
