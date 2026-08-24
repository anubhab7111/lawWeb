"""
Notification feed + preferences. This is also the router Smart Notifications
(Phase 4) reuses unchanged — every new producer just calls
app.services.notification_dispatch.send_notification(), so no new router
was needed there.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import Notification, NotificationPreference, User
from app.deps.auth import get_current_user
from app.deps.errors import MessageHTTPException

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class PreferencesRequest(BaseModel):
    emailEnabled: bool = True
    pushEnabled: bool = False
    smsEnabled: bool = False
    fcmToken: str | None = None
    phoneNumber: str | None = None


@router.get("")
def list_notifications(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    notifications = session.exec(
        select(Notification)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    return [n.to_dict() for n in notifications]


@router.patch("/{notification_id}/read")
def mark_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    notification = session.get(Notification, notification_id)
    if notification is None or notification.user_id != current_user.id:
        raise MessageHTTPException(status_code=404, detail="Notification not found")
    notification.read_at = datetime.now(timezone.utc)
    session.add(notification)
    session.commit()
    return notification.to_dict()


@router.get("/preferences")
def get_preferences(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    prefs = session.exec(
        select(NotificationPreference).where(NotificationPreference.user_id == current_user.id)
    ).first()
    if prefs is None:
        prefs = NotificationPreference(user_id=current_user.id)
        session.add(prefs)
        session.commit()
        session.refresh(prefs)
    return prefs.to_dict()


@router.put("/preferences")
def update_preferences(
    body: PreferencesRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    prefs = session.exec(
        select(NotificationPreference).where(NotificationPreference.user_id == current_user.id)
    ).first()
    if prefs is None:
        prefs = NotificationPreference(user_id=current_user.id)

    prefs.email_enabled = body.emailEnabled
    prefs.push_enabled = body.pushEnabled
    prefs.sms_enabled = body.smsEnabled
    if body.fcmToken is not None:
        prefs.fcm_token = body.fcmToken
    if body.phoneNumber is not None:
        prefs.phone_number = body.phoneNumber

    session.add(prefs)
    session.commit()
    session.refresh(prefs)
    return prefs.to_dict()
