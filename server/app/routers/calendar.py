"""
Personal Legal Calendar — mostly auto-populated (hearings from My Cases sync,
lawyer meetings from confirmed bookings — see the additive hook in
app/routers/bookings.py), plus manual custom entries. Events can be pushed to a
connected Google or Outlook calendar (see app/services/calendar_oauth.py).
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db.engine import get_session
from app.config import get_settings
from app.db.models import CalendarConnection, CalendarEvent, SavedCase, User
from app.security import jwt_secret
from app.services import calendar_oauth, calendar_sync
from app.deps.auth import get_current_user
from app.deps.errors import MessageHTTPException

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


class CreateEventRequest(BaseModel):
    title: str
    event_type: str = Field(default="custom", alias="eventType")
    start_at: datetime = Field(..., alias="startAt")
    end_at: Optional[datetime] = Field(default=None, alias="endAt")
    related_case_id: Optional[str] = Field(default=None, alias="relatedCaseId")

    class Config:
        populate_by_name = True


@router.get("/events")
def list_events(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    stmt = select(CalendarEvent).where(CalendarEvent.user_id == current_user.id)
    if date_from:
        stmt = stmt.where(CalendarEvent.start_at >= date_from)
    if date_to:
        stmt = stmt.where(CalendarEvent.start_at <= date_to)
    stmt = stmt.order_by(CalendarEvent.start_at)

    events = session.exec(stmt).all()
    return [e.to_dict() for e in events]


@router.post("/events")
def create_event(
    body: CreateEventRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if body.related_case_id:
        case = session.get(SavedCase, body.related_case_id)
        if case is None or case.user_id != current_user.id:
            raise MessageHTTPException(status_code=404, detail="Case not found")
    event = CalendarEvent(
        user_id=current_user.id,
        title=body.title,
        event_type=body.event_type,
        start_at=body.start_at,
        end_at=body.end_at,
        related_case_id=body.related_case_id,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return event.to_dict()


@router.delete("/events/{event_id}")
def delete_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    event = session.get(CalendarEvent, event_id)
    if event is None or event.user_id != current_user.id:
        raise MessageHTTPException(status_code=404, detail="Event not found")
    remote = {"google": event.google_calendar_event_id, "outlook": event.outlook_event_id}
    session.delete(event)
    session.commit()
    for provider, remote_id in remote.items():
        conn = remote_id and _connection(session, current_user.id, provider)
        if conn:
            try:
                calendar_oauth.delete_event(provider, calendar_sync.provider_token(conn), remote_id)
            except Exception as e:
                print(f"[CalendarSync] couldn't remove {provider} event {remote_id}: {e}")
    return {"message": "Event deleted"}


def _provider_or_404(provider: str) -> str:
    if provider not in calendar_oauth.PROVIDERS:
        raise MessageHTTPException(status_code=404, detail="Unknown calendar provider")
    return provider


def _connection(session: Session, user_id: str, provider: str) -> Optional[CalendarConnection]:
    return session.exec(
        select(CalendarConnection).where(
            CalendarConnection.user_id == user_id, CalendarConnection.provider == provider
        )
    ).first()


@router.get("/sync/providers")
def sync_providers(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    out = []
    for name in calendar_oauth.PROVIDERS:
        conn = _connection(session, current_user.id, name)
        out.append(
            {
                "provider": name,
                "configured": calendar_oauth.is_configured(name),
                "connected": conn is not None,
                "lastSyncedAt": conn.last_synced_at.isoformat() if conn and conn.last_synced_at else None,
            }
        )
    return out


@router.get("/sync/{provider}/connect")
def sync_connect(provider: str, current_user: User = Depends(get_current_user)):
    _provider_or_404(provider)
    if not calendar_oauth.is_configured(provider):
        raise MessageHTTPException(status_code=501, detail=f"{provider.title()} sync isn't configured on this server")
    state = jwt.encode(
        {"uid": current_user.id, "provider": provider, "exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
        jwt_secret(),
        algorithm="HS256",
    )
    return {"url": calendar_oauth.build_auth_url(provider, state)}


@router.get("/sync/{provider}/callback")
def sync_callback(
    provider: str,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    session: Session = Depends(get_session),
):
    """Browser redirect target after consent — no auth header; the signed
    `state` identifies the user."""
    _provider_or_404(provider)
    back = f"{get_settings().client_app_url.rstrip('/')}/#/calendar"
    if error or not code or not state:
        return RedirectResponse(f"{back}?calendar_error={provider}")
    try:
        claims = jwt.decode(state, jwt_secret(), algorithms=["HS256"])
        if claims.get("provider") != provider:
            raise jwt.PyJWTError("provider mismatch")
        refresh = calendar_oauth.exchange_code(provider, code)
    except (jwt.PyJWTError, calendar_oauth.CalendarProviderError) as e:
        print(f"[CalendarSync] callback failed: {e}")
        return RedirectResponse(f"{back}?calendar_error={provider}")

    conn = _connection(session, claims["uid"], provider)
    if conn is None:
        conn = CalendarConnection(user_id=claims["uid"], provider=provider, refresh_token_enc="")
    conn.refresh_token_enc = calendar_oauth.encrypt_token(refresh)
    session.add(conn)
    session.commit()
    return RedirectResponse(f"{back}?calendar_connected={provider}")


@router.post("/sync/{provider}")
def sync_now(
    provider: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _provider_or_404(provider)
    conn = _connection(session, current_user.id, provider)
    if conn is None:
        raise MessageHTTPException(status_code=400, detail=f"Connect your {provider.title()} calendar first")
    try:
        return calendar_sync.sync_connection(session, conn)
    except calendar_oauth.CalendarProviderError as e:
        raise MessageHTTPException(status_code=502, detail=str(e))


@router.delete("/sync/{provider}")
def sync_disconnect(
    provider: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _provider_or_404(provider)
    conn = _connection(session, current_user.id, provider)
    if conn is not None:
        session.delete(conn)
        session.commit()
    return {"message": "Disconnected"}
