"""
Personal Legal Calendar — mostly auto-populated (hearings from My Cases sync,
lawyer meetings from confirmed bookings — see the additive hook in
app/routers/bookings.py), plus manual custom entries. Google/Outlook sync are
out of scope for this MVP (see the plan's Phase 4 risk notes on OAuth token
lifecycle) — /sync/google below is a documented stub, not implemented.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import CalendarEvent, User
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
    session.delete(event)
    session.commit()
    return {"message": "Event deleted"}


@router.post("/sync/google")
def sync_google():
    """Stretch goal, not implemented in this MVP — needs an OAuth consent
    flow, token storage, and refresh-token maintenance (see the plan).
    Ship the internal calendar first."""
    raise MessageHTTPException(status_code=501, detail="Google Calendar sync is not yet implemented")
