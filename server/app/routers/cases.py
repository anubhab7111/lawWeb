"""
My Cases — save/track court cases by CNR or case number: timeline, hearing
dates, orders, AI summaries, personal notes. All routes require a logged-in
user (see app/deps/auth.py); every lookup is scoped to the caller's own
saved_cases rows.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import CaseAiSummary, CaseEvent, CaseNote, SavedCase, User
from app.deps.auth import get_current_user
from app.deps.errors import MessageHTTPException
from app.tools.case_data_provider import CaseDataProviderError, get_case_data_provider
from app.tools.case_sync import sync_case_events

router = APIRouter(prefix="/api/cases", tags=["cases"])


class SaveCaseRequest(BaseModel):
    cnr: Optional[str] = None
    court: Optional[str] = None
    case_number: Optional[str] = Field(default=None, alias="caseNumber")
    year: Optional[int] = None

    class Config:
        populate_by_name = True


class AddNoteRequest(BaseModel):
    note_text: str = Field(..., alias="noteText", min_length=1, max_length=5000)

    class Config:
        populate_by_name = True


def _get_owned_case(session: Session, case_id: str, user: User) -> SavedCase:
    case = session.get(SavedCase, case_id)
    if case is None or case.user_id != user.id:
        raise MessageHTTPException(status_code=404, detail="Case not found")
    return case


@router.post("")
async def save_case(
    body: SaveCaseRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    provider = get_case_data_provider()
    try:
        if body.cnr:
            record = await provider.fetch_case_by_cnr(body.cnr)
        elif body.court and body.case_number and body.year:
            record = await provider.fetch_case_by_number(body.court, body.case_number, body.year)
        else:
            raise MessageHTTPException(
                status_code=400, detail="Provide either a CNR or court+caseNumber+year"
            )
    except CaseDataProviderError as e:
        raise MessageHTTPException(status_code=404, detail=str(e))

    if record.cnr:
        existing = session.exec(
            select(SavedCase).where(SavedCase.user_id == current_user.id, SavedCase.cnr == record.cnr)
        ).first()
    else:
        # No CNR from the provider — cnr alone can't identify a duplicate
        # (the DB's own partial unique index allows multiple NULL-cnr rows
        # per user), so fall back to the court+case_number+year natural key.
        existing = session.exec(
            select(SavedCase).where(
                SavedCase.user_id == current_user.id,
                SavedCase.cnr.is_(None),
                SavedCase.court == record.court,
                SavedCase.case_number == record.case_number,
                SavedCase.year == record.year,
            )
        ).first()
    if existing is not None:
        raise MessageHTTPException(status_code=400, detail="Case already saved")

    case = SavedCase(
        user_id=current_user.id,
        cnr=record.cnr,
        court=record.court,
        case_number=record.case_number,
        year=record.year,
        title=record.title,
        status=record.status,
    )
    session.add(case)
    session.commit()
    session.refresh(case)

    await sync_case_events(session, case)
    session.refresh(case)

    return case.to_dict()


@router.get("")
def list_cases(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    cases = session.exec(
        select(SavedCase).where(SavedCase.user_id == current_user.id).order_by(SavedCase.created_at.desc())
    ).all()
    return [c.to_dict() for c in cases]


@router.get("/{case_id}")
def get_case(
    case_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    case = _get_owned_case(session, case_id, current_user)
    events = session.exec(
        select(CaseEvent).where(CaseEvent.saved_case_id == case.id).order_by(CaseEvent.event_date.desc())
    ).all()
    notes = session.exec(
        select(CaseNote).where(CaseNote.saved_case_id == case.id).order_by(CaseNote.created_at.desc())
    ).all()
    summaries = session.exec(
        select(CaseAiSummary).where(CaseAiSummary.saved_case_id == case.id).order_by(CaseAiSummary.created_at.desc())
    ).all()
    return {
        **case.to_dict(),
        "timeline": [e.to_dict() for e in events],
        "notes": [n.to_dict() for n in notes],
        "aiSummaries": [s.to_dict() for s in summaries],
    }


@router.post("/{case_id}/notes")
def add_note(
    case_id: str,
    body: AddNoteRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    case = _get_owned_case(session, case_id, current_user)
    note = CaseNote(saved_case_id=case.id, user_id=current_user.id, note_text=body.note_text)
    session.add(note)
    session.commit()
    session.refresh(note)
    return note.to_dict()


@router.post("/{case_id}/sync")
async def sync_case(
    case_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    case = _get_owned_case(session, case_id, current_user)
    await sync_case_events(session, case)
    session.refresh(case)
    return case.to_dict()


@router.delete("/{case_id}")
def delete_case(
    case_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    case = _get_owned_case(session, case_id, current_user)
    session.delete(case)
    session.commit()
    return {"message": "Case deleted"}
