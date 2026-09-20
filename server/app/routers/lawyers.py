"""
Lawyer directory endpoints backed by the PostgreSQL lawyers table,
ported from the old Express server/src/routes/lawyers.ts.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import Lawyer
from app.tools.lawyer_recommender import recommend_lawyers as recommend_lawyers_core

router = APIRouter(prefix="/api/lawyers", tags=["lawyers"])


class RecommendRequest(BaseModel):
    problemDescription: Optional[str] = None
    specialty: Optional[str] = None
    location: Optional[str] = None
    maxHourlyRate: Optional[int] = None
    minRating: Optional[float] = None


@router.get("")
def list_lawyers(
    page: Optional[int] = Query(default=None, ge=1),
    pageSize: int = Query(default=24, ge=1, le=100),
    q: Optional[str] = None,
    specialty: Optional[str] = None,
    location: Optional[str] = None,
    ids: Optional[str] = Query(default=None, description="Comma-separated lawyer ids"),
    session: Session = Depends(get_session),
):
    """Without `page` (or `ids`) this returns the whole directory as a plain
    array, as before. With `page`, it returns {items, total, page, pageSize};
    with `ids`, just those lawyers."""
    stmt = select(Lawyer)
    if ids:
        stmt = stmt.where(Lawyer.id.in_([i for i in ids.split(",") if i][:100]))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Lawyer.name.ilike(like), Lawyer.specialty.ilike(like)))
    if specialty:
        stmt = stmt.where(Lawyer.specialty == specialty)
    if location:
        stmt = stmt.where(Lawyer.location.ilike(f"%{location}%"))

    if page is None:
        return [l.to_dict() for l in session.exec(stmt.order_by(Lawyer.id)).all()]

    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    rows = session.exec(
        stmt.order_by(Lawyer.rating.desc(), Lawyer.id).offset((page - 1) * pageSize).limit(pageSize)
    ).all()
    return {"items": [l.to_dict() for l in rows], "total": total, "page": page, "pageSize": pageSize}


@router.get("/filters")
def lawyer_filters(session: Session = Depends(get_session)):
    """Values for the directory's filter dropdowns."""
    specialties = session.exec(select(Lawyer.specialty).distinct().order_by(Lawyer.specialty)).all()
    locations = session.exec(select(Lawyer.location).distinct()).all()
    states = sorted({loc.split(", ", 1)[1] for loc in locations if ", " in loc})
    return {"specialties": specialties, "states": states}


@router.get("/{lawyer_id}")
def get_lawyer(lawyer_id: str, session: Session = Depends(get_session)):
    lawyer = session.get(Lawyer, lawyer_id)
    if not lawyer:
        return JSONResponse(status_code=404, content={"message": "Lawyer not found"})
    return lawyer.to_dict()


@router.post("/recommend")
async def recommend_lawyers(body: RecommendRequest, session: Session = Depends(get_session)):
    lawyers = await recommend_lawyers_core(
        session,
        problem_description=body.problemDescription,
        specialty=body.specialty,
        location=body.location,
        max_hourly_rate=body.maxHourlyRate,
        min_rating=body.minRating,
    )
    return [lawyer.to_dict() for lawyer in lawyers]
