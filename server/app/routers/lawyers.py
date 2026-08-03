"""
Lawyer directory endpoints backed by the PostgreSQL lawyers table,
ported from the old Express server/src/routes/lawyers.ts.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
def list_lawyers(session: Session = Depends(get_session)):
    lawyers = session.exec(select(Lawyer).order_by(Lawyer.id)).all()
    return [lawyer.to_dict() for lawyer in lawyers]


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
