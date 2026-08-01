"""
Court Cause List Search — cache-first over app/tools/case_data_provider.py.
Public (no auth required — search doesn't need a saved case). Filters are
applied in-process over the (small, single-day/single-court) cached result
set rather than as JSONB queries.
"""

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import CauseListCache
from app.tools.case_data_provider import get_case_data_provider

router = APIRouter(prefix="/api/cause-list", tags=["cause-list"])

_CACHE_TTL_SECONDS = 3600  # cause lists are published once/day but hit repeatedly


@router.get("/search")
async def search(
    court: str = Query(...),
    list_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
    advocate: Optional[str] = None,
    judge: Optional[str] = None,
    case_number: Optional[str] = Query(default=None, alias="caseNumber"),
):
    try:
        parsed_date = date.fromisoformat(list_date)
    except ValueError:
        return {"message": "Invalid date, expected YYYY-MM-DD"}

    with Session(get_engine()) as session:
        cached = session.exec(
            select(CauseListCache).where(
                CauseListCache.court == court, CauseListCache.list_date == list_date
            )
        ).first()

        is_fresh = (
            cached is not None
            and cached.fetched_at is not None
            and (datetime.now(timezone.utc) - cached.fetched_at).total_seconds() < _CACHE_TTL_SECONDS
        )

        if not is_fresh:
            provider = get_case_data_provider()
            entries = await provider.fetch_cause_list(court, parsed_date)
            entries_json = [
                {
                    "court": e.court,
                    "caseNumber": e.case_number,
                    "title": e.title,
                    "advocate": e.advocate,
                    "judge": e.judge,
                    "itemNumber": e.item_number,
                    "timeSlot": e.time_slot,
                }
                for e in entries
            ]
            if cached is None:
                cached = CauseListCache(court=court, list_date=list_date, entries=entries_json)
            else:
                cached.entries = entries_json
                cached.fetched_at = datetime.now(timezone.utc)
            session.add(cached)
            session.commit()

        results = cached.entries
        if advocate:
            results = [e for e in results if advocate.lower() in e.get("advocate", "").lower()]
        if judge:
            results = [e for e in results if judge.lower() in e.get("judge", "").lower()]
        if case_number:
            results = [e for e in results if case_number.lower() in e.get("caseNumber", "").lower()]

        return {
            "court": court,
            "date": list_date,
            "publishedAt": cached.fetched_at.isoformat() if cached.fetched_at else None,
            "results": results,
        }
