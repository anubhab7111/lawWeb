"""
Court Cause List Search — cache-first over app/tools/case_data_provider.py.
Public (no auth required — search doesn't need a saved case). Filters are
applied in-process over the (small, single-day/single-court) cached result
set rather than as JSONB queries.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, delete, select

from app.db.engine import get_engine
from app.db.models import CauseListCache
from app.tools.case_data_provider import CaseDataProviderError, get_case_data_provider

router = APIRouter(prefix="/api/cause-list", tags=["cause-list"])

_CACHE_TTL_SECONDS = 3600  # cause lists are published once/day but hit repeatedly


COURTS = [
    "Supreme Court of India",
    "Delhi High Court",
    "Bombay High Court",
    "Madras High Court",
    "Calcutta High Court",
    "Karnataka High Court",
    "Allahabad High Court",
    "Gujarat High Court",
    "Punjab and Haryana High Court",
    "Kerala High Court",
    "District Court, Pune",
]


def _is_fresh(fetched_at: Optional[datetime]) -> bool:
    return (
        fetched_at is not None
        and (datetime.now(timezone.utc) - fetched_at).total_seconds() < _CACHE_TTL_SECONDS
    )


def _read_cache(court: str, list_date: str):
    """(entries, fetched_at) for a cached list, or None. Blocking — run in a thread."""
    with Session(get_engine()) as session:
        row = session.exec(
            select(CauseListCache).where(
                CauseListCache.court == court, CauseListCache.list_date == list_date
            )
        ).first()
        return (row.entries, row.fetched_at) if row else None


def _store_cache(court: str, list_date: str, entries_json: list):
    """Upsert one cached list and prune rows older than two days. Blocking."""
    with Session(get_engine()) as session:
        session.exec(
            delete(CauseListCache).where(
                CauseListCache.fetched_at < datetime.now(timezone.utc) - timedelta(days=2)
            )
        )
        row = session.exec(
            select(CauseListCache).where(
                CauseListCache.court == court, CauseListCache.list_date == list_date
            )
        ).first()
        if row is None:
            row = CauseListCache(court=court, list_date=list_date, entries=entries_json)
        else:
            row.entries = entries_json
            row.fetched_at = datetime.now(timezone.utc)
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            # A concurrent request inserted the same (court, date) first.
            session.rollback()
            row = session.exec(
                select(CauseListCache).where(
                    CauseListCache.court == court, CauseListCache.list_date == list_date
                )
            ).first()
        return row.entries, row.fetched_at


@router.get("/courts")
async def list_courts():
    return {"courts": COURTS}


@router.get("/search")
async def search(
    court: str = Query(...),
    list_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
    advocate: Optional[str] = None,
    judge: Optional[str] = None,
    case_number: Optional[str] = Query(default=None, alias="caseNumber"),
):
    if court not in COURTS:
        return JSONResponse(status_code=400, content={"message": "Unknown court"})
    try:
        parsed_date = date.fromisoformat(list_date)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid date, expected YYYY-MM-DD"},
        )

    cached = await run_in_threadpool(_read_cache, court, list_date)
    if cached is None or not _is_fresh(cached[1]):
        provider = get_case_data_provider()
        try:
            entries = await provider.fetch_cause_list(court, parsed_date)
        except CaseDataProviderError as e:
            return JSONResponse(status_code=502, content={"message": str(e)})
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
        cached = await run_in_threadpool(_store_cache, court, list_date, entries_json)

    all_entries, fetched_at = cached
    results = all_entries
    if advocate:
        needle = advocate.lower()
        results = [e for e in results if needle in (e.get("advocate") or "").lower()]
    if judge:
        needle = judge.lower()
        results = [e for e in results if needle in (e.get("judge") or "").lower()]
    if case_number:
        needle = case_number.lower()
        results = [e for e in results if needle in (e.get("caseNumber") or "").lower()]

    return {
        "court": court,
        "date": list_date,
        "publishedAt": fetched_at.isoformat() if fetched_at else None,
        "published": len(all_entries) > 0,
        "results": results,
    }
