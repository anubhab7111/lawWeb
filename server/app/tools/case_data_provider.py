"""
Case-data provider adapter (My Cases, Hearing Reminders, Cause List Search).

Everything else in the codebase talks to CaseDataProvider — never a
concrete vendor class — so swapping the licensed vendor later is a
one-line settings change (case_data_provider) plus one new subclass here.

The default provider ("mock") is an in-memory fixture so these three
features are fully testable in dev without a live vendor contract/API key.
Wire a real vendor (e.g. eCourtsIndia) by adding a subclass below and
setting CASE_DATA_PROVIDER + CASE_DATA_API_KEY + CASE_DATA_API_BASE_URL.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Optional

from app.config import get_settings


@dataclass
class CaseRecord:
    cnr: str
    court: str
    case_number: str
    year: int
    title: str
    status: str


@dataclass
class CaseEventRecord:
    event_type: str  # "hearing" | "order" | "filing"
    event_date: datetime
    title: str
    detail: str = ""
    source_url: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class CauseListFilter:
    advocate: Optional[str] = None
    judge: Optional[str] = None
    case_number: Optional[str] = None


@dataclass
class CauseListEntry:
    court: str
    case_number: str
    title: str
    advocate: str
    judge: str
    item_number: int
    time_slot: str = ""


class CaseDataProviderError(Exception):
    """Raised on lookup failure (not-found, rate-limited, vendor error).
    Callers should catch this rather than letting an unrelated exception
    type leak into routers — see app/routers/cases.py."""


class CaseDataProvider(ABC):
    @abstractmethod
    async def fetch_case_by_cnr(self, cnr: str) -> CaseRecord: ...

    @abstractmethod
    async def fetch_case_by_number(
        self, court: str, case_number: str, year: int
    ) -> CaseRecord: ...

    @abstractmethod
    async def fetch_case_history(self, cnr: str) -> List[CaseEventRecord]: ...

    @abstractmethod
    async def fetch_cause_list(
        self, court: str, list_date: date, filters: Optional[CauseListFilter] = None
    ) -> List[CauseListEntry]: ...


class MockCaseDataProvider(CaseDataProvider):
    """
    In-memory fixture provider for local dev / testing without a licensed
    vendor contract. Deterministic per-CNR (same CNR always returns the same
    fixture data) so the rest of the stack (My Cases sync, reminders, cause
    list caching) can be exercised end-to-end.
    """

    async def fetch_case_by_cnr(self, cnr: str) -> CaseRecord:
        if not cnr or len(cnr) < 4:
            raise CaseDataProviderError(f"No case found for CNR '{cnr}'")
        rng = random.Random(cnr)
        return CaseRecord(
            cnr=cnr,
            court=rng.choice(["Delhi High Court", "Bombay High Court", "District Court, Pune"]),
            case_number=f"CRL.{rng.randint(100, 999)}/{2024}",
            year=2024,
            title=f"State vs. Respondent ({cnr[-4:]})",
            status=rng.choice(["Pending", "Disposed", "Under Trial"]),
        )

    async def fetch_case_by_number(
        self, court: str, case_number: str, year: int
    ) -> CaseRecord:
        fake_cnr = f"MOCK{abs(hash((court, case_number, year))) % 10**12:012d}"
        return await self.fetch_case_by_cnr(fake_cnr)

    async def fetch_case_history(self, cnr: str) -> List[CaseEventRecord]:
        rng = random.Random(cnr)
        now = datetime.now(timezone.utc)
        events: List[CaseEventRecord] = [
            CaseEventRecord(
                event_type="filing",
                event_date=now - timedelta(days=180),
                title="Case filed",
            ),
            CaseEventRecord(
                event_type="order",
                event_date=now - timedelta(days=rng.randint(5, 30)),
                title="Interim order passed",
                detail="Notice issued to respondent; next hearing scheduled.",
            ),
            CaseEventRecord(
                event_type="hearing",
                event_date=now + timedelta(days=rng.randint(1, 20)),
                title="Next hearing",
            ),
        ]
        return events

    async def fetch_cause_list(
        self, court: str, list_date: date, filters: Optional[CauseListFilter] = None
    ) -> List[CauseListEntry]:
        rng = random.Random(f"{court}:{list_date.isoformat()}")
        entries = [
            CauseListEntry(
                court=court,
                case_number=f"CRL.{rng.randint(100, 999)}/2024",
                title=f"Case {i}",
                advocate=rng.choice(["A. Sharma", "R. Patel", "S. Iyer", "M. Khan"]),
                judge=rng.choice(["Hon'ble Justice A", "Hon'ble Justice B"]),
                item_number=i,
            )
            for i in range(1, rng.randint(5, 15))
        ]
        if filters:
            if filters.advocate:
                entries = [e for e in entries if filters.advocate.lower() in e.advocate.lower()]
            if filters.judge:
                entries = [e for e in entries if filters.judge.lower() in e.judge.lower()]
            if filters.case_number:
                entries = [e for e in entries if filters.case_number.lower() in e.case_number.lower()]
        return entries


@lru_cache()
def get_case_data_provider() -> CaseDataProvider:
    settings = get_settings()
    provider = settings.case_data_provider.lower()
    if provider == "mock":
        return MockCaseDataProvider()
    # Add licensed vendors here as they're contracted, e.g.:
    #   if provider == "ecourtsindia":
    #       return ECourtsIndiaProvider(
    #           api_key=settings.case_data_api_key,
    #           base_url=settings.case_data_api_base_url,
    #       )
    raise CaseDataProviderError(
        f"Unknown CASE_DATA_PROVIDER '{settings.case_data_provider}' — falling back to mock"
    )
