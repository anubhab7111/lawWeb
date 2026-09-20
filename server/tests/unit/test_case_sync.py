import asyncio
import uuid

from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.init_db import init_db
from app.db.models import CaseAiSummary, CaseEvent, SavedCase, User
from app.tools import case_sync


def test_resync_is_idempotent_and_summarises_orders(monkeypatch):
    init_db(embed=False)
    summaries = []

    async def fake_summary(*_args):
        summaries.append(1)
        return "Summary"

    monkeypatch.setattr(case_sync, "summarize_case_event", fake_summary)

    with Session(get_engine()) as s:
        u = User(name="S", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add(u)
        s.commit()
        case = SavedCase(user_id=u.id, cnr=f"CNR{uuid.uuid4().hex[:10]}", title="T")
        s.add(case)
        s.commit()
        s.refresh(case)

        first = asyncio.run(case_sync.sync_case_events(s, case))
        second = asyncio.run(case_sync.sync_case_events(s, case))

        assert {e.event_type for e in first} == {"filing", "order", "hearing"}
        assert second == []
        assert len(summaries) == 1
        assert len(s.exec(select(CaseEvent).where(CaseEvent.saved_case_id == case.id)).all()) == 3
        assert len(s.exec(select(CaseAiSummary).where(CaseAiSummary.saved_case_id == case.id)).all()) == 1
        assert case.last_synced_at is not None
