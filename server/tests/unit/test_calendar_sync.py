import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import CalendarConnection, CalendarEvent, User
from app.main import app
from app.routers.auth import _make_token
from app.services import calendar_oauth, calendar_sync


@pytest.fixture(scope="module")
def client():
    from app.db.init_db import init_db

    init_db(embed=False)
    return TestClient(app, follow_redirects=False)


def _user():
    with Session(get_engine()) as s:
        u = User(name="C", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add(u)
        s.commit()
        s.refresh(u)
        s.add(
            CalendarEvent(
                user_id=u.id, title="Hearing", event_type="hearing",
                start_at=datetime.now(timezone.utc) + timedelta(days=2),
            )
        )
        s.commit()
        return u.id, {"Authorization": f"Bearer {_make_token(u.id)}"}


def test_connect_callback_sync_disconnect(client, monkeypatch):
    monkeypatch.setattr(calendar_oauth, "is_configured", lambda p: True)
    monkeypatch.setattr(calendar_oauth, "build_auth_url", lambda p, state: f"https://idp.example/auth?state={state}")
    monkeypatch.setattr(calendar_oauth, "exchange_code", lambda p, code: "refresh-123")
    monkeypatch.setattr(calendar_oauth, "access_token", lambda p, refresh: "access-1")
    pushed = []
    monkeypatch.setattr(
        calendar_oauth, "push_event",
        lambda provider, token, title, start, end, existing: pushed.append(title) or f"remote-{len(pushed)}",
    )

    uid, headers = _user()
    url = client.get("/api/calendar/sync/google/connect", headers=headers).json()["url"]
    state = parse_qs(urlparse(url).query)["state"][0]

    bad = client.get("/api/calendar/sync/google/callback?code=x&state=garbage")
    assert "calendar_error=google" in bad.headers["location"]

    ok = client.get(f"/api/calendar/sync/google/callback?code=abc&state={state}")
    assert "calendar_connected=google" in ok.headers["location"]

    with Session(get_engine()) as s:
        conn = s.exec(select(CalendarConnection).where(CalendarConnection.user_id == uid)).one()
        assert conn.refresh_token_enc != "refresh-123"
        assert calendar_oauth.decrypt_token(conn.refresh_token_enc) == "refresh-123"

    first = client.post("/api/calendar/sync/google", headers=headers).json()
    second = client.post("/api/calendar/sync/google", headers=headers).json()
    assert first["pushed"] == 1 and second["pushed"] == 0
    assert pushed == ["Hearing"]

    providers = {p["provider"]: p for p in client.get("/api/calendar/sync/providers", headers=headers).json()}
    assert providers["google"]["connected"] and providers["google"]["lastSyncedAt"]

    assert client.delete("/api/calendar/sync/google", headers=headers).status_code == 200
    assert client.post("/api/calendar/sync/google", headers=headers).status_code == 400


def test_unconfigured_provider_reports_501(client):
    _, headers = _user()
    assert client.get("/api/calendar/sync/outlook/connect", headers=headers).status_code == 501
    assert client.get("/api/calendar/sync/nope/connect", headers=headers).status_code == 404


def test_token_roundtrip():
    assert calendar_sync  # module imports cleanly
    assert calendar_oauth.decrypt_token(calendar_oauth.encrypt_token("t")) == "t"
