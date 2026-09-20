import uuid

import pytest
from sqlmodel import Session

from app.db.engine import get_engine
from app.db.models import Notification, NotificationPreference, User
from app.services import fcm_client, notification_dispatch


@pytest.fixture(scope="module", autouse=True)
def _db():
    from app.db.init_db import init_db

    init_db(embed=False)


def _user_with_push():
    with Session(get_engine()) as s:
        u = User(name="P", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add(u)
        s.commit()
        s.refresh(u)
        s.add(NotificationPreference(user_id=u.id, push_enabled=True, fcm_token="tok-1"))
        s.commit()
        return u.id


def test_push_sent_and_recorded(monkeypatch):
    sent = []
    monkeypatch.setattr(notification_dispatch, "push_configured", lambda: True)
    monkeypatch.setattr(notification_dispatch, "send_push", lambda tok, t, b: sent.append(tok) or True)
    uid = _user_with_push()
    with Session(get_engine()) as s:
        rows = notification_dispatch.send_notification(s, uid, "x", "Hello", "Body", channels=["in_app"])
        assert sorted(r.channel for r in rows) == ["in_app", "push"]
        assert all(r.status == "sent" for r in rows)
    assert sent == ["tok-1"]


def test_invalid_token_is_cleared(monkeypatch):
    def gone(*_a):
        raise fcm_client.PushTokenInvalid("tok-1")

    monkeypatch.setattr(notification_dispatch, "push_configured", lambda: True)
    monkeypatch.setattr(notification_dispatch, "send_push", gone)
    uid = _user_with_push()
    with Session(get_engine()) as s:
        notification_dispatch.send_notification(s, uid, "x", "Hello", "Body", channels=["in_app"])
        prefs = s.query(NotificationPreference).filter_by(user_id=uid).one()
        assert prefs.fcm_token is None
        failed = s.query(Notification).filter_by(user_id=uid, channel="push").one()
        assert failed.status == "failed"


def test_push_unconfigured_is_a_noop():
    assert fcm_client.push_configured() is False
    assert fcm_client.send_push("tok", "t", "b") is False
