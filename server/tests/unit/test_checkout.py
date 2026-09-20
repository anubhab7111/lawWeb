import uuid
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.init_db import init_db
from app.db.models import Booking, Lawyer, User
from app.main import app
from app.routers import bookings
from app.routers.auth import _make_token


@pytest.fixture(scope="module")
def client():
    init_db(embed=False)
    return TestClient(app)


@pytest.fixture()
def user_and_lawyer():
    with Session(get_engine()) as s:
        user = User(name="T", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add(user)
        s.commit()
        s.refresh(user)
        lawyer = s.exec(select(Lawyer)).first()
        return user.id, lawyer.id, lawyer.hourly_rate


class FakeGateway:
    def __init__(self):
        self.charges = 0
        self.transaction = SimpleNamespace(sale=self.sale)

    def sale(self, payload):
        self.charges += 1
        return SimpleNamespace(
            is_success=True, message="", transaction=SimpleNamespace(id=f"tx-{self.charges}")
        )


def _total(rate):
    return f"{rate + round(rate * 0.05):.2f}"


def _post(client, user_id, lawyer_id, rate, key, when, time="10:00"):
    return client.post(
        "/api/bookings/checkout",
        headers={"Authorization": f"Bearer {_make_token(user_id)}", "Idempotency-Key": key},
        json={
            "amount": _total(rate),
            "paymentMethodNonce": "fake-valid-nonce",
            "lawyerId": lawyer_id,
            "appointmentDate": when.isoformat(),
            "appointmentTime": time,
        },
    )


def test_checkout_requires_slot(client, user_and_lawyer, monkeypatch):
    uid, lid, rate = user_and_lawyer
    monkeypatch.setattr(bookings, "get_gateway", lambda: FakeGateway())
    r = client.post(
        "/api/bookings/checkout",
        headers={"Authorization": f"Bearer {_make_token(uid)}"},
        json={"amount": _total(rate), "paymentMethodNonce": "n", "lawyerId": lid},
    )
    assert r.status_code == 400


def test_idempotent_and_conflict(client, user_and_lawyer, monkeypatch):
    uid, lid, rate = user_and_lawyer
    gw = FakeGateway()
    monkeypatch.setattr(bookings, "get_gateway", lambda: gw)
    when = date.today() + timedelta(days=3)
    key = str(uuid.uuid4())

    first = _post(client, uid, lid, rate, key, when)
    assert first.status_code == 200 and first.json()["status"] == "success"
    again = _post(client, uid, lid, rate, key, when)
    assert again.json()["transactionId"] == first.json()["transactionId"]
    assert gw.charges == 1

    clash = _post(client, uid, lid, rate, str(uuid.uuid4()), when)
    assert clash.status_code == 409
    assert gw.charges == 1

    with Session(get_engine()) as s:
        row = s.exec(select(Booking).where(Booking.idempotency_key == key)).one()
        assert row.status.value == "confirmed"
