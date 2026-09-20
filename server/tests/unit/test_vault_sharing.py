import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.db.engine import get_engine
from app.db.models import User, VaultDocument
from app.main import app
from app.routers.auth import _make_token


@pytest.fixture(scope="module")
def client():
    from app.db.init_db import init_db

    init_db(embed=False)
    return TestClient(app)


def _user(email: str) -> tuple[str, dict]:
    with Session(get_engine()) as s:
        u = User(name=email.split("@")[0], email=email, password="x")
        s.add(u)
        s.commit()
        s.refresh(u)
        return u.id, {"Authorization": f"Bearer {_make_token(u.id)}"}


def test_share_validation_edit_and_owner_flags(client):
    owner_email = f"{uuid.uuid4()}@example.com"
    friend_email = f"{uuid.uuid4()}@example.com"
    owner_id, owner_h = _user(owner_email)
    friend_id, friend_h = _user(friend_email)

    with Session(get_engine()) as s:
        doc = VaultDocument(
            user_id=owner_id, title="Order", document_type="order", object_key="k/x-a.txt",
            file_size_bytes=1, mime_type="text/plain", indexing_status="ready",
        )
        s.add(doc)
        s.commit()
        doc_id = doc.id

    assert client.post(f"/api/vault/documents/{doc_id}/share", headers=owner_h, json={"email": owner_email}).status_code == 400
    assert client.post(f"/api/vault/documents/{doc_id}/share", headers=owner_h, json={"email": friend_email, "permission": "owner"}).status_code == 400
    assert client.post(f"/api/vault/documents/{doc_id}/share", headers=owner_h, json={"email": "nobody@example.com"}).status_code == 404

    ok = client.post(f"/api/vault/documents/{doc_id}/share", headers=owner_h, json={"email": friend_email})
    assert ok.status_code == 200
    again = client.post(f"/api/vault/documents/{doc_id}/share", headers=owner_h, json={"email": friend_email, "permission": "edit"})
    assert again.json()["id"] == ok.json()["id"] and again.json()["permission"] == "edit"

    listing = client.get("/api/vault/documents", headers=friend_h).json()
    mine = next(d for d in listing if d["id"] == doc_id)
    assert mine["isOwner"] is False

    renamed = client.patch(f"/api/vault/documents/{doc_id}", headers=friend_h, json={"title": "Renamed"})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Renamed"
    assert client.delete(f"/api/vault/documents/{doc_id}", headers=friend_h).status_code == 404
