import uuid
from datetime import date, datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.db.engine import get_engine
from app.db.models import User
from app.main import app
from app.metrics.retrieval_metrics import act_aware_sections, compute_hit_rate
from app.security import jwt_secret
from app.tools.base_legal_rag import file_fingerprint
from app.tools.citation_verifier import iter_citation_occurrences


@pytest.fixture(scope="module")
def client():
    from app.db.init_db import init_db

    init_db(embed=False)
    return TestClient(app)


def test_citation_lists_are_all_captured():
    occ = iter_citation_occurrences("Sections 73 and 74 of the Indian Contract Act apply.")
    assert [o.section for o in occ] == ["73", "74"]
    assert {o.act_hint for o in occ} == {"Indian Contract"}
    occ = iter_citation_occurrences("Articles 14, 19 and 21 of the Constitution.")
    assert [o.section for o in occ] == ["14", "19", "21"]


def test_hit_requires_the_right_act():
    retrieved = [("Indian Penal Code", "420"), ("Code of Criminal Procedure", "438")]
    expected = ["Code of Criminal Procedure"]
    plain = compute_hit_rate([s for _, s in retrieved], ["420"], k=2)
    strict = compute_hit_rate(act_aware_sections(retrieved, expected), ["420"], k=2)
    assert plain == 1.0 and strict == 0.0
    # naming mismatch (no expected act matches anything) falls back to plain numbers
    assert act_aware_sections(retrieved, ["Article 19"]) == ["420", "438"]


def test_fingerprint_sees_same_size_edits(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("aaaa")
    before = file_fingerprint(f)
    f.write_text("bbbb")
    assert file_fingerprint(f) != before


def test_cause_list_cached_and_validated(client):
    when = (date.today() + timedelta(days=random_offset())).isoformat()
    assert client.get("/api/cause-list/search", params={"court": "Nowhere", "date": when}).status_code == 400
    first = client.get("/api/cause-list/search", params={"court": "Delhi High Court", "date": when}).json()
    second = client.get("/api/cause-list/search", params={"court": "Delhi High Court", "date": when}).json()
    assert first["published"] and first["results"] == second["results"]
    assert first["publishedAt"] == second["publishedAt"]  # second call served from cache


def random_offset():
    import random

    return random.randint(5, 900)


def test_me_hands_back_a_fresh_token_near_expiry(client):
    with Session(get_engine()) as s:
        u = User(name="R", email=f"{uuid.uuid4()}@example.com", password="x")
        s.add(u)
        s.commit()
        s.refresh(u)
        uid = u.id
    soon = jwt.encode(
        {"id": uid, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, jwt_secret(), algorithm="HS256"
    )
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {soon}"})
    assert r.status_code == 200 and "x-refreshed-token" in r.headers

    fresh = jwt.encode(
        {"id": uid, "exp": datetime.now(timezone.utc) + timedelta(minutes=50)}, jwt_secret(), algorithm="HS256"
    )
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {fresh}"})
    assert "x-refreshed-token" not in r.headers
