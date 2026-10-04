import asyncio
import json
from datetime import date

import httpx

import app.metrics.llm_judge as lj


class FakeResponse:
    def __init__(self, status, content="{}"):
        self.status_code = status
        self.text = "quota" if status == 429 else ""
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._content}, "finish_reason": "stop"}]}


def make_judge(monkeypatch, tmp_path, responses_by_key, usage=None):
    path = tmp_path / "usage.json"
    if usage is not None:
        path.write_text(json.dumps(usage))
    monkeypatch.setattr(lj, "_USAGE_PATH", path)

    async def no_throttle():
        return None

    monkeypatch.setattr(lj, "_throttle", no_throttle)
    seen = []

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers, json):
            key = headers["Authorization"].split()[-1]
            seen.append(key)
            return responses_by_key[key]()

    monkeypatch.setattr(lj.httpx, "AsyncClient", Client)
    judge = lj.LLMJudge(api_key="primary", daily_limit=2, retry_delay_s=0)
    judge._api_keys = ["primary", "alt"]
    return judge, seen, path


def test_alternate_key_takes_over_at_the_daily_limit(monkeypatch, tmp_path):
    judge, seen, path = make_judge(
        monkeypatch, tmp_path, {"primary": lambda: FakeResponse(200), "alt": lambda: FakeResponse(200)},
        usage={"date": date.today().isoformat(), "count": 2},  # legacy format belongs to the primary key
    )
    raw, _, reason = asyncio.run(judge._post_chat("p"))
    assert raw == "{}" and reason is None and seen == ["alt"]
    counts = json.loads(path.read_text())["by_key"]
    assert counts[lj._key_id("primary")] == 2 and counts[lj._key_id("alt")] == 1
    assert "primary" not in path.read_text() and "alt" not in path.read_text()


def test_alternate_key_takes_over_after_a_quota_refusal(monkeypatch, tmp_path):
    judge, seen, _ = make_judge(
        monkeypatch, tmp_path, {"primary": lambda: FakeResponse(429), "alt": lambda: FakeResponse(200)},
    )
    raw, _, reason = asyncio.run(judge._post_chat("p"))
    assert raw == "{}" and seen == ["primary", "alt"]
    asyncio.run(judge._post_chat("p"))
    assert seen[-1] == "alt"


def test_every_key_spent_returns_a_failure_without_calling(monkeypatch, tmp_path):
    today = date.today().isoformat()
    judge, seen, _ = make_judge(
        monkeypatch, tmp_path, {"primary": lambda: FakeResponse(200), "alt": lambda: FakeResponse(200)},
        usage={"date": today, "by_key": {lj._key_id("primary"): 2, lj._key_id("alt"): 2}},
    )
    raw, _, reason = asyncio.run(judge._post_chat("p"))
    assert raw is None and "budget" in reason and seen == []
