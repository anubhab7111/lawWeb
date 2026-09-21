"""In-memory sliding-window rate limit for the LLM-backed chat endpoints.

Per process, which is correct for this deployment (a single uvicorn worker —
see CLAUDE.md). Each generation can hold the 4GB GPU for up to three minutes,
so unauthenticated callers in particular must not be able to queue them
without bound. Behind a reverse proxy, request.client.host is the proxy's
address; configure the proxy to pass the real client address (uvicorn's
--proxy-headers) before relying on the per-IP limit.
"""

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, HTTPException, Request

from app.config import get_settings
from app.db.models import User
from app.deps.auth import get_current_user_optional

_WINDOW_SECONDS = 60.0
_MAX_TRACKED_KEYS = 10_000

_hits: Dict[str, Deque[float]] = defaultdict(deque)


def _prune(now: float) -> None:
    for key in [k for k, q in _hits.items() if not q or now - q[-1] > _WINDOW_SECONDS]:
        del _hits[key]


def check_rate_limit(key: str, limit: int, now: Optional[float] = None) -> Optional[int]:
    """Record a hit for `key`. Returns None if allowed, else the seconds until
    the oldest hit leaves the window."""
    now = time.monotonic() if now is None else now
    if len(_hits) > _MAX_TRACKED_KEYS:
        _prune(now)
    hits = _hits[key]
    while hits and now - hits[0] > _WINDOW_SECONDS:
        hits.popleft()
    if len(hits) >= limit:
        return max(1, int(_WINDOW_SECONDS - (now - hits[0])) + 1)
    hits.append(now)
    return None


def reset_rate_limits() -> None:
    _hits.clear()


async def chat_rate_limit(
    request: Request, user: Optional[User] = Depends(get_current_user_optional)
) -> None:
    limit = get_settings().chat_rate_limit_per_minute
    if limit <= 0:
        return
    key = f"user:{user.id}" if user else f"ip:{request.client.host if request.client else '?'}"
    retry_after = check_rate_limit(key, limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="You're sending messages too quickly. Please wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )
