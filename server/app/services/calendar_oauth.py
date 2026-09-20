"""
OAuth + event push for Google Calendar and Microsoft Outlook (Graph).

Only the refresh token is stored (encrypted with Fernet); access tokens are
minted per sync. A provider is "configured" when its client id/secret are set.
"""

import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.config import get_settings


@dataclass(frozen=True)
class Provider:
    name: str
    auth_url: str
    token_url: str
    scope: str
    extra_auth_params: Dict[str, str]


PROVIDERS: Dict[str, Provider] = {
    "google": Provider(
        "google",
        "https://accounts.google.com/o/oauth2/v2/auth",
        "https://oauth2.googleapis.com/token",
        "https://www.googleapis.com/auth/calendar.events",
        {"access_type": "offline", "prompt": "consent"},
    ),
    "outlook": Provider(
        "outlook",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "offline_access Calendars.ReadWrite",
        {},
    ),
}

_GOOGLE_EVENTS = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
_GRAPH_EVENTS = "https://graph.microsoft.com/v1.0/me/events"


class CalendarProviderError(Exception):
    pass


def _credentials(provider: str) -> tuple[str, str]:
    s = get_settings()
    if provider == "google":
        return s.google_client_id, s.google_client_secret
    return s.ms_client_id, s.ms_client_secret


def is_configured(provider: str) -> bool:
    if provider not in PROVIDERS:
        return False
    client_id, secret = _credentials(provider)
    return bool(client_id and secret)


def configured_providers() -> List[str]:
    return [p for p in PROVIDERS if is_configured(p)]


def redirect_uri(provider: str) -> str:
    return f"{get_settings().public_api_url.rstrip('/')}/api/calendar/sync/{provider}/callback"


def _fernet() -> Fernet:
    settings = get_settings()
    key = settings.token_enc_key or hashlib.sha256(
        (settings.jwt_secret or "dev_secret_key_123").encode()
    ).hexdigest()
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest()))


def encrypt_token(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def decrypt_token(value: str) -> str:
    return _fernet().decrypt(value.encode()).decode()


def build_auth_url(provider: str, state: str) -> str:
    p = PROVIDERS[provider]
    client_id, _ = _credentials(provider)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(provider),
        "response_type": "code",
        "scope": p.scope,
        "state": state,
        **p.extra_auth_params,
    }
    return f"{p.auth_url}?{urlencode(params)}"


def _token_request(provider: str, data: dict) -> dict:
    p = PROVIDERS[provider]
    client_id, secret = _credentials(provider)
    response = httpx.post(
        p.token_url,
        data={"client_id": client_id, "client_secret": secret, **data},
        timeout=15,
    )
    if response.status_code != 200:
        raise CalendarProviderError(f"{provider} token request failed ({response.status_code})")
    return response.json()


def exchange_code(provider: str, code: str) -> str:
    """Trade the authorization code for a refresh token."""
    payload = _token_request(
        provider,
        {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri(provider)},
    )
    refresh = payload.get("refresh_token")
    if not refresh:
        raise CalendarProviderError("The provider did not return a refresh token")
    return refresh


def access_token(provider: str, refresh_token: str) -> str:
    return _token_request(provider, {"grant_type": "refresh_token", "refresh_token": refresh_token})["access_token"]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def push_event(
    provider: str,
    token: str,
    title: str,
    start_at: datetime,
    end_at: Optional[datetime],
    existing_id: Optional[str],
) -> str:
    """Create (or update) the provider-side event; returns its id."""
    end = end_at or (start_at + timedelta(hours=1))
    headers = {"Authorization": f"Bearer {token}"}
    if provider == "google":
        body = {
            "summary": title,
            "start": {"dateTime": _iso_utc(start_at) + "Z"},
            "end": {"dateTime": _iso_utc(end) + "Z"},
        }
        url = f"{_GOOGLE_EVENTS}/{existing_id}" if existing_id else _GOOGLE_EVENTS
    else:
        body = {
            "subject": title,
            "start": {"dateTime": _iso_utc(start_at), "timeZone": "UTC"},
            "end": {"dateTime": _iso_utc(end), "timeZone": "UTC"},
        }
        url = f"{_GRAPH_EVENTS}/{existing_id}" if existing_id else _GRAPH_EVENTS
    method = "PATCH" if existing_id else "POST"
    response = httpx.request(method, url, headers=headers, json=body, timeout=15)
    if existing_id and response.status_code == 404:
        return push_event(provider, token, title, start_at, end_at, None)
    if response.status_code not in (200, 201):
        raise CalendarProviderError(f"{provider} event push failed ({response.status_code})")
    return response.json()["id"]


def delete_event(provider: str, token: str, event_id: str) -> None:
    url = f"{_GOOGLE_EVENTS}/{event_id}" if provider == "google" else f"{_GRAPH_EVENTS}/{event_id}"
    response = httpx.delete(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    if response.status_code not in (200, 204, 404, 410):
        raise CalendarProviderError(f"{provider} event delete failed ({response.status_code})")
