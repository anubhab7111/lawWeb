"""
Firebase Cloud Messaging push sender (HTTP v1 API).

Credentials come from FCM_SERVICE_ACCOUNT_JSON — either a path to the service
account file or the JSON itself. When unset, push is simply unavailable
(`push_configured()` is False and the client hides the option).
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_settings

_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"


class PushTokenInvalid(Exception):
    """FCM says this device token is gone (uninstalled / permission revoked)."""


def push_configured() -> bool:
    return bool(get_settings().fcm_service_account_json.strip())


def _load_service_account_info() -> dict:
    raw = get_settings().fcm_service_account_json.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    return json.loads(Path(raw).read_text())


@lru_cache()
def _credentials():
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_info(
        _load_service_account_info(), scopes=[_SCOPE]
    )


def _access_token() -> str:
    from google.auth.transport.requests import Request

    creds = _credentials()
    if not creds.valid:
        creds.refresh(Request())
    return creds.token


def send_push(fcm_token: str, title: str, body: str, link: Optional[str] = None) -> bool:
    """Send one notification. Returns True on success, False on a transient
    failure; raises PushTokenInvalid when the token should be discarded."""
    if not push_configured() or not fcm_token:
        return False

    creds = _credentials()
    message: dict = {
        "token": fcm_token,
        "notification": {"title": title, "body": body},
    }
    if link:
        message["webpush"] = {"fcm_options": {"link": link}}

    try:
        response = httpx.post(
            f"https://fcm.googleapis.com/v1/projects/{creds.project_id}/messages:send",
            headers={"Authorization": f"Bearer {_access_token()}"},
            json={"message": message},
            timeout=10,
        )
    except Exception as e:
        print(f"[FcmClient] push failed: {e}")
        return False

    if response.status_code == 200:
        return True
    if response.status_code == 404 or "UNREGISTERED" in response.text:
        raise PushTokenInvalid(fcm_token)
    print(f"[FcmClient] push rejected ({response.status_code}): {response.text[:200]}")
    return False
