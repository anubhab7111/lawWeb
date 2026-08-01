"""
Firebase Cloud Messaging push sender. Genuinely stubbed pending a real
service-account credential AND the frontend service-worker/permission flow
neither of which exist yet — see the implementation plan's Hearing
Reminders risk notes. Logs instead of sending until both are wired up, so
the rest of notification_dispatch.py (DB bookkeeping, email fallback) is
still fully exercised in dev.
"""

from app.config import get_settings


def send_push(fcm_token: str, title: str, body: str) -> bool:
    settings = get_settings()
    if not settings.fcm_service_account_json or not fcm_token:
        print(f"[FcmClient] FCM not configured — would push to token …{fcm_token[-6:] if fcm_token else ''}: {title}")
        return False

    # TODO: real implementation needs google-auth + the FCM HTTP v1 API
    # (legacy server-key API is deprecated). Not wired up until a service
    # account is provisioned and the frontend requests push permission.
    print(f"[FcmClient] push send not yet implemented — logging only: {title}")
    return False
