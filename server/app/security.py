"""Shared JWT secret helper used by app.routers.auth and app.deps.auth, so the secret-fallback logic can't drift between the two call sites."""

from app.config import get_settings

_settings = get_settings()
if not _settings.jwt_secret and not _settings.debug:


    raise RuntimeError(
        "JWT_SECRET is not set. Refusing to start with a known fallback "
        "secret. Set JWT_SECRET in server/.env, or set DEBUG=true to allow "
        "the insecure local-dev fallback."
    )


def jwt_secret() -> str:
    """Falls back to a known-insecure literal only when DEBUG=true (checked at import time above) — never in a demo/production configuration."""
    return get_settings().jwt_secret or "dev_secret_key_123"
