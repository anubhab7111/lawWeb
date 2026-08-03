"""
Shared JWT secret helper used by app.routers.auth and app.deps.auth, so the
secret-fallback logic can't drift between the two call sites.
"""

from app.config import get_settings


def jwt_secret() -> str:
    """Same fallback the old Express server used, so existing tokens stay
    valid even when JWT_SECRET is unset in dev."""
    return get_settings().jwt_secret or "dev_secret_key_123"
