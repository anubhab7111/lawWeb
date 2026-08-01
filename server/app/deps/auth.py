"""
Shared per-request auth dependency. Before this module, only GET /api/auth/me
verified the caller's JWT — every other route trusted a raw userId/lawyerId
field in the request body with no verification at all. New protected routes
should depend on get_current_user instead of repeating that pattern.
"""

from typing import Optional

import jwt
from fastapi import Depends, Header
from sqlmodel import Session

from app.db.engine import get_session
from app.db.models import User
from app.deps.errors import MessageHTTPException
from app.security import jwt_secret


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    token = (
        authorization.split(" ")[1]
        if authorization and " " in authorization
        else None
    )
    if not token:
        raise MessageHTTPException(status_code=401, detail="No token provided")

    try:
        decoded = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        raise MessageHTTPException(status_code=401, detail="Invalid token")

    user = session.get(User, decoded.get("id"))
    if not user:
        raise MessageHTTPException(status_code=401, detail="User not found")

    return user


def get_current_user_optional(
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """Same as get_current_user but returns None instead of raising, for
    routes that behave differently for anonymous vs. logged-in callers
    (e.g. attaching search history only when a user is present)."""
    if not authorization:
        return None
    try:
        return get_current_user(authorization=authorization, session=session)
    except MessageHTTPException:
        return None
