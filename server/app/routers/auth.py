"""
Authentication endpoints (register/login/me), ported from the old Express
server/src/routes/auth.ts. Response and error shapes are preserved exactly:
the client reads `message` from error bodies, not FastAPI's default `detail`.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.db.engine import get_session
from app.db.models import User
from app.security import jwt_secret as _jwt_secret

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"message": message})


# bcrypt hard-rejects (ValueError) any password whose UTF-8 encoding exceeds
# 72 bytes — both hashpw and checkpw. Checked explicitly so an over-length
# password is a normal 400, not an unhandled 500.
_BCRYPT_MAX_BYTES = 72


def _make_token(user_id: str) -> str:
    payload = {
        "id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _user_json(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email}


class RegisterRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None


class LoginRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None


@router.post("/register")
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    if not body.name or not body.email or not body.password:
        return _error(400, "All fields are required")
    if len(body.password.encode()) > _BCRYPT_MAX_BYTES:
        return _error(400, "Password must be 72 bytes or fewer")

    existing = session.exec(select(User).where(User.email == body.email)).first()
    if existing:
        return _error(400, "User already exists")

    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=10)).decode()
    user = User(name=body.name, email=body.email, password=hashed)
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        # Unique-constraint race between the existence check and the insert
        return _error(400, "User already exists")
    session.refresh(user)

    return JSONResponse(
        status_code=201,
        content={"token": _make_token(user.id), "user": _user_json(user)},
    )


@router.post("/login")
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if not user:
        return _error(400, "Invalid credentials")

    password_bytes = (body.password or "").encode()
    if len(password_bytes) > _BCRYPT_MAX_BYTES:
        # Can't possibly match a hash of a <=72-byte password; treat like
        # any other wrong password rather than crashing checkpw on it.
        return _error(400, "Invalid credentials")
    if not bcrypt.checkpw(password_bytes, user.password.encode()):
        return _error(400, "Invalid credentials")

    return {"token": _make_token(user.id), "user": _user_json(user)}


@router.get("/me")
def me(
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
):
    token = authorization.split(" ")[1] if authorization and " " in authorization else None
    if not token:
        return _error(401, "No token provided")

    try:
        decoded = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return _error(401, "Invalid token")

    user = session.get(User, decoded.get("id"))
    if not user:
        return _error(404, "User not found")

    return _user_json(user)
