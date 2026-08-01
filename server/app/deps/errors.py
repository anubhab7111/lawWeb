"""
Shared error helpers so new routers return the `{"message": ...}` body shape
the client already expects from auth/lawyers/bookings, instead of FastAPI's
default `{"detail": ...}` shape.
"""

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class MessageHTTPException(HTTPException):
    """HTTPException whose handler (registered in app.main) unwraps `detail`
    into a top-level `message` key. Scoped to this exception type only, so it
    doesn't change the response shape of existing HTTPException usage
    elsewhere (e.g. app.routers.chat)."""


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"message": message})
