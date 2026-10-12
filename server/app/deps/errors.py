"""Shared error helpers so new routers return the `{"message": ...}` body shape the client already expects from auth/lawyers/bookings, instead of FastAPI's default `{"detail":..."""

from fastapi import HTTPException


class MessageHTTPException(HTTPException):
    """HTTPException whose handler (registered in app.main) unwraps `detail` into a top-level `message` key."""
