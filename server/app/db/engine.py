"""
Database engine and session handling for the PostgreSQL database.
"""

from functools import lru_cache

from sqlmodel import Session, create_engine

from app.config import get_settings


def _normalize_url(url: str) -> str:
    """Convert a Prisma-style DATABASE_URL to an SQLAlchemy one."""
    # Prisma URLs carry a ?schema=... query param that libpq rejects.
    base = url.split("?", 1)[0]
    return base.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache()
def get_engine():
    """Single shared engine for the whole server."""
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(_normalize_url(settings.database_url))


def get_session():
    """FastAPI dependency yielding a database session."""
    with Session(get_engine()) as session:
        yield session
