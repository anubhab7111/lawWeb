"""
Initialize the database: create tables from schema.sql if they are missing,
then seed the Indian lawyer directory (app/data/lawyers.json) and embed their bios. Idempotent — safe to run repeatedly.

Usage: python -m app.db.init_db [--skip-embeddings]
"""

import json
import sys
from pathlib import Path

from sqlalchemy import inspect
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.migrations import run_migrations
from app.db.models import Lawyer

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

LAWYERS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "lawyers.json"


def _load_fixture() -> list[Lawyer]:
    rows = json.loads(LAWYERS_FIXTURE.read_text(encoding="utf-8"))
    return [
        Lawyer(
            id=r["id"],
            name=r["name"],
            specialty=r["specialty"],
            experience=r["experience"],
            rating=r["rating"],
            hourly_rate=r["hourlyRate"],
            location=r["location"],
            bio=r["bio"],
            cases=r["cases"],
            success_rate=r["successRate"],
            education=r["education"],
            languages=r["languages"],
            availability=r["availability"],
        )
        for r in rows
    ]


def _embed_missing_lawyers(engine) -> None:
    import asyncio

    from app.tools.lawyer_recommender import embed_lawyers_batch

    with Session(engine) as session:
        lawyers = session.exec(select(Lawyer).where(Lawyer.bio_embedding.is_(None))).all()
        if not lawyers:
            return
        print(f"Embedding {len(lawyers)} lawyer bios (skip with --skip-embeddings)...")
        vectors = asyncio.run(embed_lawyers_batch([(l.specialty, l.bio) for l in lawyers]))
        for lawyer, vector in zip(lawyers, vectors):
            lawyer.bio_embedding = vector
            session.add(lawyer)
        session.commit()


def init_db(embed: bool = True) -> None:
    engine = get_engine()

    if not inspect(engine).has_table("users"):
        sql = SCHEMA_FILE.read_text()
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)
        print("Created tables from schema.sql")
    else:
        print("Tables already exist, skipping schema.sql")

    # Incremental changes for existing dev DBs — see app/db/migrations.py
    # for the convention (there is no Alembic in this project).
    run_migrations(engine)

    with Session(engine) as session:
        existing = set(session.exec(select(Lawyer.id)).all())
        fresh = [l for l in _load_fixture() if l.id not in existing]
        session.add_all(fresh)
        session.commit()
        print(f"Seeded {len(fresh)} lawyers ({len(existing)} already present)")

    if embed:
        _embed_missing_lawyers(engine)


if __name__ == "__main__":
    init_db(embed="--skip-embeddings" not in sys.argv)
