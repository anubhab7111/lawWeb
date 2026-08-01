"""
Initialize the database: create tables from schema.sql if they are missing,
then seed the demo lawyers. Idempotent — safe to run repeatedly.

Usage: python -m app.db.init_db
"""

from pathlib import Path

from sqlalchemy import inspect
from sqlmodel import Session

from app.db.engine import get_engine
from app.db.migrations import run_migrations
from app.db.models import Lawyer

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

# Legacy mock ids '1'..'5' kept as primary keys so the client's lawyer
# routes keep working unchanged.
SEED_LAWYERS = [
    Lawyer(
        id="1",
        name="Sarah Mitchell",
        specialty="Criminal Defense",
        experience=15,
        rating=4.9,
        hourly_rate=350,
        location="New York, NY",
        bio="Experienced criminal defense attorney with a proven track record in white-collar crimes, DUI cases, and federal offenses.",
        cases=450,
        success_rate=92,
        education="Harvard Law School, JD",
        languages=["English", "Spanish"],
        availability="Available this week",
    ),
    Lawyer(
        id="2",
        name="David Chen",
        specialty="Family Law",
        experience=12,
        rating=4.8,
        hourly_rate=275,
        location="Los Angeles, CA",
        bio="Compassionate family lawyer specializing in divorce, child custody, and adoption cases with a focus on mediation.",
        cases=380,
        success_rate=88,
        education="Stanford Law School, JD",
        languages=["English", "Mandarin"],
        availability="Available next week",
    ),
    Lawyer(
        id="3",
        name="Jennifer Rodriguez",
        specialty="Business & Corporate Law",
        experience=18,
        rating=4.9,
        hourly_rate=425,
        location="Chicago, IL",
        bio="Corporate attorney specializing in mergers, acquisitions, and business formation with Fortune 500 experience.",
        cases=520,
        success_rate=95,
        education="Yale Law School, JD",
        languages=["English", "Spanish"],
        availability="Available in 2 weeks",
    ),
    Lawyer(
        id="4",
        name="Michael Thompson",
        specialty="Personal Injury",
        experience=10,
        rating=4.7,
        hourly_rate=300,
        location="Houston, TX",
        bio="Dedicated personal injury lawyer fighting for accident victims and securing maximum compensation.",
        cases=340,
        success_rate=90,
        education="University of Texas Law School, JD",
        languages=["English"],
        availability="Available this week",
    ),
    Lawyer(
        id="5",
        name="Emily Watson",
        specialty="Real Estate Law",
        experience=14,
        rating=4.8,
        hourly_rate=325,
        location="Miami, FL",
        bio="Real estate attorney handling residential and commercial transactions, zoning issues, and property disputes.",
        cases=410,
        success_rate=93,
        education="Columbia Law School, JD",
        languages=["English", "Portuguese"],
        availability="Available next week",
    ),
]


def init_db() -> None:
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
        seeded = 0
        for lawyer in SEED_LAWYERS:
            if session.get(Lawyer, lawyer.id) is None:
                session.add(lawyer)
                seeded += 1
        session.commit()
        print(f"Seeded {seeded} lawyers ({len(SEED_LAWYERS) - seeded} already present)")


if __name__ == "__main__":
    init_db()
