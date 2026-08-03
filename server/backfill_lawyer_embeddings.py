#!/usr/bin/env python3
"""
backfill_lawyer_embeddings.py — compute bio_embedding for lawyers that don't
have one yet (e.g. after adding the pgvector column, or after seeding new
demo lawyers).

Usage:
    python backfill_lawyer_embeddings.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure server/ directory is in path and is the CWD, since config/data paths
# resolve relative to CWD elsewhere in this codebase.
_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import Lawyer
from app.tools.lawyer_recommender import embed_lawyers_batch


async def main() -> None:
    engine = get_engine()
    with Session(engine) as session:
        lawyers = session.exec(
            select(Lawyer).where(Lawyer.bio_embedding.is_(None))
        ).all()

        if not lawyers:
            print("No lawyers need embedding backfill.")
            return

        print(f"Embedding {len(lawyers)} lawyer(s)...")
        pairs = [(lawyer.specialty, lawyer.bio) for lawyer in lawyers]
        vectors = await embed_lawyers_batch(pairs)

        for lawyer, vector in zip(lawyers, vectors):
            lawyer.bio_embedding = vector
            session.add(lawyer)

        session.commit()
        print(f"Backfilled {len(lawyers)} lawyer(s).")


if __name__ == "__main__":
    asyncio.run(main())
