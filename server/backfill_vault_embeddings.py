#!/usr/bin/env python3
"""
backfill_vault_embeddings.py — re-embed Legal Document Vault chunks.

Vault chunk vectors live in vault_document_embeddings and are produced by the
shared embedding model. After switching that model (e.g. to BGE-M3) the stored
vectors are incompatible with new query vectors, so every indexed document must
be re-embedded from its already-extracted text (no re-download / re-OCR).

Usage:
    python backfill_vault_embeddings.py          # docs with no embeddings yet
    python backfill_vault_embeddings.py --all    # re-embed EVERY indexed doc
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from sqlmodel import Session, delete, select

from app.db.engine import get_engine
from app.db.models import VaultDocument, VaultDocumentEmbedding
from app.services.vault_indexer import index_vault_document


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill vault document embeddings.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-embed every indexed document (use after an embedding-model change).",
    )
    args = parser.parse_args()

    engine = get_engine()
    with Session(engine) as session:
        docs = session.exec(
            select(VaultDocument).where(VaultDocument.extracted_text.is_not(None))
        ).all()

        # Capture id + text as plain values now, while the session is live:
        # index_vault_document runs after this block closes, and session.commit()
        # below expires ORM attributes, so reading doc.* later would raise
        # DetachedInstanceError.
        targets: list[tuple[str, str]] = []
        for doc in docs:
            has_vectors = session.exec(
                select(VaultDocumentEmbedding.id)
                .where(VaultDocumentEmbedding.vault_document_id == doc.id)
                .limit(1)
            ).first()
            if args.all or not has_vectors:
                targets.append((doc.id, doc.extracted_text or ""))

        if not targets:
            print("No vault documents need embedding backfill.")
            return

        print(f"Re-embedding {len(targets)} vault document(s)...")
        # Clear stale vectors first — index_vault_document appends, so without
        # this a --all run would duplicate chunks.
        for doc_id, _text in targets:
            session.exec(
                delete(VaultDocumentEmbedding).where(
                    VaultDocumentEmbedding.vault_document_id == doc_id
                )
            )
        session.commit()

    # index_vault_document opens its own session per document.
    for doc_id, text in targets:
        await index_vault_document(doc_id, text)

    print(f"Backfilled {len(targets)} vault document(s).")


if __name__ == "__main__":
    asyncio.run(main())
