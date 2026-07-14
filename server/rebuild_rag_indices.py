#!/usr/bin/env python3
"""
rebuild_rag_indices.py — CLI tool to force-rebuild FAISS/BM25 indexes for specific domains.
Usage:
    python rebuild_rag_indices.py --all
    python rebuild_rag_indices.py --domain constitutional
    python rebuild_rag_indices.py --domain criminal
"""

import asyncio
import argparse
import shutil
from pathlib import Path
import os
import sys

# Ensure server/ directory is in path and is the CWD, since
# BaseLegalRAGSystem resolves its data_dir ("app/data") relative to CWD.
_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from app.tools import (
    get_constitutional_rag_system,
    get_criminal_rag_system,
    get_civil_rag_system,
)

DOMAINS = {
    "constitutional": get_constitutional_rag_system,
    "criminal": get_criminal_rag_system,
    "civil": get_civil_rag_system,
}


async def rebuild_domain(domain: str):
    if domain not in DOMAINS:
        print(f"Error: Unknown domain '{domain}'")
        return

    print(f"\n[Rebuild] Processing domain: {domain}...")

    # 1. Get the system
    system = DOMAINS[domain]()

    # 2. Identify the FAISS directory
    faiss_dir = _SERVER_DIR / "app" / "data" / "faiss_index" / domain

    # 3. Delete existing index if it exists
    if faiss_dir.exists():
        print(f"  Removing existing index at {faiss_dir}...")
        shutil.rmtree(faiss_dir)

    # 4. Force rebuild via initialize
    # initialize() checks _should_rebuild(), which returns True if faiss_dir is missing
    print(f"  Building new index from PDFs...")
    success = await system.initialize()

    if success:
        print(
            f"  Successfully rebuilt '{domain}' index with {len(system._chunks)} chunks."
        )

        # Diagnostic for constitutional
        if domain == "constitutional":
            art19 = [c for c in system._chunks.values() if "19" in c.section_number]
            print(f"  Diagnostic: Article 19 has {len(art19)} chunks.")
    else:
        print(f"  Failed to rebuild '{domain}' index.")


async def main():
    parser = argparse.ArgumentParser(description="Rebuild legal RAG FAISS indexes.")
    parser.add_argument(
        "--domain",
        type=str,
        help="Specific domain to rebuild (e.g. constitutional, criminal)",
    )
    parser.add_argument("--all", action="store_true", help="Rebuild ALL domains")

    args = parser.parse_args()

    if args.all:
        for domain in DOMAINS:
            await rebuild_domain(domain)
    elif args.domain:
        await rebuild_domain(args.domain)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
