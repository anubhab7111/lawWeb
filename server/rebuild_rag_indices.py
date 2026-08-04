#!/usr/bin/env python3
"""
rebuild_rag_indices.py — CLI tool to force-rebuild FAISS indexes for specific domains.
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

from app.tools import get_unified_rag_system
from app.tools.case_law_rag import get_case_law_rag_system

# All bare-act domains live in the single unified index; the old per-domain
# indexes (criminal/civil/constitutional) are thin adapters over it now.
# case_law is a separate index (Supreme Court/HC judgments) — included here so
# `--all` rebuilds it too (e.g. after an embedding-model swap), rather than
# relying solely on the startup build path.
DOMAINS = {
    "unified": get_unified_rag_system,
    "case_law": get_case_law_rag_system,
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
        # Bare-act systems hold `_chunks`; the case-law system holds `_cases`.
        indexed = getattr(system, "_chunks", None)
        if indexed is None:
            indexed = getattr(system, "_cases", None) or {}
        print(f"  Successfully rebuilt '{domain}' index with {len(indexed)} item(s).")

        # Diagnostic: per-domain chunk counts in the unified index
        if domain == "unified":
            counts = {}
            for c in system._chunks.values():
                counts[c.domain] = counts.get(c.domain, 0) + 1
            for d in sorted(counts):
                print(f"    {d}: {counts[d]} chunks")
    else:
        print(f"  Failed to rebuild '{domain}' index.")

    # Each domain's embedding model stays resident in GPU/CPU memory for the
    # life of this process (systems are cached singletons), and the FAISS
    # vector_store also holds its own reference to the embeddings object —
    # both must be dropped (and garbage-collected) before the CUDA cache
    # actually frees up, or the next domain has no headroom.
    system.embeddings = None
    system.vector_store = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
