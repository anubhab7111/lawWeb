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

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from app.ingest.paths import case_law_dir
from app.tools import get_unified_rag_system
from app.tools.case_law_rag import get_case_law_rag_system

DOMAINS = {
    "unified": get_unified_rag_system,
    "case_law": get_case_law_rag_system,
}
REBUILD_DEVICE = "cuda"


async def rebuild_domain(domain: str):
    if domain not in DOMAINS:
        print(f"Error: Unknown domain '{domain}'")
        return

    print(f"\n[Rebuild] Processing domain: {domain}...")

    # 1. Get the system
    system = DOMAINS[domain]()

    # Sources may sit on a removable drive: refuse before deleting the old
    # index, or an unmounted drive would leave no index at all.
    source_dir = getattr(system, "_bare_acts_dir", None) or case_law_dir()
    if not Path(source_dir).is_dir():
        print(f"  Error: source directory {source_dir} not found (drive mounted?). Aborting.")
        return

    # 2. Identify the FAISS directory
    faiss_dir = _SERVER_DIR / "app" / "data" / "faiss_index" / domain

    # 3. Delete existing index if it exists
    if faiss_dir.exists():
        print(f"  Removing existing index at {faiss_dir}...")
        shutil.rmtree(faiss_dir)

    # 4. Force rebuild with a throwaway embedding model on CUDA. This bypasses
    # the query-time shared singleton, whose auto policy may use CPU to reserve
    # VRAM for Ollama.
    print(f"  Building new index from PDFs...")
    success = await system.build_offline(device=REBUILD_DEVICE)

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
        _require_cuda()
        for domain in DOMAINS:
            await rebuild_domain(domain)
    elif args.domain:
        _require_cuda()
        await rebuild_domain(args.domain)
    else:
        parser.print_help()


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "CUDA rebuild requires PyTorch in the legal_chatbot_env environment."
        ) from e

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable; refusing to rebuild on CPU. "
            "Check the NVIDIA driver and PyTorch CUDA installation."
        )

    print(
        f"[rebuild] Using CUDA on {torch.cuda.get_device_name(0)} "
        f"(PyTorch CUDA {torch.version.cuda})"
    )


if __name__ == "__main__":
    asyncio.run(main())
