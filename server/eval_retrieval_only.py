#!/usr/bin/env python3
"""
eval_retrieval_only.py — LLM-free retrieval evaluation.

Scores the production retrieval path (unified hybrid index: BM25 + dense +
reranker) against the ground-truth dataset with Hit Rate@1/3/5 and MRR.
No Ollama required, so it runs even when the LLM doesn't fit in memory —
useful for fast retrieval-tuning iterations and before/after comparisons.

Usage (conda env legal_chatbot_env):
    python eval_retrieval_only.py              # original 18 queries
    python eval_retrieval_only.py --extended   # all queries incl. new domains
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from app.metrics.ground_truth import (
    GROUND_TRUTH,
    relevant_sections_for,
)
from app.metrics.retrieval_metrics import compute_hit_rate, compute_mrr_single
from app.tools.legal_retrieval import retrieve_statutes
from app.tools.unified_legal_rag import get_unified_rag_system

ORIGINAL_COUNT = 18  # first N entries predate the extended set


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="all queries")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    entries = [e for e in GROUND_TRUTH if relevant_sections_for(e)]
    if not args.extended:
        original_queries = {e["query"] for e in GROUND_TRUTH[:ORIGINAL_COUNT]}
        entries = [e for e in entries if e["query"] in original_queries]

    rag = get_unified_rag_system()
    if not await rag.initialize():
        print("FATAL: unified RAG failed to initialize")
        sys.exit(1)

    rows = []
    for entry in entries:
        result, _parsed = await retrieve_statutes(entry["query"], k=args.k)
        seen: set = set()
        sections = []
        for c in result.chunks:
            sec = c.section_number.replace("Article", "").strip()
            if sec and sec not in seen:
                seen.add(sec)
                sections.append(sec)
        relevant = relevant_sections_for(entry)
        relevant = [s.replace("Article", "").strip() for s in relevant]
        rows.append(
            {
                "domain": entry["domain"],
                "query": entry["query"],
                "retrieved": sections,
                "relevant": relevant,
                "hr1": compute_hit_rate(sections, relevant, k=1),
                "hr3": compute_hit_rate(sections, relevant, k=3),
                "hr5": compute_hit_rate(sections, relevant, k=5),
                "rr": compute_mrr_single(sections, relevant),
            }
        )

    n = len(rows)
    print(f"\n{'=' * 72}")
    print(f"  RETRIEVAL-ONLY EVAL — {n} queries with section ground truth")
    print(f"{'=' * 72}")
    for r in rows:
        flag = "HIT " if r["hr5"] else "MISS"
        print(
            f"  {flag} rr={r['rr']:.2f} [{r['domain']:<18}] "
            f"{r['query'][:52]}..."
        )
        print(f"        retrieved={r['retrieved']}  relevant={r['relevant']}")
    print(f"{'-' * 72}")
    print(f"  Hit Rate @ 1 : {sum(r['hr1'] for r in rows) / n:.3f}")
    print(f"  Hit Rate @ 3 : {sum(r['hr3'] for r in rows) / n:.3f}")
    print(f"  Hit Rate @ 5 : {sum(r['hr5'] for r in rows) / n:.3f}")
    print(f"  MRR          : {sum(r['rr'] for r in rows) / n:.3f}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())
