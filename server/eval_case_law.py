#!/usr/bin/env python3
"""
eval_case_law.py — lightweight retrieval-quality check for the case-law
index, using ground truth that already exists for free: each doctrine in
app/data/legal_ontology.json lists its own landmark_cases (see
app/data/case_law_manifest.py::build_manifest()). For every doctrine with
at least one landmark case, query the case-law RAG with the doctrine name
and check whether that doctrine's landmark case(s) land in the top-k.

Reuses compute_hit_rate/compute_mrr_single verbatim (same functions
eval_retrieval_only.py uses for the bare-acts index) — no new metric code.

Usage (conda env legal_chatbot_env, run from server/):
    python eval_case_law.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from app.data.case_law_manifest import build_manifest
from app.metrics.retrieval_metrics import compute_hit_rate, compute_mrr_single
from app.tools.case_law_rag import get_case_law_rag_system

K = 5


async def main() -> None:
    rag = get_case_law_rag_system()
    if not await rag.initialize():
        print("FATAL: case-law RAG failed to initialize")
        sys.exit(1)

    manifest = build_manifest()
    hits = []
    rrs = []
    rows = []

    for entry in manifest:
        doctrine_names = entry["doctrines"]
        relevant = [entry["case_name"]]
        query = " ".join(doctrine_names) if doctrine_names else entry["case_name"]

        results = await rag.retrieve(query, k=K)
        retrieved = [r.case_name for r in results]

        hr = compute_hit_rate(retrieved, relevant, k=K)
        rr = compute_mrr_single(retrieved, relevant)
        hits.append(hr)
        rrs.append(rr)
        rows.append(
            {
                "doctrine_query": query,
                "target_case": entry["case_name"],
                "hit_at_5": hr,
                "reciprocal_rank": rr,
                "top_retrieved": retrieved[:3],
            }
        )
        status = "HIT" if hr else "miss"
        print(f"[{status}] {query!r} -> target {entry['case_name']!r}; got {retrieved[:3]}")

    hit_rate_at_5 = sum(hits) / len(hits) if hits else 0.0
    mrr = sum(rrs) / len(rrs) if rrs else 0.0

    print(f"\nn_queries={len(manifest)}  hit_rate@{K}={hit_rate_at_5:.3f}  mrr={mrr:.3f}")

    out_path = _SERVER_DIR / "eval_case_law_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"hit_rate_at_5": hit_rate_at_5, "mrr": mrr, "rows": rows},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved detail to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
