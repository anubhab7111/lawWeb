#!/usr/bin/env python3
"""
eval_iltur_retrieval.py — LLM-free retrieval evaluation on IL-TUR `lsi`.

Draws a seeded sample of IL-TUR test cases, splits each case's long facts into
sentence windows, retrieves statutes per window with the production path
(`retrieve_statutes`), fuses the per-window rankings (RRF) and scores the fused
IPC/CrPC section ranking against the case's label set by section number.
Reports Hit@k, Recall@k, MRR, micro-F1@k, and how many of IL-TUR's 100 label
sections the index can resolve at all (label coverage).

Usage (conda env legal_chatbot_env, from server/):
    python eval_iltur_retrieval.py --tag baseline
    python eval_iltur_retrieval.py --sample-size 20 --tag smoke     # quick check
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from dotenv import load_dotenv

load_dotenv(_SERVER_DIR / ".env")

from app.ingest.paths import corpus_available, corpus_path  # noqa: E402
from app.metrics.iltur_eval import (  # noqa: E402
    aggregate,
    case_metrics,
    fact_windows,
    label_coverage,
    rrf_fuse,
)
from app.metrics.iltur_loader import decode_labels, label_names, sample_iltur_cases  # noqa: E402
from app.tools.legal_retrieval import retrieve_statutes  # noqa: E402
from app.tools.unified_legal_rag import get_unified_rag_system  # noqa: E402

KS = (1, 3, 5, 10)


def is_ipc_or_crpc(act_name: str) -> bool:
    name = act_name.lower()
    return "penal code" in name or "criminal procedure" in name


def section_key(section_number: str) -> str:
    return section_number.replace("Article", "").strip().upper()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_SERVER_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


async def rank_case(row, args):
    sentences = row["text"] if isinstance(row["text"], list) else [row["text"]]
    windows = fact_windows(
        sentences, window=args.window, stride=args.stride, max_windows=args.max_windows
    )
    rankings = []
    for window in windows:
        result, _parsed = await retrieve_statutes(window, k=args.k_window)
        ranking = []
        for chunk in result.chunks:
            if not is_ipc_or_crpc(chunk.act_name):
                continue
            key = section_key(chunk.section_number)
            if key and key not in ranking:
                ranking.append(key)
        rankings.append(ranking)
    return rrf_fuse(rankings), len(windows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=4, help="sentences per window")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--k-window", type=int, default=10, help="retrieval depth per window")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    rag = get_unified_rag_system()
    if not await rag.initialize():
        sys.exit("FATAL: unified RAG failed to initialize")

    names = label_names()
    all_labels = decode_labels(list(range(len(names))), names)
    indexed_numbers = [
        section_key(c.section_number) for c in rag._chunks.values() if is_ipc_or_crpc(c.act_name)
    ]
    coverage = label_coverage(all_labels, indexed_numbers)
    print(
        f"[coverage] {coverage['covered']}/{coverage['labels']} IL-TUR label sections "
        f"resolvable in the index; missing: {coverage['missing']}"
    )

    rows = sample_iltur_cases(args.sample_size, seed=args.seed)
    per_case, details, gold_sizes = [], [], []
    started = time.time()
    for i, row in enumerate(rows, start=1):
        gold = decode_labels(row["labels"], names)
        ranked, n_windows = await rank_case(row, args)
        metrics = case_metrics(ranked, gold, ks=KS)
        per_case.append(metrics)
        gold_sizes.append(len(gold))
        details.append(
            {"id": row.get("id"), "gold": gold, "windows": n_windows, "top10": ranked[:10], **metrics}
        )
        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - started
            print(
                f"[{i}/{len(rows)}] {elapsed:.0f}s elapsed, "
                f"hit@5 so far {sum(c['hit@5'] for c in per_case) / i:.3f}"
            )

    summary = aggregate(per_case, gold_sizes, ks=KS)
    report = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "config": vars(args),
        "label_coverage": coverage,
        "summary": summary,
        "cases": details,
    }

    print("\n" + "=" * 60)
    print(f" IL-TUR retrieval ({args.tag}) — n={summary['n']} seed={args.seed}")
    print("=" * 60)
    for k in KS:
        print(
            f"  @{k:<2} hit {summary[f'hit@{k}']:.3f}  recall {summary[f'recall@{k}']:.3f}"
            f"  micro-F1 {summary[f'micro_f1@{k}']:.3f}"
        )
    print(f"  MRR {summary['mrr']:.3f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"iltur_retrieval_{args.tag}_{stamp}.json"
    out_dirs = [_SERVER_DIR / "results"]
    if corpus_available():
        out_dirs.append(corpus_path("builds", "eval"))
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(json.dumps(report, indent=1))
        print(f"  saved {out_dir / name}")


if __name__ == "__main__":
    asyncio.run(main())
