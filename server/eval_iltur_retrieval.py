#!/usr/bin/env python3
"""
eval_iltur_retrieval.py — LLM-free retrieval evaluation on IL-TUR `lsi`.

Draws a seeded sample of IL-TUR cases and scores three section rankings against
each case's label set, by section number:
  statutes   per-window statute retrieval (production `retrieve_statutes`),
             fused across the case's fact windows with RRF
  precedent  similar past IL-TUR train+dev cases voting for their sections
             (app/tools/precedent_rag.py)
  fused      weighted RRF of the two
Reports Hit@k, Recall@k, MRR, micro-F1@k, plus label coverage of the index.

Tune on --split dev (each queried dev case is masked out of the precedent
index); report on --split test, which is never indexed.

Usage (conda env legal_chatbot_env, from server/):
    python eval_iltur_retrieval.py --tag baseline                    # test, all systems
    python eval_iltur_retrieval.py --split dev --skip-statutes       # fast precedent tuning
    python eval_iltur_retrieval.py --sample-size 20 --tag smoke
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
from app.tools.precedent_rag import get_precedent_index, retrieve_precedent_sections  # noqa: E402
from app.tools.unified_legal_rag import get_unified_rag_system  # noqa: E402

KS = (1, 3, 5, 10)
SAVED_DEPTH = 30


def is_ipc(act_name: str) -> bool:
    """IL-TUR lsi labels are IPC sections only (verified: every label's statute
    text matches the IPC provision), so only IPC chunks can score."""
    name = act_name.lower()
    return "penal code" in name and "bharatiya" not in name


def section_key(section_number: str) -> str:
    return section_number.replace("Article", "").strip().upper()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_SERVER_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


async def rank_statutes(sentences, args):
    windows = fact_windows(
        sentences, window=args.window, stride=args.stride, max_windows=args.max_windows
    )
    rankings = []
    for window in windows:
        result, _parsed = await retrieve_statutes(window, k=args.k_window)
        ranking = []
        for chunk in result.chunks:
            if not is_ipc(chunk.act_name):
                continue
            key = section_key(chunk.section_number)
            if key and key not in ranking:
                ranking.append(key)
        rankings.append(ranking)
    return rrf_fuse(rankings)


async def rank_precedent(row, sentences, args):
    exclude = None if args.split == "test" else [str(row["id"])]
    voted = await retrieve_precedent_sections(
        sentences, k=args.precedent_k, power=args.precedent_power, exclude_case_ids=exclude
    )
    return [section for section, _score in voted]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=4, help="sentences per window")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--k-window", type=int, default=10, help="statute retrieval depth per window")
    parser.add_argument("--precedent-k", type=int, default=20, help="neighbour windows per query window")
    parser.add_argument("--precedent-power", type=float, default=8.0, help="similarity exponent for votes")
    parser.add_argument("--fusion-weights", type=float, nargs=2, default=[1.0, 1.0], metavar=("STATUTE", "PRECEDENT"))
    parser.add_argument("--skip-statutes", action="store_true", help="precedent only (fast; for tuning)")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    use_precedent = get_precedent_index().available
    if not use_precedent:
        print("[precedent] index not built — evaluating statutes only")
    use_statutes = not args.skip_statutes
    if not use_statutes and not use_precedent:
        sys.exit("nothing to evaluate")

    names = label_names()
    coverage = None
    if use_statutes:
        rag = get_unified_rag_system()
        if not await rag.initialize():
            sys.exit("FATAL: unified RAG failed to initialize")
        indexed_numbers = [
            section_key(c.section_number) for c in rag._chunks.values() if is_ipc(c.act_name)
        ]
        coverage = label_coverage(decode_labels(list(range(len(names))), names), indexed_numbers)
        print(
            f"[coverage] {coverage['covered']}/{coverage['labels']} IL-TUR label sections "
            f"resolvable in the index; missing: {coverage['missing']}"
        )

    rows = sample_iltur_cases(args.sample_size, seed=args.seed, split=args.split)
    per_system = {}
    details = []
    gold_sizes = []
    started = time.time()
    for i, row in enumerate(rows, start=1):
        sentences = row["text"] if isinstance(row["text"], list) else [row["text"]]
        gold = decode_labels(row["labels"], names)
        gold_sizes.append(len(gold))

        ranks = {}
        if use_statutes:
            ranks["statutes"] = await rank_statutes(sentences, args)
        if use_precedent:
            ranks["precedent"] = await rank_precedent(row, sentences, args)
        if len(ranks) == 2:
            ranks["fused"] = rrf_fuse(
                [ranks["statutes"], ranks["precedent"]], weights=args.fusion_weights
            )

        entry = {"id": row.get("id"), "gold": gold}
        for system, ranked in ranks.items():
            metrics = case_metrics(ranked, gold, ks=KS)
            per_system.setdefault(system, []).append(metrics)
            entry[system] = {"top10": ranked[:10], **metrics}
        entry["saved_rankings"] = {s: r[:SAVED_DEPTH] for s, r in ranks.items() if s != "fused"}
        details.append(entry)

        if i % 10 == 0 or i == len(rows):
            progress = " ".join(
                f"{s}={sum(c['hit@5'] for c in m) / i:.3f}" for s, m in per_system.items()
            )
            print(f"[{i}/{len(rows)}] {time.time() - started:.0f}s  hit@5: {progress}")

    summary = {s: aggregate(m, gold_sizes, ks=KS) for s, m in per_system.items()}
    report = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "config": vars(args),
        "label_coverage": coverage,
        "summary": summary,
        "cases": details,
    }

    n = len(rows)
    print("\n" + "=" * 72)
    print(f" IL-TUR retrieval ({args.tag}) — {args.split} n={n} seed={args.seed}")
    print("=" * 72)
    print(f"  {'system':<10}" + "".join(f"hit@{k:<3}" for k in KS) + "  MRR    F1@5   recall@10")
    for system, agg in summary.items():
        print(
            f"  {system:<10}"
            + "".join(f"{agg[f'hit@{k}']:<7.3f}" for k in KS)
            + f"  {agg['mrr']:<6.3f} {agg['micro_f1@5']:<6.3f} {agg['recall@10']:.3f}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"iltur_retrieval_{args.tag}_{args.split}_{stamp}.json"
    out_dirs = [_SERVER_DIR / "results"]
    if corpus_available():
        out_dirs.append(corpus_path("builds", "eval"))
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(json.dumps(report, indent=1))
        print(f"  saved {out_dir / name}")


if __name__ == "__main__":
    asyncio.run(main())
