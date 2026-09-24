#!/usr/bin/env python3
"""
tune_iltur_retrieval.py — cache raw retrieval output for offline tuning.

Full evaluations re-run every retrieval (~40 min per 150 cases). Tuning how the
per-window results are *aggregated* only needs each window's raw candidates, so
this records them once, for a seeded sample of IL-TUR **dev** cases:

  statutes    the raw `retrieve_statutes` output per window across the scoped Acts
              (so per-Act weights, depth and BNS->IPC translation can be tried offline)
  precedent   the precedent-index votes per case, with the case masked out

The cache goes to <corpus>/builds/tune/.

Usage (from server/):
    python tune_iltur_retrieval.py cache --sample-size 150 --tag dev150 \\
        --acts "Indian Penal Code" "Code of Criminal Procedure" "Bharatiya Nyaya Sanhita BNS" \\
        --min-score 0.0 --candidate-pool 100 --rerank-pool 60 --statute-depth 60
    python tune_iltur_retrieval.py cache --skip-statutes --precedent      # precedent only (fast)
"""

import argparse
import asyncio
import os
import pickle
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from dotenv import load_dotenv

load_dotenv(_SERVER_DIR / ".env")

from app.ingest.paths import corpus_path  # noqa: E402
from app.metrics.iltur_eval import fact_windows  # noqa: E402
from app.metrics.iltur_loader import decode_labels, label_names, sample_iltur_cases  # noqa: E402
from app.tools.legal_retrieval import retrieve_statutes  # noqa: E402
from app.tools.precedent_rag import retrieve_precedent_sections  # noqa: E402


def cache_path(tag: str, part: str) -> Path:
    return corpus_path("builds", "tune", f"{tag}.{part}.pkl")


async def build_statutes(rows, args):
    cases = []
    started = time.time()
    for i, row in enumerate(rows, start=1):
        windows = fact_windows(row["text"], window=4, stride=3, max_windows=args.max_windows)
        per_window = []
        for window in windows:
            result, _ = await retrieve_statutes(
                window,
                k=args.statute_depth,
                acts=args.acts or None,
                min_score=args.min_score,
                candidate_pool=args.candidate_pool,
                rerank_pool=args.rerank_pool,
            )
            per_window.append(
                [(c.act_name, c.section_number.replace("Article", "").strip().upper(), round(c.score, 4)) for c in result.chunks]
            )
        cases.append({"id": str(row["id"]), "per_window": per_window})
        if i % 10 == 0:
            print(f"  statutes {i}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    return {"cases": cases}


async def build_precedent(rows):
    out = {}
    for i, row in enumerate(rows, start=1):
        sentences = row["text"] if isinstance(row["text"], list) else [row["text"]]
        out[str(row["id"])] = await retrieve_precedent_sections(sentences, exclude_case_ids=[str(row["id"])])
        if i % 50 == 0:
            print(f"  precedent {i}/{len(rows)}", flush=True)
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["cache"])
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="dev150")
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--statute-depth", type=int, default=40)
    parser.add_argument("--acts", nargs="*", help="scope statute retrieval to these exact act names")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--candidate-pool", type=int, default=30)
    parser.add_argument("--rerank-pool", type=int, default=20)
    parser.add_argument("--statutes-part", default="statutes", help="cache part name (one per configuration)")
    parser.add_argument("--skip-statutes", action="store_true")
    parser.add_argument("--precedent", action="store_true", help="also cache precedent votes")
    args = parser.parse_args()

    names = label_names()
    rows = sample_iltur_cases(args.sample_size, seed=args.seed, split="dev")
    corpus_path("builds", "tune").mkdir(parents=True, exist_ok=True)
    with open(cache_path(args.tag, "gold"), "wb") as f:
        pickle.dump({str(r["id"]): decode_labels(r["labels"], names) for r in rows}, f)

    if args.precedent:
        with open(cache_path(args.tag, "precedent"), "wb") as f:
            pickle.dump(await build_precedent(rows), f)
        print("precedent cache written", flush=True)
    if not args.skip_statutes:
        with open(cache_path(args.tag, args.statutes_part), "wb") as f:
            pickle.dump(await build_statutes(rows, args), f)
        print("statutes cache written", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
