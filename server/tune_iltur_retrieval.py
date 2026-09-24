#!/usr/bin/env python3
"""
tune_iltur_retrieval.py — cache raw retrieval output for offline tuning.

Full evaluations re-run every retrieval (~20 min per 150 cases). Tuning how the
per-window results are *aggregated* only needs each window's raw candidates, so
this records them once, for a seeded sample of IL-TUR **dev** cases:

  judgments   dense top-K and lexical top-K passages per window, with the
              passage's judgment, cited sections, role and position
  statutes    the raw `retrieve_statutes` output per window across ALL acts
              (so IPC scoping and BNS->IPC translation can be tried offline)

The cache goes to <corpus>/builds/tune/. Judgments whose source text overlaps a
dev case (`decontaminate --split dev`) are dropped when experiments read it, so
tuned parameters cannot exploit a dev case's own judgment.

Usage (from server/):
    python tune_iltur_retrieval.py cache --sample-size 150 --tag dev150
    python tune_iltur_retrieval.py cache --skip-statutes            # judgments only (fast)
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
from app.tools.judgment_rag import get_judgment_index  # noqa: E402
from app.tools.legal_retrieval import retrieve_statutes  # noqa: E402
from app.tools.precedent_rag import encode_texts  # noqa: E402


def cache_path(tag: str, part: str) -> Path:
    return corpus_path("builds", "tune", f"{tag}.{part}.pkl")


async def build_judgments(rows, args):
    from app.tools.base_legal_rag import _get_shared_embeddings

    idx = get_judgment_index()
    idx.load()
    emb = await _get_shared_embeddings()
    cases, needed = [], set()
    started = time.time()
    for i, row in enumerate(rows, start=1):
        windows = fact_windows(row["text"], window=4, stride=3, max_windows=args.max_windows)
        dense = idx.dense(encode_texts(emb, windows), args.depth)
        lexical = [idx.lexical(w, args.lex_depth, max_terms=60) for w in windows]
        for hits in dense:
            needed.update(r for r, _ in hits)
        for rows_ in lexical:
            needed.update(rows_)
        cases.append({"id": str(row["id"]), "windows": windows, "dense": dense, "lexical": lexical})
        if i % 25 == 0:
            print(f"  judgments {i}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    return {"cases": cases, "row_info": idx.row_info(sorted(needed)), "doc_chunks": idx.doc_chunk_counts()}


async def build_statutes(rows, args):
    cases = []
    started = time.time()
    for i, row in enumerate(rows, start=1):
        windows = fact_windows(row["text"], window=4, stride=3, max_windows=args.max_windows)
        per_window = []
        for window in windows:
            result, _ = await retrieve_statutes(window, k=args.statute_depth)
            per_window.append(
                [(c.act_name, c.section_number.replace("Article", "").strip().upper(), round(c.score, 4)) for c in result.chunks]
            )
        cases.append({"id": str(row["id"]), "per_window": per_window})
        if i % 10 == 0:
            print(f"  statutes {i}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    return {"cases": cases}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["cache"])
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--tag", default="dev150")
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--depth", type=int, default=100, help="dense neighbours per window")
    parser.add_argument("--lex-depth", type=int, default=50)
    parser.add_argument("--statute-depth", type=int, default=40)
    parser.add_argument("--skip-statutes", action="store_true")
    parser.add_argument("--skip-judgments", action="store_true")
    args = parser.parse_args()

    names = label_names()
    rows = sample_iltur_cases(args.sample_size, seed=args.seed, split=args.split)
    gold = {str(r["id"]): decode_labels(r["labels"], names) for r in rows}
    corpus_path("builds", "tune").mkdir(parents=True, exist_ok=True)
    with open(cache_path(args.tag, "gold"), "wb") as f:
        pickle.dump(gold, f)

    if not args.skip_judgments:
        with open(cache_path(args.tag, "judgments"), "wb") as f:
            pickle.dump(await build_judgments(rows, args), f)
        print("judgments cache written", flush=True)
    if not args.skip_statutes:
        with open(cache_path(args.tag, "statutes"), "wb") as f:
            pickle.dump(await build_statutes(rows, args), f)
        print("statutes cache written", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
