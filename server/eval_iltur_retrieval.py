#!/usr/bin/env python3
"""
eval_iltur_retrieval.py — LLM-free retrieval evaluation on IL-TUR `lsi`.

Draws a seeded sample of IL-TUR cases and scores section rankings against each
case's label set. Labels are bare section numbers (their cases mix IPC and CrPC:
label 482 is CrPC 482 quashing, 438 anticipatory bail), so scoring is by number:
  statutes   windowed statute retrieval combined across the narrative
             (app/tools/fact_statutes.py)
  precedent  similar past IL-TUR train+dev cases voting for their sections
             (app/tools/precedent_rag.py)
  fused      weighted RRF of the two (statutes 0.25, precedent 1.0: tuned on dev)
  popularity no-input baseline: always the most frequent labels
Reports Hit@k, MRR, nDCG@k, MAP, macro-F1@k (the metric IL-TUR's lsi task is scored
with) and 95% bootstrap intervals. Hit@k and MRR are lenient here (about 4 gold labels
per case, and one label in a third of them), so read every system against the
popularity row: a system that doesn't beat it adds nothing.

Tune on --split dev (each queried dev case is masked out of the precedent index);
report on --split test, which is
never indexed.

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
    bootstrap_ci,
    case_metrics,
    label_coverage,
    macro_f1,
    popularity_ranking,
    rrf_fuse,
)
from app.metrics.iltur_loader import decode_labels, label_names, sample_iltur_cases  # noqa: E402
from app.tools.fact_statutes import rank_sections_from_facts  # noqa: E402
from app.tools.precedent_rag import get_precedent_index, retrieve_precedent_sections  # noqa: E402
from app.tools.unified_legal_rag import get_unified_rag_system  # noqa: E402

KS = (1, 3, 5, 10)
SAVED_DEPTH = 30


def is_numbered_act(act_name: str) -> bool:
    """IPC or CrPC: the Acts whose section numbers IL-TUR labels can denote."""
    name = act_name.lower()
    return ("penal code" in name or "criminal procedure" in name) and "bharatiya" not in name


def section_key(section_number: str) -> str:
    return section_number.replace("Article", "").strip().upper()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_SERVER_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


async def rank_statutes(sentences):
    return [number for number, _score in await rank_sections_from_facts(sentences)]


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
    parser.add_argument("--precedent-k", type=int, default=20, help="neighbour windows per query window")
    parser.add_argument("--precedent-power", type=float, default=8.0, help="similarity exponent for votes")
    parser.add_argument(
        "--fusion-weights", type=float, nargs=2, default=[0.25, 1.0],
        metavar=("STATUTE", "PRECEDENT"), help="weights for the fused ranking",
    )
    parser.add_argument("--skip-statutes", action="store_true", help="precedent only (fast; for tuning)")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    use_precedent = get_precedent_index().available
    if not use_precedent:
        print("[precedent] index not built — evaluating statutes only")
    popularity = popularity_ranking(get_precedent_index().load_labels()) if use_precedent else []
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
            section_key(c.section_number) for c in rag._chunks.values() if is_numbered_act(c.act_name)
        ]
        coverage = label_coverage(decode_labels(list(range(len(names))), names), indexed_numbers)
        print(
            f"[coverage] {coverage['covered']}/{coverage['labels']} IL-TUR label sections "
            f"resolvable in the index; missing: {coverage['missing']}"
        )

    rows = sample_iltur_cases(args.sample_size, seed=args.seed, split=args.split)
    per_system = {}
    all_ranked = {}
    details = []
    gold_sizes = []
    golds = []
    started = time.time()
    for i, row in enumerate(rows, start=1):
        sentences = row["text"] if isinstance(row["text"], list) else [row["text"]]
        gold = decode_labels(row["labels"], names)
        gold_sizes.append(len(gold))

        ranks = {}
        golds.append(gold)
        if use_statutes:
            ranks["statutes"] = await rank_statutes(sentences)
        if use_precedent:
            ranks["precedent"] = await rank_precedent(row, sentences, args)
        if popularity:
            ranks["popularity"] = popularity
        if len(ranks) == 3:
            ranks["fused"] = rrf_fuse(
                [ranks["statutes"], ranks["precedent"]], weights=args.fusion_weights
            )

        entry = {"id": row.get("id"), "gold": gold}
        for system, ranked in ranks.items():
            metrics = case_metrics(ranked, gold, ks=KS)
            per_system.setdefault(system, []).append(metrics)
            entry[system] = {"top10": ranked[:10], **metrics}
            all_ranked.setdefault(system, []).append(ranked)
        entry["saved_rankings"] = {s: r[:SAVED_DEPTH] for s, r in ranks.items() if s != "fused"}
        details.append(entry)

        if i % 10 == 0 or i == len(rows):
            progress = " ".join(
                f"{s}={sum(c['hit@5'] for c in m) / i:.3f}" for s, m in per_system.items()
            )
            print(f"[{i}/{len(rows)}] {time.time() - started:.0f}s  hit@5: {progress}")

    summary = {s: aggregate(m, gold_sizes, ks=KS) for s, m in per_system.items()}
    for system, agg in summary.items():
        agg["macro_f1@3"] = macro_f1(all_ranked[system], golds, 3)
        agg["macro_f1@5"] = macro_f1(all_ranked[system], golds, 5)
        agg["hit@5_ci"] = bootstrap_ci([c["hit@5"] for c in per_system[system]])
        agg["mrr_ci"] = bootstrap_ci([c["rr"] for c in per_system[system]])
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
    print(f"  {'system':<11}hit@1   hit@5 [95% CI]      hit@10  MRR [95% CI]        nDCG@5  MAP    macroF1@5")
    for system, agg in summary.items():
        h, m = agg["hit@5_ci"], agg["mrr_ci"]
        print(
            f"  {system:<11}{agg['hit@1']:<8.3f}{agg['hit@5']:.3f} [{h[0]:.3f},{h[1]:.3f}]  {agg['hit@10']:<7.3f}"
            f"{agg['mrr']:.3f} [{m[0]:.3f},{m[1]:.3f}]  {agg['ndcg@5']:<7.3f}{agg['map']:<7.3f}{agg['macro_f1@5']:.3f}"
        )
    print("  (popularity = no-input baseline; a system must beat it to add value)")

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
