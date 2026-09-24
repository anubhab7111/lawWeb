#!/usr/bin/env python3
"""
report_iltur.py — official-protocol IL-TUR `lsi` results from cached scores.

Reads per-case scores written by score_iltur.py (and the classifier), fits each
decision rule on the full dev split and reports macro-F1 exactly as the IL-TUR
leaderboard computes it, on the full dev and test splits. With several systems it
also fits a weighted fusion (weights and thresholds chosen on dev only).

Published reference points (IL-TUR, ACL 2024; macro-F1, full test):
  LeSICiN 28.08 (SOTA) | InLegalBERT 26.23 | GPT-4 0-shot 23.99 | LegalBERT 21.74 | BERT 18.44

Usage (from server/):
    python report_iltur.py precedent
    python report_iltur.py precedent classifier --fuse
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from dotenv import load_dotenv

load_dotenv(_SERVER_DIR / ".env")

import numpy as np  # noqa: E402

from app.ingest.paths import corpus_path  # noqa: E402
from app.metrics import iltur_official as io  # noqa: E402
from app.metrics.iltur_loader import label_names  # noqa: E402
from score_iltur import gold_labels, load_scores  # noqa: E402

REFERENCE = {"LeSICiN (SOTA)": 28.08, "InLegalBERT": 26.23, "GPT-4 0-shot": 23.99}


def matrices(system, vocab, dev_ids, test_ids):
    dev = load_scores(system, "dev")
    test = load_scores(system, "test")
    missing = [i for i in dev_ids if i not in dev] + [i for i in test_ids if i not in test]
    if missing:
        raise SystemExit(f"{system}: {len(missing)} cases unscored (e.g. {missing[:3]}); run score_iltur.py first")
    return (
        io.score_matrix([dev[i] for i in dev_ids], vocab),
        io.score_matrix([test[i] for i in test_ids], vocab),
    )


def show(name, results):
    for rule, r in results.items():
        print(f"  {name:<22}{rule:<22} dev {r['dev']:6.2f}   test {r['test']:6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("systems", nargs="+")
    parser.add_argument("--fuse", action="store_true")
    args = parser.parse_args()

    vocab = list(label_names())
    dev_gold_d, test_gold_d = gold_labels("dev"), gold_labels("test")
    dev_ids, test_ids = sorted(dev_gold_d), sorted(test_gold_d)
    dev_gold = io.gold_matrix([dev_gold_d[i] for i in dev_ids], vocab)
    test_gold = io.gold_matrix([test_gold_d[i] for i in test_ids], vocab)
    print(f"dev {len(dev_ids)} cases, test {len(test_ids)} cases, {len(vocab)} labels\n")

    mats, report = {}, {"systems": {}}
    for system in args.systems:
        mats[system] = matrices(system, vocab, dev_ids, test_ids)
        results = io.evaluate_rules(*mats[system][:1], dev_gold, mats[system][1], test_gold)
        report["systems"][system] = results
        show(system, results)

    if args.fuse and len(args.systems) > 1:
        grid = [0.0, 0.25, 0.5, 1.0, 2.0]
        best = None
        for weights in itertools.product(grid, repeat=len(args.systems) - 1):
            w = (1.0, *weights)
            dev_s = sum(wi * mats[s][0] for wi, s in zip(w, args.systems))
            t = io.fit_global_threshold(dev_s / max(dev_s.max(), 1e-9), dev_gold)
            f1 = io.official_macro_f1(dev_gold, io.predict_threshold(dev_s / max(dev_s.max(), 1e-9), [t] * len(vocab)))
            if best is None or f1 > best[0]:
                best = (f1, w)
        w = best[1]
        dev_s = sum(wi * mats[s][0] for wi, s in zip(w, args.systems))
        test_s = sum(wi * mats[s][1] for wi, s in zip(w, args.systems))
        norm = lambda m: m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)
        results = io.evaluate_rules(norm(dev_s), dev_gold, norm(test_s), test_gold)
        report["fusion"] = {"weights": dict(zip(args.systems, w)), "results": results}
        print(f"\n  fusion weights (dev-fitted): {dict(zip(args.systems, w))}")
        show("fused", results)

    print("\n  published (full test):", ", ".join(f"{k} {v}" for k, v in REFERENCE.items()))
    out = corpus_path("builds", "eval", f"official_{'_'.join(args.systems)}_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
