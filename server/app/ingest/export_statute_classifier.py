#!/usr/bin/env python3
"""
Package the trained statute classifier for serving.

Copies the best checkpoint (<corpus>/builds/classifier/<tag>/best.pt) into the repo's
index area (app/data/faiss_index/statute_classifier/, gitignored like the indices) with
the per-label decision thresholds fitted on the full IL-TUR dev split, so the chatbot
never needs the corpus drive.

Usage (from server/):  python -m app.ingest.export_statute_classifier --tag c2
"""

from __future__ import annotations

import argparse
import json
import shutil

from app.ingest.paths import corpus_path
from app.metrics import iltur_official as io
from app.metrics.iltur_loader import label_names
from app.tools.statute_classifier import BASE_MODEL, SERVING_DIR


def main() -> None:
    import sys

    sys.path.insert(0, str(SERVING_DIR.parents[2]))
    from score_iltur import gold_labels, load_scores

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="c2")
    args = parser.parse_args()

    vocab = list(label_names())
    dev_gold = gold_labels("dev")
    dev_scores = load_scores("classifier", "dev")
    ids = sorted(dev_gold)
    scores = io.score_matrix([dev_scores[i] for i in ids], vocab, normalize=False)
    gold = io.gold_matrix([dev_gold[i] for i in ids], vocab)
    t = io.fit_global_threshold(scores, gold)
    thresholds = io.fit_label_thresholds(scores, gold, fallback=t, shrink=0.3)

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus_path("builds", "classifier", args.tag, "best.pt"), SERVING_DIR / "model.pt")
    (SERVING_DIR / "meta.json").write_text(json.dumps({
        "base_model": BASE_MODEL,
        "labels": vocab,
        "thresholds": [float(x) for x in thresholds],
        "global_threshold": float(t),
        "source": f"train_lsi_classifier.py tag {args.tag}; thresholds fitted on IL-TUR lsi dev (raw probabilities)",
    }, indent=1))
    print(f"[export] {SERVING_DIR}: global threshold {t:.2f}, per-label thresholds fitted")


if __name__ == "__main__":
    main()
