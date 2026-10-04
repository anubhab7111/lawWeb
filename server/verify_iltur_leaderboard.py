#!/usr/bin/env python3
"""
verify_iltur_leaderboard.py — score test predictions with the IL-TUR leaderboard's own code.

Independent check of report_iltur.py: builds label-set predictions for the test split
(decision thresholds fitted on dev only), then scores them with `evaluate_lsi` from the
leaderboard Space (spaces/Exploration-Lab/IL-TUR-Leaderboard, eval_utils.py), unmodified,
stored with its label vocabulary under <corpus>/builds/eval/leaderboard/.

Usage (from server/):  python verify_iltur_leaderboard.py classifier
                       python verify_iltur_leaderboard.py classifier precedent_trainmem:0.25   (fusion)
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from app.metrics import iltur_official as io
from app.metrics.iltur_loader import label_names
from app.ingest.paths import corpus_path
from report_iltur import matrices
from score_iltur import gold_labels

LB = corpus_path("builds", "eval", "leaderboard")

spec = importlib.util.spec_from_file_location("eval_utils", str(LB / "eval_utils.py"))
eu = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(eu)
except Exception as e:  # the leaderboard module imports extras (BLEU, rouge...); take just the function
    src = open(str(LB / "eval_utils.py")).read()
    start = src.index("def evaluate_lsi")
    end = src.index("\ndef ", start + 10)
    ns = {}
    exec("import json\nimport numpy as np\nfrom sklearn.metrics import f1_score\n" + src[start:end], ns)
    eu = type("m", (), {"evaluate_lsi": staticmethod(ns["evaluate_lsi"])})

vocab = list(label_names())
parts = [a.split(":") for a in (sys.argv[1:] or ["classifier"])]
weights = [(name, float(w[0]) if w else 1.0) for name, *w in parts]
system = "+".join(f"{w:g}x{name}" if w != 1.0 else name for name, w in weights)
dev_g, test_g = gold_labels("dev"), gold_labels("test")
dev_ids, test_ids = sorted(dev_g), sorted(test_g)
mats = [(w, matrices(name, vocab, dev_ids, test_ids)) for name, w in weights]
dm, tm = (sum(w * m[k] for w, m in mats) for k in (0, 1))
if len(mats) > 1:  # same row normalisation as report_iltur.py --fuse
    norm = lambda m: m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)  # noqa: E731
    dm, tm = norm(dm), norm(tm)
dg = io.gold_matrix([dev_g[i] for i in dev_ids], vocab)
t = io.fit_global_threshold(dm, dg)
th = io.fit_label_thresholds(dm, dg, fallback=t, shrink=0.3)
pred = io.predict_threshold(tm, th)
predictions = {cid: io.labels_from_row(pred[i], vocab) for i, cid in enumerate(test_ids)}
print(f"{system}: mean labels predicted per case {sum(map(len, predictions.values())) / len(predictions):.2f} "
      f"(gold {sum(map(len, test_g.values())) / len(test_g):.2f})")
os.chdir(LB)  # evaluate_lsi opens ./lsi_label_vocab.json
result = eu.evaluate_lsi({i: test_g[i] for i in test_ids}, predictions)
print("OFFICIAL leaderboard evaluate_lsi ->", result)
json.dump(predictions, open(LB / f"{system}_test_predictions.json", "w"))
