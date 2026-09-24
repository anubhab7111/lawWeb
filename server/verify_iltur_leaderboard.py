#!/usr/bin/env python3
"""
verify_iltur_leaderboard.py — score test predictions with the IL-TUR leaderboard's own code.

Independent check of report_iltur.py: builds label-set predictions for the test split
(decision thresholds fitted on dev only), then scores them with `evaluate_lsi` from the
leaderboard Space (spaces/Exploration-Lab/IL-TUR-Leaderboard, eval_utils.py), unmodified,
stored with its label vocabulary under <corpus>/builds/eval/leaderboard/.

Usage (from server/):  python verify_iltur_leaderboard.py classifier
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from app.metrics import iltur_official as io
from app.metrics.iltur_loader import label_names
from app.ingest.paths import corpus_path
from score_iltur import gold_labels, load_scores

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
system = sys.argv[1] if len(sys.argv) > 1 else "classifier"
dev_g, test_g = gold_labels("dev"), gold_labels("test")
dev_ids, test_ids = sorted(dev_g), sorted(test_g)
dev_s, test_s = load_scores(system, "dev"), load_scores(system, "test")
dm = io.score_matrix([dev_s[i] for i in dev_ids], vocab)
tm = io.score_matrix([test_s[i] for i in test_ids], vocab)
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
