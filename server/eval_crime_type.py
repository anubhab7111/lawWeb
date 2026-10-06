"""Crime-type classifier: build the gold set, train, and gate (>= 0.90 accuracy on the test split).

Gold: IndianBailJudgments-1200 case facts (CC BY 4.0). A prediction is correct when it is
the case's labelled crime type or a crime type its own cited IPC sections establish (half
the cases charge more than one crime but carry a single label). Check, not gated: ILSIC
lay-forum test questions whose cited statutes map to exactly one crime type.

    python eval_crime_type.py build-gold
    python eval_crime_type.py train
    python eval_crime_type.py score --system keyword|lr --split dev|test
"""

import argparse
import ast
import asyncio
import collections
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

import numpy as np

from app.tools.crime_reporter import CLASSIFIER_DIR, FAMILY_TYPES

GOLD = Path("tests/gold/crime_type.jsonl")
ILSIC_TEST = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset/FT-Layman-test.jsonl.xz")
TYPE_MAP = Path("app/data/crime_type_map.json")
GATE = 0.90
C_GRID = (1, 2, 4, 8, 16)
CLASS_WEIGHT = "balanced"


def _type_map() -> dict:
    mapping = json.loads(TYPE_MAP.read_text())
    mapping.pop("_note")
    return mapping


def build_gold() -> None:
    from datasets import load_dataset

    mapping = _type_map()
    data = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200", split="train")
    seen, rows = set(), []
    for r in data:
        text = " ".join((r["facts"] or "").split())
        key = hashlib.sha1(text.lower().encode()).hexdigest()
        if not text or key in seen:
            continue
        seen.add(key)
        family = "Murder" if r["crime_type"] == "Attempt to Murder" else r["crime_type"]
        sections = [s.strip().upper() for s in ast.literal_eval(r["ipc_sections"] or "[]")]
        accept = set(FAMILY_TYPES[family]) | {mapping[f"IPC:{s}"] for s in sections if f"IPC:{s}" in mapping}
        rows.append(
            {
                "id": str(r["case_id"]),
                "text": text,
                "family": family,
                "ipc_sections": sections,
                "accept": sorted(accept),
                "extra": [" ".join((r[f] or "").split()) for f in ("summary", "legal_issues") if r[f]],
                "split": "test" if int(hashlib.sha1(str(r["case_id"]).encode()).hexdigest(), 16) % 2 else "dev",
            }
        )
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    GOLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    multi = sum(len(r["accept"]) > len(FAMILY_TYPES[r["family"]]) for r in rows)
    print(f"wrote {len(rows)} rows ({len(data) - len(rows)} empty/duplicate dropped), {multi} multi-crime, to {GOLD}")
    print(collections.Counter((r["split"], r["family"]) for r in rows))


def load_gold(split: str) -> list:
    return [r for r in map(json.loads, GOLD.read_text().splitlines()) if r["split"] == split]


def ilsic_check_rows() -> list:
    """ILSIC test questions whose citations map to exactly one crime type."""
    mapping = _type_map()
    pat = re.compile(r"Section\s+(\S+)\s+of\s+(.+?)(?:,\s*\d{4})?$")
    rows = []
    for line in lzma.open(ILSIC_TEST, "rt"):
        r = json.loads(line)
        a = r["answer"]
        labels = a if isinstance(a, list) else (ast.literal_eval(a) if a.startswith("[") else [a])
        types = set()
        for label in labels:
            m = pat.match(label.strip())
            if not m:
                continue
            sec, act = m.groups()
            if act.startswith("The Indian Penal Code") and f"IPC:{sec}" in mapping:
                types.add(mapping[f"IPC:{sec}"])
            types.update(v for k, v in mapping.items() if k.startswith("ACT:") and act.startswith(k[4:]))
        if len(types) == 1:
            rows.append({"text": r["instruction"], "accept": list(types)})
    return rows


async def embed(texts: list) -> np.ndarray:
    from app.tools.base_legal_rag import _get_shared_embeddings

    emb = await _get_shared_embeddings()
    X = np.array(emb.embed_documents(texts), dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def predict(system: str, texts: list) -> list:
    from app.tools import crime_reporter

    if system == "keyword":
        return [crime_reporter.detect_crime_type(t) for t in texts]
    X = asyncio.run(embed(texts))
    return [crime_reporter.crime_type_from_vector(x, t) for x, t in zip(X, texts)]


def report(name: str, rows: list, preds: list) -> float:
    ok = [p in r["accept"] for r, p in zip(rows, preds)]
    acc = sum(ok) / len(ok)
    print(f"\n[{name}] accuracy {acc:.3f} (n={len(rows)})")
    by = collections.defaultdict(list)
    for r, p, good in zip(rows, preds, ok):
        by[r.get("family") or r["accept"][0]].append((good, p))
    for label, items in sorted(by.items()):
        wrong = collections.Counter(p for good, p in items if not good).most_common(3)
        print(f"  {label:20s} recall {sum(g for g, _ in items) / len(items):.3f}  n={len(items):3d}  misses {wrong}")
    return acc


def train() -> None:
    """Logistic regression on dev facts + case summaries; C picked by 5-fold, case-grouped CV
    scored on facts with the gold accept sets."""
    from sklearn.linear_model import LogisticRegression

    from app.tools.crime_reporter import resolve_family

    rows = load_gold("dev")
    texts, y, fold_of, owner = [], [], [], []
    for i, r in enumerate(rows):
        for t in [r["text"], *r["extra"]]:
            texts.append(t)
            y.append(r["family"])
            fold_of.append(i % 5)
            owner.append(i if t is r["text"] else -1)
    X = asyncio.run(embed(texts))
    y, fold_of, owner = np.array(y), np.array(fold_of), np.array(owner)

    def cv_acc(c: float) -> float:
        hits = 0
        for k in range(5):
            clf = LogisticRegression(C=c, max_iter=3000, class_weight=CLASS_WEIGHT).fit(X[fold_of != k], y[fold_of != k])
            test = np.where((fold_of == k) & (owner >= 0))[0]
            for i, family in zip(test, clf.predict(X[test])):
                r = rows[owner[i]]
                hits += resolve_family(family, r["text"]) in r["accept"]
        return hits / len(rows)

    scores = {c: cv_acc(c) for c in C_GRID}
    best = max(scores, key=scores.get)
    print("case-grouped 5-fold dev accuracy:", {c: round(s, 3) for c, s in scores.items()}, "-> C =", best)
    clf = LogisticRegression(C=best, max_iter=3000, class_weight=CLASS_WEIGHT).fit(X, y)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CLASSIFIER_DIR / "weights.npz",
        W=clf.coef_.astype(np.float32),
        b=clf.intercept_.astype(np.float32),
        labels=np.array(clf.classes_),
    )
    print(f"saved {CLASSIFIER_DIR / 'weights.npz'} ({len(clf.classes_)} families, {len(texts)} training texts)")


def score(system: str, split: str) -> int:
    rows = load_gold(split)
    acc = report(f"{system} / bail {split}", rows, predict(system, [r["text"] for r in rows]))
    check = ilsic_check_rows()
    report(f"{system} / ILSIC lay check (not gated)", check, predict(system, [r["text"] for r in check]))
    if split == "test":
        print(f"\nGATE {'PASS' if acc >= GATE else 'FAIL'}: {acc:.3f} vs {GATE}")
        return 0 if acc >= GATE else 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-gold", "train", "score"])
    ap.add_argument("--system", default="lr", choices=["keyword", "lr"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    if a.cmd == "build-gold":
        build_gold()
    elif a.cmd == "train":
        train()
    else:
        sys.exit(score(a.system, a.split))
