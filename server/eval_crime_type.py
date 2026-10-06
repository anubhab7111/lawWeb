"""Crime-type classifier: build the gold set, train, and gate (>= 0.90 accuracy on the test split).

Gold: IndianBailJudgments-1200 case facts (CC-BY-4.0), its crime_type merged onto
crime_reporter's types. Check (not gated): ILSIC lay-forum test questions whose cited
statutes map to exactly one crime type.

    python eval_crime_type.py build-gold
    python eval_crime_type.py score --system keyword|lr --split dev|test
    python eval_crime_type.py train
"""

import argparse
import asyncio
import ast
import collections
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

import numpy as np

GOLD = Path("tests/gold/crime_type.jsonl")
ILSIC_TEST = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset/FT-Layman-test.jsonl.xz")
TYPE_MAP = Path("app/data/crime_type_map.json")
GATE = 0.90

# A prediction is correct when it falls in the accepted set: the bail data merges
# some of our types (theft/robbery, rape/harassment).
BAIL_TO_TYPES = {
    "Theft or Robbery": ["theft", "robbery"],
    "Dowry Harassment": ["dowry"],
    "Sexual Offense": ["rape", "harassment"],
    "Fraud or Cheating": ["fraud"],
    "Cyber Crime": ["cybercrime"],
    "Extortion": ["threat"],
    "Kidnapping": ["kidnapping"],
    "Murder": ["murder"],
    "Attempt to Murder": ["murder"],
    "Domestic Violence": ["domestic_violence"],
    "Narcotics": ["general"],
    "Others": ["general"],
}


def build_gold() -> None:
    from datasets import load_dataset

    data = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200", split="train")
    seen, rows = set(), []
    for r in data:
        text = " ".join((r["facts"] or "").split())
        key = hashlib.sha1(text.lower().encode()).hexdigest()
        if not text or key in seen:
            continue
        seen.add(key)
        split = "test" if int(hashlib.sha1(str(r["case_id"]).encode()).hexdigest(), 16) % 2 else "dev"
        rows.append(
            {
                "id": str(r["case_id"]),
                "text": text,
                "source_label": r["crime_type"],
                "accept": BAIL_TO_TYPES[r["crime_type"]],
                "split": split,
            }
        )
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    GOLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows ({len(data) - len(rows)} empty/duplicate dropped) to {GOLD}")
    print(collections.Counter((r["split"], r["source_label"]) for r in rows))


def load_gold(split: str) -> list:
    return [r for r in map(json.loads, GOLD.read_text().splitlines()) if r["split"] == split]


def ilsic_check_rows() -> list:
    """ILSIC test questions whose citations map to exactly one crime type."""
    mapping = json.loads(TYPE_MAP.read_text())
    mapping.pop("_note")
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


def predict(system: str, texts: list, X: np.ndarray) -> list:
    if system == "keyword":
        from app.tools.crime_reporter import detect_crime_type

        return [detect_crime_type(t) for t in texts]
    if system == "lr":
        from app.tools.crime_reporter import _load_classifier

        W, b, labels = _load_classifier()
        return [labels[i] for i in np.argmax(X @ W.T + b, axis=1)]
    raise ValueError(system)


def report(name: str, rows: list, preds: list) -> float:
    ok = [p in r["accept"] for r, p in zip(rows, preds)]
    acc = sum(ok) / len(ok)
    print(f"\n[{name}] accuracy {acc:.3f} (n={len(rows)})")
    by = collections.defaultdict(list)
    for r, p, good in zip(rows, preds, ok):
        by["/".join(r["accept"])].append((good, p))
    for label, items in sorted(by.items()):
        wrong = collections.Counter(p for good, p in items if not good).most_common(3)
        print(f"  {label:22s} recall {sum(g for g, _ in items) / len(items):.3f}  n={len(items):3d}  misses {wrong}")
    return acc


def train() -> None:
    """Logistic regression on the dev split; C picked by 5-fold cross-validation on dev."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rows = load_gold("dev")
    X = asyncio.run(embed([r["text"] for r in rows]))
    y = [r["accept"][0] if len(r["accept"]) == 1 else r["source_label"] for r in rows]
    best = max(
        (cross_val_score(LogisticRegression(C=c, max_iter=2000), X, y, cv=5).mean(), c)
        for c in (0.5, 1, 2, 4, 8, 16, 32)
    )
    print(f"5-fold dev accuracy {best[0]:.3f} at C={best[1]}")
    clf = LogisticRegression(C=best[1], max_iter=2000).fit(X, y)
    from app.tools.crime_reporter import CLASSIFIER_DIR

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CLASSIFIER_DIR / "weights.npz", W=clf.coef_.astype(np.float32),
             b=clf.intercept_.astype(np.float32), labels=np.array(clf.classes_))
    print(f"saved {CLASSIFIER_DIR / 'weights.npz'} labels={list(clf.classes_)}")


def evaluate(system: str, split: str) -> int:
    rows = load_gold(split)
    texts = [r["text"] for r in rows]
    X = asyncio.run(embed(texts)) if system != "keyword" else None
    acc = report(f"{system} / bail {split}", rows, predict(system, texts, X))
    check = ilsic_check_rows()
    cX = asyncio.run(embed([r["text"] for r in check])) if system != "keyword" else None
    report(f"{system} / ILSIC lay check (not gated)", check, predict(system, [r["text"] for r in check], cX))
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
        sys.exit(evaluate(a.system, a.split))
