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
from typing import Optional

import numpy as np

from app.tools.crime_reporter import CLASSIFIER_DIR, FAMILY_TYPES

GOLD = Path("tests/gold/crime_type.jsonl")
ILSIC_DIR = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset")
# Weight of ILSIC lay questions (train split) relative to bail rows; 0 = bail only.
ILSIC_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
TYPE_FAMILY = {t: f for f, types in FAMILY_TYPES.items() for t in types if t != "general"}
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


def ilsic_rows(split: str) -> list:
    """ILSIC lay questions whose citations map to exactly one crime type."""
    mapping = _type_map()
    pat = re.compile(r"Section\s+(\S+)\s+of\s+(.+?)(?:,\s*\d{4})?$")
    rows = []
    for line in lzma.open(ILSIC_DIR / f"FT-Layman-{split}.jsonl.xz", "rt"):
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


EMBED_BATCH = 32


async def embed(texts: list, checkpoint: Optional[Path] = None) -> np.ndarray:
    """Unit-normalised BGE-M3 embeddings, in ThermalGuard-paced batches. With `checkpoint`,
    each batch is saved there and reused on the next run, so an interrupted run resumes."""
    from app.ingest.thermal import ThermalGuard
    from app.tools.base_legal_rag import _get_shared_embeddings

    emb = await _get_shared_embeddings()
    guard = ThermalGuard()
    parts = []
    for n, i in enumerate(range(0, len(texts), EMBED_BATCH)):
        part_path = checkpoint / f"{n:05d}.npy" if checkpoint else None
        if part_path and part_path.exists():
            parts.append(np.load(part_path))
            continue
        part = np.array(emb.embed_documents(texts[i : i + EMBED_BATCH]), dtype=np.float32)
        if part_path:
            np.save(part_path.with_suffix(".tmp.npy"), part)
            part_path.with_suffix(".tmp.npy").rename(part_path)
        parts.append(part)
        guard.step()
    X = np.vstack(parts)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cached_embed(texts: list) -> np.ndarray:
    """Training embeddings, checkpointed per batch under a content-hash directory."""
    key = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:16]
    checkpoint = Path.home() / ".cache" / "lawweb" / f"crime_type_embed_{key}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    return asyncio.run(embed(texts, checkpoint))


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
    """Logistic regression on bail dev facts + case summaries, plus ILSIC lay crime questions
    (train split) at a weight; C and that weight picked by 5-fold, case-grouped CV on bail dev
    facts with the gold accept sets."""
    from sklearn.linear_model import LogisticRegression

    from app.tools.crime_reporter import resolve_family

    rows = load_gold("dev")
    texts, y, fold_of, owner, lay = [], [], [], [], []
    for i, r in enumerate(rows):
        for t in [r["text"], *r["extra"]]:
            texts.append(t)
            y.append(r["family"])
            fold_of.append(i % 5)
            owner.append(i if t is r["text"] else -1)
            lay.append(False)
    for r in ilsic_rows("train"):
        if r["accept"][0] in TYPE_FAMILY:
            texts.append(r["text"][:2000])
            y.append(TYPE_FAMILY[r["accept"][0]])
            fold_of.append(-1)
            owner.append(-1)
            lay.append(True)
    X = cached_embed(texts)
    y, fold_of, owner, lay = np.array(y), np.array(fold_of), np.array(owner), np.array(lay)

    def fit(c: float, w: float, mask: np.ndarray):
        keep = mask & (~lay | (w > 0))
        weight = np.where(lay[keep], w, 1.0)
        return LogisticRegression(C=c, max_iter=3000, class_weight=CLASS_WEIGHT).fit(
            X[keep], y[keep], sample_weight=weight
        )

    from app.ingest.thermal import ThermalGuard

    guard = ThermalGuard(duty=0.5, threads=2)

    def cv_acc(c: float, w: float) -> float:
        hits = 0
        for k in range(5):
            clf = fit(c, w, fold_of != k)
            guard.step()
            test = np.where((fold_of == k) & (owner >= 0))[0]
            for i, family in zip(test, clf.predict(X[test])):
                r = rows[owner[i]]
                hits += resolve_family(family, r["text"]) in r["accept"]
        return hits / len(rows)

    scores = {}
    for w in ILSIC_WEIGHTS:
        for c in C_GRID:
            scores[(c, w)] = cv_acc(c, w)
            print(f"lay weight {w} C={c}: {scores[(c, w)]:.3f}", flush=True)
    best_c, best_w = max(scores, key=scores.get)
    print(f"-> C={best_c}, lay weight={best_w} ({int(lay.sum())} lay rows available)")
    clf = fit(best_c, best_w, np.ones(len(y), dtype=bool))
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
    check = ilsic_rows("test")
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
