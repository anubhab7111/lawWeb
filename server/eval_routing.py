"""Intent routing: score classify_intent_embedding on tests/gold/routing.jsonl (gate: test accuracy >= 0.90).

Gold rows carry text directly (synthetic, chat_messages) or by reference: ILSIC questions are read
from the corpus drive (CC BY-NC-SA, not committed) and CLINC150 test queries from Hugging Face.

    python eval_routing.py --split dev|test [--source ilsic,clinc150,...]
"""

import argparse
import asyncio
import collections
import json
import lzma
import re
import sys
from pathlib import Path

GOLD = Path("tests/gold/routing.jsonl")
ILSIC_DIR = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset")
GATE = 0.90


def load_gold(split: str) -> list:
    rows = [json.loads(l) for l in GOLD.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == split]
    ilsic, clinc = {}, None
    for r in rows:
        if r["source"] == "ilsic":
            m = re.fullmatch(r"ilsic_(train|dev|test)(\d+)", r["id"])
            if m.group(1) not in ilsic:
                ilsic[m.group(1)] = [json.loads(l)["instruction"]
                                     for l in lzma.open(ILSIC_DIR / f"FT-Layman-{m.group(1)}.jsonl.xz", "rt")]
            r["text"] = " ".join(ilsic[m.group(1)][int(m.group(2))].split())
        elif r["source"] == "clinc150":
            if clinc is None:
                from datasets import load_dataset

                clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
            r["text"] = clinc[int(r["id"].removeprefix("clinc_test"))]["text"]
    return rows


def check_no_leak(rows: list) -> None:
    from app.intent_classifier import INTENT_REFERENCE_EXAMPLES

    refs = {t.strip().lower() for ts in INTENT_REFERENCE_EXAMPLES.values() for t in ts}
    leaked = [r["id"] for r in rows if r["text"].strip().lower() in refs]
    if leaked:
        sys.exit(f"gold messages duplicate reference examples: {leaked}")


async def classify(rows: list) -> list:
    from app.intent_classifier import classify_intent_embedding

    return [await classify_intent_embedding(r["text"], r["has_document"]) for r in rows]


def report(rows: list, results: list) -> float:
    ok = [res.primary_intent == r["intent"] for r, res in zip(rows, results)]
    acc = sum(ok) / len(ok)
    ambiguous = sum(res.is_ambiguous for res in results) / len(results)
    print(f"accuracy {acc:.3f} (n={len(rows)}); routed to tiebreak/clarify (ambiguous) {ambiguous:.1%}")
    by = collections.defaultdict(list)
    for r, res, good in zip(rows, results, ok):
        by[r["intent"]].append((good, res.primary_intent, r["source"]))
    for intent, items in sorted(by.items()):
        misses = collections.Counter(p for g, p, _ in items if not g).most_common(3)
        print(f"  {intent:18s} recall {sum(g for g, _, _ in items) / len(items):.3f} n={len(items):3d} misses {misses}")
    by_src = collections.defaultdict(list)
    for r, good in zip(rows, ok):
        by_src[r["source"]].append(good)
    print("  by source:", {s: f"{sum(v) / len(v):.3f} (n={len(v)})" for s, v in by_src.items()})
    return acc


async def embed(texts: list):
    import numpy as np

    from app.tools.base_legal_rag import _get_shared_embeddings

    emb = await _get_shared_embeddings()
    X = np.array(emb.embed_documents(texts), dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cross_validate() -> None:
    """5-fold CV on dev: reference examples = current examples + the other folds' dev messages.
    Scored with the classifier's own rule (top-K mean, document prior). Also an LR head on the
    same folds for comparison."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from app import intent_classifier as ic

    rows = load_gold("dev")
    intents = list(ic.INTENT_REFERENCE_EXAMPLES)
    base_texts = [(i, t) for i in intents for t in ic.INTENT_REFERENCE_EXAMPLES[i]]
    B = asyncio.run(embed([t for _, t in base_texts]))
    D = asyncio.run(embed([r["text"] for r in rows]))
    base_lab = np.array([i for i, _ in base_texts])
    dev_lab = np.array([r["intent"] for r in rows])
    has_doc = np.array([r["has_document"] for r in rows])
    fold = np.arange(len(rows)) % 5

    def centroid_scores(ref_X, ref_lab, Q, docs, k):
        S = Q @ ref_X.T
        out = np.zeros((len(Q), len(intents)))
        for j, intent in enumerate(intents):
            s = np.sort(S[:, ref_lab == intent], axis=1)[:, ::-1][:, :k]
            out[:, j] = s.mean(axis=1)
        d = intents.index("document_analysis")
        out[:, d] += np.where(docs, ic.DOCUMENT_ANALYSIS_BOOST, -ic.DOCUMENT_ANALYSIS_ABSENCE_PENALTY)
        return out

    def summarize(name, scores):
        pred = np.array(intents)[scores.argmax(axis=1)]
        top2 = np.sort(scores, axis=1)[:, -2:]
        ambiguous = ((top2[:, 1] - top2[:, 0]) < ic.AMBIGUITY_MARGIN) | (top2[:, 1] < ic.MIN_CONFIDENT_SCORE)
        print(f"{name:42s} acc {np.mean(pred == dev_lab):.3f}  ambiguous {ambiguous.mean():.1%}")

    summarize("current examples (no dev added)", centroid_scores(B, base_lab, D, has_doc, ic.TOP_K_MEAN))
    for k in (1, 3, 5):
        scores = np.zeros((len(rows), len(intents)))
        for f in range(5):
            tr, te = fold != f, fold == f
            ref_X = np.vstack([B, D[tr]])
            ref_lab = np.concatenate([base_lab, dev_lab[tr]])
            scores[te] = centroid_scores(ref_X, ref_lab, D[te], has_doc[te], k)
        summarize(f"current + other-fold dev examples, top-{k}", scores)
    for c in (1, 4, 16):
        scores = np.zeros((len(rows), len(intents)))
        for f in range(5):
            tr, te = fold != f, fold == f
            X = np.hstack([np.vstack([B, D[tr]]), np.concatenate([np.zeros(len(B)), has_doc[tr]])[:, None]])
            y = np.concatenate([base_lab, dev_lab[tr]])
            clf = LogisticRegression(C=c, max_iter=3000).fit(X, y)
            P = clf.predict_proba(np.hstack([D[te], has_doc[te][:, None].astype(float)]))
            scores[te] = P[:, [list(clf.classes_).index(i) for i in intents]]
        pred = np.array(intents)[scores.argmax(axis=1)]
        print(f"{'LR head, C=' + str(c):42s} acc {np.mean(pred == dev_lab):.3f}")
        top = scores.max(axis=1)
        for p in (0.3, 0.4, 0.5, 0.6):
            sure = top >= p
            print(f"    ambiguous below p={p}: {1 - sure.mean():.1%} of messages; "
                  f"accuracy on the rest {np.mean(pred[sure] == dev_lab[sure]):.3f}")


HEAD_C = 1.0


def train() -> None:
    """Fit the intent LR head on the current reference examples + every dev message."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from app import intent_classifier as ic

    rows = load_gold("dev")
    base = [(i, t) for i, ts in ic.INTENT_REFERENCE_EXAMPLES.items() for t in ts]
    X = asyncio.run(embed([t for _, t in base] + [r["text"] for r in rows]))
    doc = np.array([0.0] * len(base) + [float(r["has_document"]) for r in rows])
    y = [i for i, _ in base] + [r["intent"] for r in rows]
    clf = LogisticRegression(C=HEAD_C, max_iter=3000).fit(np.hstack([X, doc[:, None]]), y)
    ic.INTENT_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ic.INTENT_HEAD_PATH, W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32),
             labels=np.array(clf.classes_))
    print(f"saved {ic.INTENT_HEAD_PATH} ({len(y)} training messages, classes {list(clf.classes_)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--source", default="", help="comma-separated sources to keep (default: all)")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--train", action="store_true", help="fit the LR head on reference examples + dev")
    ap.add_argument("--cv", action="store_true", help="5-fold CV of reference-example and LR options on dev")
    a = ap.parse_args()
    if a.train:
        train()
        return 0
    if a.cv:
        cross_validate()
        return 0
    rows = load_gold(a.split)
    if a.source:
        rows = [r for r in rows if r["source"] in a.source.split(",")]
    check_no_leak(rows)
    results = asyncio.run(classify(rows))
    acc = report(rows, results)
    if a.show_misses:
        for r, res in zip(rows, results):
            if res.primary_intent != r["intent"]:
                print(f"  [{r['intent']} -> {res.primary_intent}] {r['text'][:150]}")
    if a.split == "test" and not a.source:
        print(f"\nGATE {'PASS' if acc >= GATE else 'FAIL'}: {acc:.3f} vs {GATE}")
        return 0 if acc >= GATE else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
