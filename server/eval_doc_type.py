"""Document type: gold set, reference fitting, and gate (test accuracy >= 0.90).

Gold (tests/gold/doc_type.jsonl): 15 documents per type, 5 reference ("dev") + 10 test. Court orders
are real Supreme Court judgments from the corpus drive and FIRs are real scans from the ICDAR 2023 FIR
dataset (OCR text cached locally); both are stored as ids and never committed. The other 11 types are
hand-written (tests/gold/doc_type/synthetic/). Each document is judged on its first 2,000 characters.

    python eval_doc_type.py build-gold
    python eval_doc_type.py fit            # embed the dev documents into the reference file
    python eval_doc_type.py score --system regex|embedding --split dev|test
"""

import argparse
import asyncio
import collections
import glob
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

GOLD = Path("tests/gold/doc_type.jsonl")
SYNTHETIC_DIR = Path("tests/gold/doc_type/synthetic")
JUDGMENTS = Path("/run/media/ushtro/anubhab_x9/lawweb/judgments/sc/text")
FIR_OCR = Path.home() / ".cache/lawweb/fir_icdar/ocr.json"
CHARS = 2000
PER_TYPE, REFS = 15, 5
GATE = 0.90
SYNTHETIC_TYPES = {
    "sale_deed": "Sale Deed", "agreement_to_sell": "Agreement to Sell", "power_of_attorney": "Power of Attorney",
    "rent_agreement": "Rent Agreement", "will": "Will / Testament", "partnership_deed": "Partnership Deed",
    "affidavit": "Affidavit", "bail_application": "Bail Application", "complaint": "Complaint (CrPC)",
    "notice": "Notice (CrPC/CPC)", "chargesheet": "Chargesheet",
}


def synthetic_docs(stem: str) -> list:
    return [d.strip() for d in (SYNTHETIC_DIR / f"{stem}.txt").read_text().split("\n=====\n") if d.strip()]


def build_gold() -> None:
    rows = []
    for stem, doc_type in SYNTHETIC_TYPES.items():
        rows += [{"id": f"{stem}:{k}", "source": "synthetic", "type": doc_type} for k in range(len(synthetic_docs(stem)))]
    judgments = sorted(str(p.relative_to(JUDGMENTS)) for p in JUDGMENTS.glob("year=*/*.txt.gz"))
    rows += [{"id": j, "source": "sc_judgment", "type": "Court Order / Judgment"}
             for j in random.Random(17).sample(judgments, PER_TYPE)]
    rows += [{"id": n, "source": "icdar_fir", "type": "FIR"} for n in sorted(json.loads(FIR_OCR.read_text()))[:PER_TYPE]]
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    for rs in by_type.values():
        rs.sort(key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
        for k, r in enumerate(rs):
            r["split"] = "dev" if k < REFS else "test"
    GOLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows", collections.Counter((r["type"], r["split"]) for r in rows))


def load_gold(split: str) -> list:
    rows = [r for r in map(json.loads, GOLD.read_text().splitlines()) if r["split"] == split]
    fir = json.loads(FIR_OCR.read_text())
    for r in rows:
        if r["source"] == "synthetic":
            stem, k = r["id"].split(":")
            text = synthetic_docs(stem)[int(k)]
        elif r["source"] == "sc_judgment":
            text = gzip.open(JUDGMENTS / r["id"], "rt", errors="ignore").read()
        else:
            text = fir[r["id"]]
        r["text"] = text[:CHARS]
    return rows


async def embed(texts: list):
    import numpy as np

    from app.tools.base_legal_rag import _get_shared_embeddings

    emb = await _get_shared_embeddings()
    X = np.array(emb.embed_documents(texts), dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def fit() -> None:
    import numpy as np

    from app.tools.document_classifier import REFERENCE_PATH

    rows = load_gold("dev")
    X = asyncio.run(embed([r["text"] for r in rows]))
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE_PATH, X=X.astype(np.float32), labels=np.array([r["type"] for r in rows]))
    print(f"saved {len(rows)} reference vectors to {REFERENCE_PATH}")


def predict(system: str, rows: list) -> list:
    from app.tools.document_classifier import DocumentClassifier

    clf = DocumentClassifier()
    if system == "regex":
        return [clf.classify(r["text"]).document_type for r in rows]
    X = asyncio.run(embed([r["text"] for r in rows]))
    return [clf.classify(r["text"], x).document_type for r, x in zip(rows, X)]


def score(system: str, split: str) -> int:
    rows = load_gold(split)
    preds = predict(system, rows)
    ok = [p == r["type"] for r, p in zip(rows, preds)]
    acc = sum(ok) / len(ok)
    print(f"[{system} / {split}] accuracy {acc:.3f} (n={len(rows)})")
    by = collections.defaultdict(list)
    for r, p, good in zip(rows, preds, ok):
        by[r["type"]].append((good, p))
    for t, items in sorted(by.items()):
        misses = collections.Counter(p for g, p in items if not g).most_common(3)
        print(f"  {t:24s} {sum(g for g, _ in items)}/{len(items)}  misses {misses}")
    src = collections.defaultdict(list)
    for r, good in zip(rows, ok):
        src[r["source"]].append(good)
    print("  by source:", {s: f"{sum(v) / len(v):.3f} (n={len(v)})" for s, v in src.items()})
    if split == "test":
        print(f"\nGATE {'PASS' if acc >= GATE else 'FAIL'}: {acc:.3f} vs {GATE}")
        return 0 if acc >= GATE else 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-gold", "fit", "score"])
    ap.add_argument("--system", default="embedding", choices=["regex", "embedding"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    if a.cmd == "build-gold":
        build_gold()
    elif a.cmd == "fit":
        fit()
    else:
        sys.exit(score(a.system, a.split))
