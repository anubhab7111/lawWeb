#!/usr/bin/env python3
"""
Export the IL-TUR `lsi` splits to parquet on the corpus drive, with labels
decoded from ClassLabel ids to section numbers.

  iltur/lsi/{train,dev,test}.parquet   id, sentences, labels (section numbers), label_ids
  iltur/lsi/statutes.parquet           label name + statute text for the 100 labels
  iltur/lsi/EXPORT.json                counts, label stats, licence

The test split is exported for evaluation only. `assert_disjoint` enforces that no
case id appears in more than one split, so nothing evaluated on can be indexed.

IL-TUR is CC BY-NC-SA 4.0 (non-commercial).

Usage (from server/):
    python -m app.ingest.iltur_export
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.ingest.paths import corpus_path
from app.metrics.iltur_loader import DATASET_CONFIG, DATASET_ID, decode_labels, label_names

LICENSE = "CC BY-NC-SA 4.0 (non-commercial)"


def assert_disjoint(ids_by_split: Dict[str, List[str]]) -> None:
    seen: Dict[str, str] = {}
    for split, ids in ids_by_split.items():
        for case_id in ids:
            if case_id in seen and seen[case_id] != split:
                raise AssertionError(
                    f"IL-TUR case {case_id} appears in both '{seen[case_id]}' and '{split}'"
                )
            seen[case_id] = split


def lsi_dir() -> Path:
    return corpus_path("iltur", "lsi")


def export() -> Dict:
    from datasets import load_dataset

    from app.config import get_settings

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, token=get_settings().huggingface_token or None)
    names = label_names()
    out = lsi_dir()
    out.mkdir(parents=True, exist_ok=True)

    ids_by_split: Dict[str, List[str]] = {}
    summary: Dict = {"license": LICENSE, "source": DATASET_ID, "splits": {}}
    for split in ("train", "dev", "test"):
        rows = []
        counts: Counter = Counter()
        for row in ds[split]:
            sentences = row["text"] if isinstance(row["text"], list) else [row["text"]]
            labels = decode_labels(row["labels"], names)
            counts.update(labels)
            rows.append(
                {
                    "id": str(row["id"]),
                    "sentences": sentences,
                    "labels": labels,
                    "label_ids": list(row["labels"]),
                }
            )
        ids_by_split[split] = [r["id"] for r in rows]
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet", index=False)
        summary["splits"][split] = {
            "cases": len(rows),
            "mean_sentences": sum(len(r["sentences"]) for r in rows) / max(len(rows), 1),
            "distinct_labels": len(counts),
        }
    assert_disjoint(ids_by_split)

    statutes = [
        {"label": names[i], "text": " ".join(row["text"]) if isinstance(row["text"], list) else row["text"]}
        for i, row in enumerate(ds["statutes"])
    ]
    pd.DataFrame(statutes).to_parquet(out / "statutes.parquet", index=False)
    summary["statutes"] = len(statutes)
    (out / "EXPORT.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    print(json.dumps(export(), indent=1))
