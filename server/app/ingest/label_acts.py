#!/usr/bin/env python3
"""
Which Act each IL-TUR `lsi` label usually refers to, measured on the judgment corpus.

IL-TUR labels are bare section names ("Section 482"). The dataset attaches IPC text
to all of them, but its cases often mean the Code of Criminal Procedure (482 =
quashing, 438 = anticipatory bail). For each label number this counts how the
Supreme Court judgments cite it — as IPC or as CrPC — and records the majority Act,
so the chatbot pins the provision the number actually means.

Output: app/data/iltur_label_acts.json  {"Section 482": {"act": "CrPC", "ipc": n, "crpc": n}, ...}

Usage (from server/):  python -m app.ingest.label_acts
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

from app.ingest.paths import corpus_path
from app.tools.label_reranker import statutes

OUT = Path(__file__).resolve().parent.parent / "data" / "iltur_label_acts.json"


def label_number(label: str) -> str:
    return re.sub(r"\(.*\)", "", label.replace("Section ", "")).strip()


def majority_act(ipc: int, crpc: int, min_share: float = 0.6) -> str:
    """CrPC only when judgments clearly cite it that way; IPC otherwise (the label
    vocabulary is IPC's)."""
    total = ipc + crpc
    return "CrPC" if total and crpc / total >= min_share else "IPC"


def count_citations() -> Counter:
    counts: Counter = Counter()
    for path in sorted(corpus_path("judgments", "sc", "chunks").glob("year=*/chunks.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                for cite in json.loads(line)["sections_cited"]:
                    act, _, number = cite.partition(":")
                    if act in ("IPC", "CrPC"):
                        counts[(act, number)] += 1
    return counts


def main() -> None:
    counts = count_citations()
    out = {}
    for label in statutes():
        n = label_number(label)
        ipc, crpc = counts[("IPC", n)], counts[("CrPC", n)]
        out[label] = {"act": majority_act(ipc, crpc), "ipc": ipc, "crpc": crpc}
    OUT.write_text(json.dumps(out, indent=0, sort_keys=True))
    crpc_labels = sorted((l for l, v in out.items() if v["act"] == "CrPC"), key=lambda l: int(re.sub(r"\D", "", l)))
    print(f"[label_acts] {len(out)} labels; CrPC-majority: {crpc_labels}")


if __name__ == "__main__":
    main()
