#!/usr/bin/env python3
"""
Keep IL-TUR test cases out of the retrieval corpus.

IL-TUR cases are drawn from real Supreme Court / High Court judgments, so a
judgment in our index can contain a test case's facts verbatim — and, with them,
the very sections we are scored on. This finds those judgments.

Method: every IL-TUR *test* case is reduced to a set of hashed 10-word shingles
(<ENTITY> masks break a shingle rather than match it). A judgment is flagged for
a test case when it shares at least `min_shingles` distinct shingles with that one
case: boilerplate legal phrasing shares a handful, verbatim reuse of a fact
section shares hundreds. Near-verbatim only — a paraphrase would slip through,
which is why the residual risk is reported alongside the numbers.

Usage (from server/):
    python -m app.ingest.decontaminate                 # scan every processed SC year
    python -m app.ingest.decontaminate --years 1985
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from app.ingest.iltur_export import lsi_dir
from app.ingest.paths import corpus_path

SHINGLE_WORDS = 10
MIN_SHINGLES = 25
_TOKEN = re.compile(r"[a-z0-9\x00]+")
_MASK = "\x00"
_PRIME = np.uint64(1099511628211)


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower().replace("<entity>", f" {_MASK} "))


_word_cache: Dict[str, int] = {}


def _word_id(word: str) -> int:
    wid = _word_cache.get(word)
    if wid is None:
        wid = _word_cache[word] = zlib.crc32(word.encode()) | (1 << 33)
    return wid


def shingle_hashes(tokens: List[str], n: int = SHINGLE_WORDS) -> np.ndarray:
    """uint64 hash of every n-word window that contains no entity mask."""
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((_word_id(t) for t in tokens), dtype=np.uint64, count=len(tokens))
    count = len(tokens) - n + 1
    h = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _PRIME + ids[k : k + count]
    masked = np.fromiter((t == _MASK for t in tokens), dtype=bool, count=len(tokens))
    window_masked = np.convolve(masked.astype(np.int8), np.ones(n, dtype=np.int8), mode="valid") > 0
    return h[~window_masked]


@dataclass
class TestShingles:
    """Sorted shingle hashes of all test cases, each mapped to its case."""

    __test__ = False  # not a pytest class

    hashes: np.ndarray
    case_of: np.ndarray
    case_ids: List[str]

    @classmethod
    def from_texts(cls, texts: Dict[str, str]) -> "TestShingles":
        case_ids = list(texts)
        parts_h, parts_c = [], []
        for idx, case_id in enumerate(case_ids):
            h = np.unique(shingle_hashes(tokenize(texts[case_id])))
            parts_h.append(h)
            parts_c.append(np.full(len(h), idx, dtype=np.int32))
        hashes = np.concatenate(parts_h) if parts_h else np.empty(0, dtype=np.uint64)
        case_of = np.concatenate(parts_c) if parts_c else np.empty(0, dtype=np.int32)
        order = np.argsort(hashes, kind="stable")
        return cls(hashes[order], case_of[order], case_ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, hashes=self.hashes, case_of=self.case_of, case_ids=np.array(self.case_ids))

    @classmethod
    def load(cls, path: Path) -> "TestShingles":
        data = np.load(path)
        return cls(data["hashes"], data["case_of"], [str(c) for c in data["case_ids"]])


def contaminated_cases(
    index: TestShingles, tokens: List[str], min_shingles: int = MIN_SHINGLES
) -> Dict[str, int]:
    """Test cases sharing at least min_shingles distinct shingles with the text."""
    h = np.unique(shingle_hashes(tokens))
    if not len(h) or not len(index.hashes):
        return {}
    pos = np.searchsorted(index.hashes, h)
    pos = np.clip(pos, 0, len(index.hashes) - 1)
    hit = index.hashes[pos] == h
    # A shingle shared by several test cases counts for each; equal hashes are adjacent.
    counts: Counter = Counter()
    for p in pos[hit]:
        j = p
        while j > 0 and index.hashes[j - 1] == index.hashes[p]:
            j -= 1
        while j < len(index.hashes) and index.hashes[j] == index.hashes[p]:
            counts[int(index.case_of[j])] += 1
            j += 1
    return {index.case_ids[c]: n for c, n in counts.items() if n >= min_shingles}


def build_test_index(split: str = "test") -> TestShingles:
    cache = corpus_path("builds", "decontam", f"iltur_{split}_shingles.npz")
    if cache.exists():
        return TestShingles.load(cache)
    df = pd.read_parquet(lsi_dir() / f"{split}.parquet", columns=["id", "sentences"])
    index = TestShingles.from_texts({str(r.id): " ".join(r.sentences) for r in df.itertuples()})
    index.save(cache)
    return index


def scan_years(years: Iterable[int], min_shingles: int = MIN_SHINGLES, split: str = "test") -> Dict:
    index = build_test_index(split)
    root = corpus_path("judgments", "sc", "clean")
    flagged: Dict[str, Dict] = {}
    scanned = 0
    for year in years:
        for path in sorted((root / f"year={year}").glob("*.txt.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                hits = contaminated_cases(index, tokenize(f.read()), min_shingles)
            scanned += 1
            if hits:
                flagged[f"sc-{path.name.removesuffix('.txt.gz')}"] = hits
    return {"scanned": scanned, "flagged_docs": flagged, "min_shingles": min_shingles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--min-shingles", type=int, default=MIN_SHINGLES)
    parser.add_argument(
        "--split", default="test", choices=["test", "dev"],
        help="dev: also exclude a tuning set's source judgments, so tuned parameters can't exploit them",
    )
    args = parser.parse_args()

    root = corpus_path("judgments", "sc", "clean")
    years = args.years or sorted(int(p.name.split("=")[1]) for p in root.glob("year=*"))
    report = scan_years(years, args.min_shingles, args.split)
    out = corpus_path("builds", "decontam")
    out.mkdir(parents=True, exist_ok=True)
    name = "sc_flagged.json" if args.split == "test" else f"sc_flagged_{args.split}.json"
    (out / name).write_text(json.dumps(report, indent=1))
    sizes = sorted(max(v.values()) for v in report["flagged_docs"].values())
    print(
        f"[decontam] scanned {report['scanned']} judgments; flagged {len(sizes)} "
        f"(max shared shingles per doc: median {sizes[len(sizes) // 2] if sizes else 0}, "
        f"max {sizes[-1] if sizes else 0})"
    )


if __name__ == "__main__":
    main()
