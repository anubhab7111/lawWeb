#!/usr/bin/env python3
"""
import_case_law.py — add one landmark judgment to the curated case-law corpus
from an official text copy (e.g. the Supreme Court Reports text on the corpus
drive, from the AWS Open Data indian-supreme-court-judgments set).

The case must be in app/data/case_law_manifest.py, which supplies its
doctrines and statutes. Afterwards run generate_case_firac.py (and check its
output against the judgment), then rebuild_rag_indices.py --domain case_law.

Usage (conda env legal_chatbot_env, run from server/):
    python import_case_law.py --case "Fateh Chand v. Balkishan Dass (1963)" \
        --from-file .../1964_1_515_534.txt.gz \
        --title "Fateh Chand vs Balkishan Das on 15 January, 1963" \
        --citation "[1964] 1 S.C.R. 515" --date 1963-01-15 --bench-size 5
"""

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from app.data.case_law_manifest import ManifestEntry, build_manifest
from app.ingest.paths import case_law_dir

_CONSTITUTION_BENCH_MIN = 5
MAX_TEXT_CHARS = 60000


def _slug(case_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", case_name.lower()).strip("_")[:80]


def _court_rank(court: str, bench_size: int) -> int:
    """Higher = more authoritative. Used to order case-law retrieval."""
    court = court.lower()
    if "supreme court" in court:
        return 4 if bench_size >= _CONSTITUTION_BENCH_MIN else 3
    return 2 if "high court" in court else 1


def record_from_file(entry: ManifestEntry, path: Path, *, title: str, citation: str, court: str,
                     date: str, bench_size: int, url: str) -> dict:
    """`title` ("A vs B on 15 January, 1963") becomes the first text line, as in the existing records."""
    raw = gzip.open(path, "rt", encoding="utf-8").read() if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    body = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return {
        "case_name": entry["case_name"],
        "citation": citation,
        "court": court,
        "bench_size": bench_size,
        "court_rank": _court_rank(court, bench_size),
        "date": date,
        "status": "reported",
        "doctrines": entry["doctrines"],
        "statutes_cited": entry["statutes_cited"],
        "url": url,
        "text": f"{title}\n\nEquivalent citations: {citation}\n\n{body}"[:MAX_TEXT_CHARS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="case name exactly as in the manifest")
    parser.add_argument("--from-file", required=True, type=Path, help=".txt or .txt.gz judgment text")
    parser.add_argument("--title", required=True, help="'A vs B on 15 January, 1963'")
    parser.add_argument("--citation", default="")
    parser.add_argument("--court", default="Supreme Court of India")
    parser.add_argument("--date", default="", help="YYYY-MM-DD")
    parser.add_argument("--bench-size", type=int, default=1)
    parser.add_argument("--url", default="")
    parser.add_argument("--force", action="store_true", help="overwrite an existing record (drops its firac)")
    args = parser.parse_args()

    entry = next((e for e in build_manifest() if e["case_name"] == args.case), None)
    if not entry:
        sys.exit(f"{args.case!r} is not in the case-law manifest")
    out_dir = case_law_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slug(entry['case_name'])}.json"
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path.name} already exists; pass --force to replace it")
    record = record_from_file(entry, args.from_file, title=args.title, citation=args.citation, court=args.court,
                              date=args.date, bench_size=args.bench_size, url=args.url)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out_path.name}: {len(record['text'])} chars")


if __name__ == "__main__":
    main()
