#!/usr/bin/env python3
"""
Move the legacy in-repo source corpus (app/data/...) onto the corpus drive.

  app/data/bare_acts/**                   -> <root>/statutes/bare_acts/**
  app/data/{mappings,rules,guides,
            notifications,explanatory}/** -> <root>/statutes/<same>/**
  app/data/case_law/*.json                -> <root>/case_law/curated/*.json

Relative paths under bare_acts/ are preserved, so the per-PDF fingerprints
stored in the existing indices stay valid and no rebuild is forced.

Every file is copied, then re-read from the destination and compared by sha256
against the source. A row per file is written to <root>/manifest/legacy_migration.jsonl.
Source files are only removed with --delete-source, and only after every file
verified; nothing is removed if any check failed.

Usage (from server/):
    python -m app.ingest.migrate_legacy                  # copy + verify
    python -m app.ingest.migrate_legacy --delete-source  # ...then remove local copies
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from app.ingest.paths import LEGACY_DATA_DIR, require_corpus

STATUTE_DIRS = ("bare_acts", "mappings", "rules", "guides", "notifications", "explanatory")
SKIP_NAMES = {"__pycache__", ".DS_Store"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan(src_root: Path, dst_root: Path) -> Iterator[Tuple[Path, Path]]:
    """(source file, destination file) pairs for the whole legacy corpus."""
    for name in STATUTE_DIRS:
        base = src_root / name
        for p in sorted(base.rglob("*")) if base.is_dir() else []:
            if p.is_file() and not (SKIP_NAMES & set(p.parts)):
                yield p, dst_root / "statutes" / name / p.relative_to(base)
    case_dir = src_root / "case_law"
    for p in sorted(case_dir.glob("*.json")) if case_dir.is_dir() else []:
        yield p, dst_root / "case_law" / "curated" / p.name


def migrate(src_root: Path, dst_root: Path, delete_source: bool = False) -> Dict:
    rows: List[Dict] = []
    failures: List[str] = []
    for src, dst in plan(src_root, dst_root):
        digest = sha256(src)
        if not (dst.exists() and sha256(dst) == digest):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        ok = sha256(dst) == digest and dst.stat().st_size == src.stat().st_size
        if not ok:
            failures.append(str(src))
        rows.append(
            {
                "src": str(src.relative_to(src_root)),
                "dst": str(dst.relative_to(dst_root)),
                "size": src.stat().st_size,
                "sha256": digest,
                "verified": ok,
            }
        )

    manifest = dst_root / "manifest" / "legacy_migration.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    removed = 0
    if delete_source and not failures:
        for src, _dst in plan(src_root, dst_root):
            src.unlink()
            removed += 1
    return {
        "files": len(rows),
        "bytes": sum(r["size"] for r in rows),
        "failures": failures,
        "removed": removed,
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    result = migrate(LEGACY_DATA_DIR, require_corpus(), delete_source=args.delete_source)
    print(
        f"[migrate] {result['files']} files, {result['bytes'] / 1e6:.1f} MB; "
        f"{len(result['failures'])} failed verification; {result['removed']} source files removed"
    )
    print(f"[migrate] manifest: {result['manifest']}")
    if result["failures"]:
        print("[migrate] FAILED:", *result["failures"], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
