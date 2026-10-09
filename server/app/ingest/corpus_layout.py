#!/usr/bin/env python3
"""
Canonical layout of the corpus drive, plus a verifier that enforces it.

  README.md   MANIFEST.json   manifest/        provenance, counts, file hashes
  statutes/   case_law/       judgments/       source documents (raw, immutable)
  iltur/                                        benchmark export
  builds/     quarantine/                      derived artefacts, rejected files
  _legacy/    myenv/                           old scripts; a venv (left as-is)

`--init` creates the folders and README, and tidies strays at the root: old
top-level scripts move to _legacy/, and empty leftover `YYYY/` year folders are
removed (a non-empty one is left alone and reported).
`--verify` fails on anything outside the layout, empty directories, SC files
that disagree with their manifest, and (with --deep) sha256 mismatches.

Usage (from server/):
    python -m app.ingest.corpus_layout --init
    python -m app.ingest.corpus_layout --verify [--deep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from app.ingest.paths import require_corpus

TOP_LEVEL_DIRS = (
    "manifest",
    "quarantine",
    "statutes",
    "case_law",
    "judgments",
    "iltur",
    "builds",
    "tools",
    "_legacy",
)
TOP_LEVEL_FILES = ("README.md", "MANIFEST.json")
LEFT_ALONE = ("myenv", "indiankanoon")  # a virtualenv breaks if moved; indiankanoon/ is a retired API cache
STRAY_SCRIPTS = ("download.sh", "divide.sh")
_YEAR_DIR = re.compile(r"^\d{4}$")

README = """# LawWeb corpus

Source and derived data for the LawWeb legal chatbot (server/app/ingest/*).
Only vector/lexical indices live in the repo; everything else is here.

| Folder | Contents |
|---|---|
| statutes/ | bare acts by domain, mappings, rules, guides, notifications (PDF, as downloaded) |
| case_law/curated/ | landmark-judgment JSON used by the case-law index |
| judgments/sc/ | Supreme Court judgments: pdf/ (raw), text/, clean/, chunks/, metadata/, manifest/ |
| iltur/ | IL-TUR benchmark export (CC BY-NC-SA 4.0 — non-commercial use only) |
| builds/ | embedding caches, evaluation reports, QA reports |
| manifest/ | per-file sha256 manifests and download ledgers |
| quarantine/ | files rejected by a quality gate, each with a reason |
| _legacy/ | scripts from before the pipeline existed |

Layers flow one way: raw -> text -> clean -> chunks -> repo indices.
Verify with `python -m app.ingest.corpus_layout --verify` from server/.

Sources: Supreme Court judgments — AWS Open Data registry
(indian-supreme-court-judgments); statutes — India Code / official gazettes;
IL-TUR — Exploration-Lab/IL-TUR on HuggingFace.
"""


def _is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def init_layout(root: Path) -> Dict[str, List[str]]:
    actions: Dict[str, List[str]] = {"created": [], "moved": [], "removed": [], "kept": []}
    for name in TOP_LEVEL_DIRS:
        target = root / name
        if not target.exists():
            target.mkdir(parents=True)
            actions["created"].append(name)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(README)
        actions["created"].append("README.md")

    for script in STRAY_SCRIPTS:
        src = root / script
        if src.is_file():
            shutil.move(str(src), str(root / "_legacy" / script))
            actions["moved"].append(script)

    for child in sorted(root.iterdir()):
        if _YEAR_DIR.match(child.name) and child.is_dir():
            if _is_empty_dir(child):
                child.rmdir()
                actions["removed"].append(child.name)
            else:
                actions["kept"].append(child.name)
    return actions


def verify(root: Path, deep: bool = False) -> List[str]:
    """Problems found under the corpus root (empty list = clean)."""
    problems: List[str] = []
    allowed = set(TOP_LEVEL_DIRS) | set(TOP_LEVEL_FILES) | set(LEFT_ALONE)
    for child in sorted(root.iterdir()):
        if child.name not in allowed:
            problems.append(f"unexpected top-level entry: {child.name}")

    for base in TOP_LEVEL_DIRS:
        for path in sorted((root / base).rglob("*")) if (root / base).is_dir() else []:
            if path.is_dir() and _is_empty_dir(path):
                problems.append(f"empty directory: {path.relative_to(root)}")

    sc = root / "judgments" / "sc"
    for manifest in sorted((sc / "manifest").glob("year=*.jsonl")) if (sc / "manifest").is_dir() else []:
        year = manifest.stem.split("=")[1]
        rows = [json.loads(line) for line in manifest.open() if line.strip()]
        pdf_dir = sc / "pdf" / f"year={year}"
        on_disk = {p.name for p in pdf_dir.glob("*.pdf")} if pdf_dir.is_dir() else set()
        listed = {r["name"] for r in rows}
        for name in sorted(listed - on_disk):
            problems.append(f"sc {year}: in manifest but missing on disk: {name}")
        for name in sorted(on_disk - listed):
            problems.append(f"sc {year}: on disk but not in manifest: {name}")
        if deep:
            for r in rows:
                p = pdf_dir / r["name"]
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
                    problems.append(f"sc {year}: sha256 mismatch: {r['name']}")

    legacy = root / "manifest" / "legacy_migration.jsonl"
    if legacy.exists():
        quarantined = {p.name.removesuffix(".reason.json") for p in (root / "quarantine").rglob("*.reason.json")}
        for line in legacy.open():
            r = json.loads(line)
            p = root / r["dst"]
            if not p.exists() and Path(r["dst"]).stem in quarantined:
                continue
            if not p.exists():
                problems.append(f"legacy file missing: {r['dst']}")
            elif deep and hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
                problems.append(f"legacy sha256 mismatch: {r['dst']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--deep", action="store_true", help="re-hash files against manifests")
    args = parser.parse_args()
    root = require_corpus()

    if args.init:
        for kind, names in init_layout(root).items():
            if names:
                print(f"[layout] {kind}: {', '.join(names[:12])}{' ...' if len(names) > 12 else ''} ({len(names)})")
    if args.verify:
        problems = verify(root, deep=args.deep)
        for p in problems:
            print(f"[verify] {p}")
        print(f"[verify] {'OK' if not problems else f'{len(problems)} problem(s)'}")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
