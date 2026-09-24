#!/usr/bin/env python3
"""
Build the old-code <-> new-code section correspondences from the official tables.

  IPC  1860  <-> Bharatiya Nyaya Sanhita 2023            (IPC_to_BNS_Comparative_Table.pdf)
  CrPC 1973  <-> Bharatiya Nagarik Suraksha Sanhita 2023 (CrPC_to_BNSS_Comparative_Table.pdf)

Each table lists an old section beside the provision that replaced it, e.g.
IPC `302 -> BNS 103`, CrPC `482 -> BNSS 528`. Sub-clause suffixes are dropped: the
index is keyed on base section numbers. Rows the table marks as not carried over
("-", "Omitted") have no counterpart.

Output: app/data/section_maps.json
    {"ipc_bns":   {"old_to_new": {...}, "new_to_old": {...}},
     "crpc_bnss": {"old_to_new": {...}, "new_to_old": {...}}}
a small fixture, so retrieval never needs the corpus drive.

Usage (from server/):
    python -m app.ingest.section_maps
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from app.ingest.paths import statutes_dir

OUT = Path(__file__).resolve().parent.parent / "data" / "section_maps.json"
TABLES = {
    "ipc_bns": "IPC_to_BNS_Comparative_Table.pdf",
    "crpc_bnss": "CrPC_to_BNSS_Comparative_Table.pdf",
}
_SECTION = re.compile(r"^(\d{1,3}[A-Z]{0,2})(?:\(\w+\))*$")


def base_section(cell: Optional[str]) -> Optional[str]:
    """'3(5)' -> '3', '121A' -> '121A', '-' -> None."""
    text = (cell or "").strip().replace(" ", "")
    m = _SECTION.match(text)
    return m.group(1) if m else None


def rows_to_pairs(rows: List[List[Optional[str]]]) -> List[Tuple[str, str]]:
    pairs = []
    for row in rows:
        if len(row) < 3:
            continue
        old, new = base_section(row[0]), base_section(row[2])
        if old and new:
            pairs.append((old, new))
    return pairs


def build_maps(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
    old_to_new: Dict[str, List[str]] = {}
    new_to_old: Dict[str, List[str]] = {}
    for old, new in pairs:
        if new not in old_to_new.setdefault(old, []):
            old_to_new[old].append(new)
        if old not in new_to_old.setdefault(new, []):
            new_to_old[new].append(old)
    return {"old_to_new": old_to_new, "new_to_old": new_to_old}


def extract(pdf_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                pairs.extend(rows_to_pairs(table))
    return pairs


def load_maps() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    return json.loads(OUT.read_text())


def main() -> None:
    maps = {}
    for name, pdf in TABLES.items():
        pairs = extract(statutes_dir() / "mappings" / pdf)
        maps[name] = build_maps(pairs)
        print(f"[section_maps] {name}: {len(pairs)} rows -> {len(maps[name]['old_to_new'])} old, "
              f"{len(maps[name]['new_to_old'])} new sections")
    OUT.write_text(json.dumps(maps, indent=0, sort_keys=True))
    for old in ("302", "420", "34", "498A", "304B", "120B"):
        print(f"  IPC {old} -> BNS {maps['ipc_bns']['old_to_new'].get(old)}")
    for old in ("482", "438", "313", "161", "164", "173", "156", "437"):
        print(f"  CrPC {old} -> BNSS {maps['crpc_bnss']['old_to_new'].get(old)}")


if __name__ == "__main__":
    main()
