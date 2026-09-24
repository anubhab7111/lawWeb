#!/usr/bin/env python3
"""
Build the IPC <-> BNS section correspondence from the official comparative table.

The table (statutes/mappings/IPC_to_BNS_Comparative_Table.pdf) lists each IPC
section beside the Bharatiya Nyaya Sanhita provision that replaced it, e.g.
`302 -> 103`, `420 -> 318`, `34 -> 3(5)`. Sub-clause suffixes are dropped: the
index is keyed on base section numbers. Rows the table marks as not carried over
("-", "Omitted") have no BNS counterpart.

Output: app/data/ipc_bns_map.json, {"ipc_to_bns": {...}, "bns_to_ipc": {...}}, a
small fixture so retrieval never needs the corpus drive.

Usage (from server/):
    python -m app.ingest.ipc_bns_map
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from app.ingest.paths import statutes_dir

OUT = Path(__file__).resolve().parent.parent / "data" / "ipc_bns_map.json"
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
        ipc, bns = base_section(row[0]), base_section(row[2])
        if ipc and bns:
            pairs.append((ipc, bns))
    return pairs


def build_maps(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
    ipc_to_bns: Dict[str, List[str]] = {}
    bns_to_ipc: Dict[str, List[str]] = {}
    for ipc, bns in pairs:
        if bns not in ipc_to_bns.setdefault(ipc, []):
            ipc_to_bns[ipc].append(bns)
        if ipc not in bns_to_ipc.setdefault(bns, []):
            bns_to_ipc[bns].append(ipc)
    return {"ipc_to_bns": ipc_to_bns, "bns_to_ipc": bns_to_ipc}


def extract(pdf_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                pairs.extend(rows_to_pairs(table))
    return pairs


def load_map() -> Dict[str, Dict[str, List[str]]]:
    return json.loads(OUT.read_text())


def main() -> None:
    pairs = extract(statutes_dir() / "mappings" / "IPC_to_BNS_Comparative_Table.pdf")
    maps = build_maps(pairs)
    OUT.write_text(json.dumps(maps, indent=0, sort_keys=True))
    print(f"[ipc_bns_map] {len(pairs)} rows -> {len(maps['ipc_to_bns'])} IPC sections, "
          f"{len(maps['bns_to_ipc'])} BNS sections -> {OUT}")
    for ipc in ("302", "420", "34", "498A", "304B", "376", "120B", "149"):
        print(f"  IPC {ipc} -> BNS {maps['ipc_to_bns'].get(ipc)}")


if __name__ == "__main__":
    main()
