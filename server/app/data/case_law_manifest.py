"""
case_law_manifest.py — landmark cases to fetch into the curated case-law
corpus, derived from app/data/legal_ontology.json's `landmark_cases` lists.

Each manifest entry: {case_name, doctrines[], statutes_cited[[act, section]]}
— doctrines/statutes come straight from the ontology entries that cited the
case, so a case fetched for "anticipatory bail" is pre-linked to CrPC §438 /
BNSS §482 without any manual re-entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, TypedDict

_ONTOLOGY_PATH = Path(__file__).resolve().parent / "legal_ontology.json"


class ManifestEntry(TypedDict):
    case_name: str
    doctrines: List[str]
    statutes_cited: List[List[str]]  # [act, section] pairs


def build_manifest() -> List[ManifestEntry]:
    with open(_ONTOLOGY_PATH, encoding="utf-8") as f:
        ontology = json.load(f)

    by_case: Dict[str, ManifestEntry] = {}
    for d in ontology["doctrines"]:
        for case_name in d.get("landmark_cases", []):
            entry = by_case.setdefault(
                case_name, {"case_name": case_name, "doctrines": [], "statutes_cited": []}
            )
            if d["doctrine"] not in entry["doctrines"]:
                entry["doctrines"].append(d["doctrine"])
            for pair in d.get("sections", []):
                if pair not in entry["statutes_cited"]:
                    entry["statutes_cited"].append(pair)

    return sorted(by_case.values(), key=lambda e: e["case_name"])


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"{len(manifest)} unique landmark cases in manifest")
    for e in manifest:
        print(f"  {e['case_name']}  <- {e['doctrines']}")
