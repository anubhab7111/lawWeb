#!/usr/bin/env python3
"""
fetch_case_law.py — build the curated landmark-judgment corpus.

For each case in app/data/case_law_manifest.py: search Indian Kanoon for the
best-matching judgment, fetch its full text via the /doc/ endpoint, extract
court/bench/date/citation, and save to app/data/case_law/<slug>.json.

Idempotent/resumable — already-fetched cases (by slug) are skipped, so a
partial or failed run can simply be re-invoked. Two metered API calls per
new case (search + doc fetch); a short delay is added between cases to
avoid hammering the paid API.

Usage (conda env legal_chatbot_env, run from server/):
    python fetch_case_law.py            # fetch all missing cases
    python fetch_case_law.py --limit 10 # fetch at most 10 new cases
    python fetch_case_law.py --force    # re-fetch everything
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from bs4 import BeautifulSoup

from app.config import get_settings
from app.data.case_law_manifest import ManifestEntry, build_manifest
from app.tools.indian_kanoon import IndianKanoonClient

CASE_LAW_DIR = _SERVER_DIR / "app" / "data" / "case_law"

_CONSTITUTION_BENCH_MIN = 5


def _slug(case_name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", case_name.lower()).strip("_")
    return s[:80]


_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _search_query(case_name: str) -> str:
    """
    'Bachan Singh v. State of Punjab (1980)' -> 'Bachan Singh vs State of
    Punjab'. IK's relevance ranking degrades badly on the bare "v." form and
    the trailing year — both get tokenized as noise and drown short party
    names in "State of X" boilerplate from unrelated cases.
    """
    name = _YEAR_RE.sub("", case_name).strip()
    return re.sub(r"\bv\.\s*", "vs ", name, flags=re.IGNORECASE)


def _title_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z]+", a.lower()))
    wb = set(re.findall(r"[a-z]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _pick_best_result(case_name: str, docs: list) -> Optional[dict]:
    if not docs:
        return None
    cited_year = _YEAR_RE.search(case_name)
    cited_year = cited_year.group(1) if cited_year else None

    def score(d: dict) -> float:
        overlap = _title_overlap(_search_query(case_name), d.get("title", ""))
        is_sc = "supreme court" in (d.get("docsource", "") or "").lower()
        year_match = bool(
            cited_year and str(d.get("publishdate", "")).startswith(cited_year)
        )
        return overlap + (0.2 if is_sc else 0.0) + (0.3 if year_match else 0.0)

    # "Daily Orders" are brief procedural listings (hearing dates, notices),
    # not the reasoned judgment — never the right pick when an actual
    # judgment of the same case exists among the results, even if its title
    # (with "& Ors" etc.) scores marginally lower on raw word overlap.
    is_daily_order = lambda d: "daily order" in (d.get("docsource", "") or "").lower()
    judgments = [d for d in docs if not is_daily_order(d)]
    pool = judgments or docs

    scored = sorted(pool, key=score, reverse=True)
    best_doc = scored[0]
    return best_doc if score(best_doc) >= 0.4 else None


def _extract_bench_size(doc_html: str) -> int:
    soup = BeautifulSoup(doc_html[:5000], "html.parser")
    bench_el = soup.find(class_="doc_bench")
    if not bench_el:
        return 1
    judges = bench_el.find_all("a")
    return max(1, len(judges))


def _extract_citation(doc_html: str) -> str:
    soup = BeautifulSoup(doc_html[:5000], "html.parser")
    cite_el = soup.find(class_="doc_citations")
    if cite_el:
        return cite_el.get_text(strip=True).replace("Equivalent citations:", "").strip()
    return ""


def _court_rank(docsource: str, bench_size: int) -> int:
    """Higher = more authoritative. Used to order case-law retrieval."""
    src = (docsource or "").lower()
    if "supreme court" in src:
        return 4 if bench_size >= _CONSTITUTION_BENCH_MIN else 3
    if "high court" in src:
        return 2
    return 1


def _html_to_text(doc_html: str, max_chars: int = 60000) -> str:
    """
    Strip judgment HTML to plain text, capped for very long judgments
    (a handful of landmark cases run past 500KB of prose; the corpus needs
    substantial-but-bounded text per case, not the entire multi-hour read).
    """
    soup = BeautifulSoup(doc_html, "html.parser")
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


async def fetch_one(client: IndianKanoonClient, entry: ManifestEntry) -> Optional[dict]:
    # search_documents() parses results via LegalDocument, which drops the
    # docsource field _pick_best_result needs to prefer Supreme Court hits —
    # go one level lower and hit the search endpoint directly for raw fields.
    session = await client._get_session()
    async with session.post(
        f"{client.BASE_URL}/search/",
        params={"formInput": _search_query(entry["case_name"]), "pagenum": 0},
    ) as resp:
        if resp.status != 200:
            print(f"  search failed ({resp.status}) for {entry['case_name']}")
            return None
        data = await resp.json()
    best = _pick_best_result(entry["case_name"], data.get("docs", []))
    if not best:
        print(f"  no confident match for: {entry['case_name']}")
        return None

    doc = await client.fetch_document(str(best["tid"]))
    if not doc:
        return None

    doc_html = doc.get("doc", "")
    bench_size = _extract_bench_size(doc_html)
    return {
        "case_name": entry["case_name"],
        "citation": _extract_citation(doc_html) or doc.get("title", ""),
        "court": doc.get("docsource", ""),
        "bench_size": bench_size,
        "court_rank": _court_rank(doc.get("docsource", ""), bench_size),
        "date": doc.get("publishdate", ""),
        "status": "reported",  # no automated overruled-detection; verify manually if needed
        "doctrines": entry["doctrines"],
        "statutes_cited": entry["statutes_cited"],
        "url": f"https://indiankanoon.org/doc/{best['tid']}/",
        "tid": str(best["tid"]),
        "text": _html_to_text(doc_html),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    CASE_LAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()

    api_key = get_settings().indian_kanoon_api_key
    if not api_key:
        print("FATAL: INDIAN_KANOON_API_KEY not set in .env")
        sys.exit(1)

    client = IndianKanoonClient(api_key)
    fetched = 0
    skipped = 0
    failed = 0
    try:
        for entry in manifest:
            slug = _slug(entry["case_name"])
            out_path = CASE_LAW_DIR / f"{slug}.json"
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            if args.limit is not None and fetched >= args.limit:
                break

            print(f"[{fetched + failed + 1}] Fetching: {entry['case_name']}")
            try:
                result = await fetch_one(client, entry)
            except Exception as e:
                print(f"  ERROR: {e}")
                result = None

            if result:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(
                    f"  saved: {result['court']} | bench={result['bench_size']} "
                    f"| {len(result['text'])} chars"
                )
                fetched += 1
            else:
                failed += 1

            await asyncio.sleep(0.6)  # be a polite metered-API citizen
    finally:
        await client.close()

    print(
        f"\nDone. fetched={fetched} skipped(existing)={skipped} "
        f"failed={failed} manifest_total={len(manifest)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
