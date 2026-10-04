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
from app.ingest.paths import case_law_dir, corpus_path
from app.data.case_law_manifest import ManifestEntry, build_manifest
from app.tools.indian_kanoon import IndianKanoonClient

CASE_LAW_DIR = case_law_dir()

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


def _cited_year(case_name: str) -> Optional[int]:
    m = _YEAR_RE.search(case_name)
    return int(m.group(1)) if m else None


def _in_window(cited: int, decided: Optional[int]) -> bool:
    """Landmark names often carry the report year, up to two years after the
    decision (Bhajan Lal "(1992)" was decided in November 1990)."""
    return decided is not None and cited - 2 <= decided <= cited + 1


def _search_queries(case_name: str) -> list:
    """Supreme Court first, then any court; both limited to the cited year's window.
    An unrestricted search let a same-titled High Court order or a later case
    between the same parties win (e.g. Bhajan Lal (1992) -> a 2009 HC revision)."""
    q = _search_query(case_name)
    year = _cited_year(case_name)
    window = f" fromdate: 1-1-{year - 2} todate: 31-12-{year + 1}" if year else ""
    return [f"{q} doctypes: supremecourt{window}", f"{q}{window}"]


def _doc_year(d: dict) -> Optional[int]:
    m = re.match(r"(\d{4})", str(d.get("publishdate", "")))
    return int(m.group(1)) if m else None


def suspect_reason(record: dict) -> Optional[str]:
    """Why a saved record probably holds the wrong judgment, or None."""
    year = _cited_year(record.get("case_name", ""))
    fetched = _doc_year({"publishdate": record.get("date", "")})
    if year and fetched and not _in_window(year, fetched):
        return f"named {year}, fetched a {fetched} judgment"
    if "supreme court" not in (record.get("court") or "").lower():
        return f"fetched a {record.get('court') or 'non-Supreme Court'} document"
    title = (record.get("text") or "").split("\n", 1)[0].split(" on ")[0]
    if title and _party_match(record.get("case_name", ""), title) < 0.5:
        return f"fetched {title[:60]!r}"
    return None


_BOILERPLATE = {"vs", "and", "ors", "anr", "others", "another", "the", "state", "union", "india", "ltd", "limited",
                "pvt", "private", "govt", "government"}


def _distinctive(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in _BOILERPLATE}


def _party_match(case_name: str, title: str) -> float:
    """Share of the case name's distinctive party words found in the title.
    Containment, not Jaccard: IK titles add "& Ors", "M/s", full corporate names."""
    wanted = _distinctive(_search_query(case_name))
    return len(wanted & _distinctive(title)) / len(wanted) if wanted else 0.0


def _pick_best_result(case_name: str, docs: list) -> Optional[dict]:
    """Best judgment for the named case, or None rather than a wrong one.

    Hard requirements: the case's distinctive party names in the title, a date
    in the cited year's window, and not a High Court "Orders" listing (hearing
    dates and counsel appearances). IK files some landmark Supreme Court rulings
    under "Daily Orders" (Subramanian Swamy, 13 May 2016), so those are accepted
    only when dated in the exact cited year."""
    cited = _cited_year(case_name)

    def source(d: dict) -> str:
        return (d.get("docsource", "") or "").lower()

    def acceptable(d: dict) -> bool:
        if _party_match(case_name, d.get("title", "")) < 0.5:
            return False
        if cited and not _in_window(cited, _doc_year(d)):
            return False
        is_sc = "supreme court" in source(d)
        listing = bool(re.search(r"daily order|- orders", source(d)))
        if listing and not is_sc:
            return False
        return not (listing and cited and _doc_year(d) != cited)

    def score(d: dict) -> float:
        is_sc = "supreme court" in source(d)
        exact_year = bool(cited and _doc_year(d) == cited)
        listing = "daily order" in source(d)
        return (_party_match(case_name, d.get("title", "")) + (0.5 if is_sc else 0.0)
                + (0.4 if exact_year else 0.0) - (0.3 if listing else 0.0))

    pool = [d for d in docs if acceptable(d)]
    return max(pool, key=score) if pool else None


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
    docs = []
    for query in _search_queries(entry["case_name"]):
        async with session.post(
            f"{client.BASE_URL}/search/", params={"formInput": query, "pagenum": 0}
        ) as resp:
            if resp.status != 200:
                print(f"  search failed ({resp.status}) for {entry['case_name']}")
                return None
            docs += (await resp.json()).get("docs", [])
    best = _pick_best_result(entry["case_name"], docs)
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
    parser.add_argument("--audit", action="store_true", help="list saved records that look like the wrong judgment (no API calls)")
    parser.add_argument("--refetch-suspect", action="store_true", help="quarantine and re-fetch the records --audit lists")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="re-fetch cases whose name contains NAME (repeatable); the old record is quarantined")
    args = parser.parse_args()

    CASE_LAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()

    def saved(entry):
        path = CASE_LAW_DIR / f"{_slug(entry['case_name'])}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    if args.audit:
        flagged = [(e["case_name"], suspect_reason(r)) for e in manifest if (r := saved(e)) and suspect_reason(r)]
        for name, reason in flagged:
            print(f"  SUSPECT {name}: {reason}")
        print(f"{len(flagged)} suspect of {len(manifest)}")
        return

    def refetch_reason(entry) -> Optional[str]:
        if any(n.lower() in entry["case_name"].lower() for n in args.only):
            return "named with --only"
        record = saved(entry)
        return suspect_reason(record) if record and args.refetch_suspect else None

    quarantine = corpus_path("quarantine", "case_law")

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
            reason = refetch_reason(entry) if out_path.exists() else None
            if reason:
                quarantine.mkdir(parents=True, exist_ok=True)
                out_path.replace(quarantine / out_path.name)
                (quarantine / f"{slug}.reason.json").write_text(json.dumps({"reason": reason}))
                print(f"  quarantined {out_path.name}: {reason}")
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
