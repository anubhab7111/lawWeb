#!/usr/bin/env python3
"""
Extract, clean, dedup and chunk the downloaded Supreme Court judgments.

Per year, under <corpus>/judgments/sc/:
  text/year=YYYY/<stem>.txt.gz        raw pdftotext output
  text/year=YYYY/_extraction.parquet  one row per PDF: quality metrics, status, sha
  clean/year=YYYY/<stem>.txt.gz       cleaned paragraphs
  chunks/year=YYYY/chunks.jsonl.gz    schema-validated chunk records
  chunks/year=YYYY/_DONE.json         marks a finished year (pipeline_version, counts)
and quarantine/extract|chunk/sc/... for rejected documents / chunks, each with a reason.

Resumable: a year with a matching _DONE.json is skipped (--force redoes it).
A QA report and a random spot-check sample are written to <corpus>/builds/qa/.

Usage (from server/):
    python -m app.ingest.process_judgments --years 1985 2005
    python -m app.ingest.process_judgments               # every downloaded year
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.ingest.judgment_chunker import (
    PIPELINE_VERSION,
    ChunkValidationError,
    chunk_judgment,
    validate_chunk,
)
from app.ingest.judgment_clean import text_quality
from app.ingest.judgment_extract import extract_pages, pages_to_paragraphs, plain_text
from app.ingest.paths import corpus_path

MAX_INVALID_CHUNK_RATE = 0.01


def sc_root() -> Path:
    return corpus_path("judgments", "sc")


def load_metadata(year: int) -> Dict[str, Dict]:
    path = sc_root() / "metadata" / f"year={year}.parquet"
    if not path.exists():
        return {}
    cols = ["path", "title", "judge", "citation", "decision_date", "disposal_nature", "case_id"]
    df = pd.read_parquet(path, columns=cols)
    return {Path(str(row["path"])).stem.removesuffix("_EN"): row for row in df.to_dict("records")}


def _gz_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


def process_pdf(args: tuple) -> Dict:
    """Worker: one PDF -> extraction row + chunks (or a quarantine reason)."""
    pdf_path, year, meta, root = args
    root = Path(root)
    stem = Path(pdf_path).stem.removesuffix("_EN")
    doc_id = f"sc-{stem}"
    row: Dict = {"doc_id": doc_id, "stem": stem, "year": year}
    try:
        pages = extract_pages(str(pdf_path))
    except (subprocess.SubprocessError, OSError) as e:
        return {**row, "status": "quarantined", "reason": f"pdftotext failed: {type(e).__name__}", "chunks": []}
    raw = plain_text(pages)

    quality = text_quality(raw, pages=max(len(pages), 1))
    row.update({k: quality[k] for k in ("chars", "pages", "chars_per_page", "alpha_ratio", "garbled_ratio")})
    if not quality["ok"]:
        return {**row, "status": "quarantined", "reason": quality["reason"], "chunks": []}

    _gz_write(root / "text" / f"year={year}" / f"{stem}.txt.gz", raw)
    clean = "\n\n".join(pages_to_paragraphs(pages))
    _gz_write(root / "clean" / f"year={year}" / f"{stem}.txt.gz", clean)
    row["clean_sha256"] = hashlib.sha256(clean.encode()).hexdigest()

    chunks, invalid = [], []
    for chunk in chunk_judgment(doc_id, year, clean, meta):
        try:
            validate_chunk(chunk)
            chunks.append(chunk)
        except ChunkValidationError as e:
            invalid.append({"chunk_id": chunk["chunk_id"], "reason": str(e)})
    if not chunks:
        return {**row, "status": "quarantined", "reason": "no valid chunks", "chunks": [], "invalid": invalid}
    return {**row, "status": "ok", "reason": None, "chunks": chunks, "invalid": invalid}


def process_year(year: int, workers: int, force: bool = False) -> Optional[Dict]:
    root = sc_root()
    done = root / "chunks" / f"year={year}" / "_DONE.json"
    if done.exists() and not force and json.loads(done.read_text()).get("pipeline_version") == PIPELINE_VERSION:
        return None

    pdfs = sorted((root / "pdf" / f"year={year}").glob("*.pdf"))
    meta = load_metadata(year)
    jobs = [(str(p), year, meta.get(p.stem.removesuffix("_EN"), {}), str(root)) for p in pdfs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_pdf, jobs, chunksize=4))

    seen: Dict[str, str] = {}
    all_chunks: List[Dict] = []
    n_invalid = 0
    for r in results:
        n_invalid += len(r.get("invalid", []))
        sha = r.get("clean_sha256")
        if r["status"] == "ok" and sha in seen:
            r.update(status="duplicate", reason=f"duplicate_of {seen[sha]}")
            r["chunks"] = []
        elif r["status"] == "ok":
            seen[sha] = r["doc_id"]
        all_chunks.extend(r["chunks"])

    for r in results:
        if r["status"] == "quarantined":
            q = corpus_path("quarantine", "extract", "sc", f"year={year}")
            q.mkdir(parents=True, exist_ok=True)
            (q / f"{r['stem']}.reason.json").write_text(json.dumps({k: r[k] for k in r if k != "chunks"}, default=str))

    out = root / "chunks" / f"year={year}"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "chunks.jsonl.gz.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.rename(out / "chunks.jsonl.gz")

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("chunks", "invalid")} for r in results])
    text_dir = root / "text" / f"year={year}"
    text_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(text_dir / "_extraction.parquet", index=False)

    n_ok = sum(r["status"] == "ok" for r in results)
    rate = n_invalid / max(len(all_chunks) + n_invalid, 1)
    summary = {
        "year": year,
        "pipeline_version": PIPELINE_VERSION,
        "pdfs": len(pdfs),
        "ok": n_ok,
        "quarantined": Counter(r["reason"] for r in results if r["status"] == "quarantined"),
        "duplicates": sum(r["status"] == "duplicate" for r in results),
        "chunks": len(all_chunks),
        "invalid_chunks": n_invalid,
        "invalid_rate": rate,
    }
    done.write_text(json.dumps(summary, default=dict, indent=1))
    if rate > MAX_INVALID_CHUNK_RATE:
        raise RuntimeError(f"year {year}: {rate:.1%} of chunks failed validation")
    return summary


def keep_year(stem: str, folder_years: List[int]) -> int:
    """Which year folder keeps a judgment that the bucket lists in several.

    The bucket lists a case under both its decision year and its report year;
    the filename's leading year is the report year, so that copy wins, else the
    earliest folder.
    """
    report_year = int(stem.removeprefix("S_")[:4])
    return report_year if report_year in folder_years else min(folder_years)


def dedupe_across_years(years: List[int]) -> Dict:
    """Drop the copies of a judgment that sit in more than one year folder.

    Chunk files are rewritten without the dropped copies, their extraction rows
    are marked `duplicate`, and every drop is recorded in
    <corpus>/manifest/sc_cross_year_duplicates.jsonl. Raw PDFs are left alone.
    Idempotent: rows already marked duplicate are ignored.
    """
    root = sc_root()
    owners: Dict[str, List[int]] = {}
    tables: Dict[int, pd.DataFrame] = {}
    for year in years:
        path = root / "text" / f"year={year}" / "_extraction.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        tables[year] = df
        for stem in df.loc[df["status"] == "ok", "stem"]:
            owners.setdefault(stem, []).append(year)

    drops: Dict[int, set] = {}
    ledger: List[Dict] = []
    for stem, folders in owners.items():
        if len(folders) < 2:
            continue
        keep = keep_year(stem, folders)
        for year in folders:
            if year != keep:
                drops.setdefault(year, set()).add(stem)
                ledger.append({"doc_id": f"sc-{stem}", "kept_year": keep, "dropped_year": year})

    for year, stems in drops.items():
        doc_ids = {f"sc-{s}" for s in stems}
        chunk_dir = root / "chunks" / f"year={year}"
        kept = 0
        tmp = chunk_dir / "chunks.jsonl.gz.tmp"
        with gzip.open(chunk_dir / "chunks.jsonl.gz", "rt", encoding="utf-8") as src, gzip.open(
            tmp, "wt", encoding="utf-8"
        ) as dst:
            for line in src:
                if json.loads(line)["doc_id"] not in doc_ids:
                    dst.write(line)
                    kept += 1
        tmp.rename(chunk_dir / "chunks.jsonl.gz")

        df = tables[year]
        mask = df["stem"].isin(stems)
        df.loc[mask, "status"] = "duplicate"
        df.loc[mask, "reason"] = "cross-year duplicate"
        df.to_parquet(root / "text" / f"year={year}" / "_extraction.parquet", index=False)
        done = chunk_dir / "_DONE.json"
        summary = json.loads(done.read_text())
        summary.update(chunks=kept, cross_year_duplicates=summary.get("cross_year_duplicates", 0) + len(stems))
        done.write_text(json.dumps(summary, indent=1))

    if ledger:
        path = corpus_path("manifest", "sc_cross_year_duplicates.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.writelines(json.dumps(row) + "\n" for row in ledger)
    return {"dropped_docs": len(ledger), "years_rewritten": sorted(drops)}


def qa_report(years: List[int], samples: int = 20) -> Dict:
    root = sc_root()
    roles, words, per_doc_cites = Counter(), [], []
    docs, cited_chunks, total_chunks = set(), 0, 0
    reservoir: List[Dict] = []
    rng = random.Random(7)
    for year in years:
        path = root / "chunks" / f"year={year}" / "chunks.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                total_chunks += 1
                roles[c["role"]] += 1
                words.append(c["n_words"])
                docs.add(c["doc_id"])
                cited_chunks += bool(c["sections_cited"])
                if len(reservoir) < samples:
                    reservoir.append(c)
                elif rng.random() < samples / total_chunks:
                    reservoir[rng.randrange(samples)] = c
    words.sort()
    q = lambda p: words[int(len(words) * p)] if words else 0
    report = {
        "years": [years[0], years[-1]] if years else [],
        "docs": len(docs),
        "chunks": total_chunks,
        "roles": dict(roles),
        "words_p10_p50_p90_max": [q(0.1), q(0.5), q(0.9), words[-1] if words else 0],
        "chunks_with_statute_citation": cited_chunks / max(total_chunks, 1),
    }
    qa = corpus_path("builds", "qa")
    qa.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (qa / f"judgments_sc_{stamp}.json").write_text(json.dumps(report, indent=1))
    with open(qa / f"spotcheck_sc_{stamp}.md", "w") as f:
        for c in reservoir:
            f.write(f"### {c['chunk_id']} — {c['case_title']} ({c['year']}) [{c['role']}] {c['sections_cited']}\n\n{c['text']}\n\n")
    report["spotcheck"] = str(qa / f"spotcheck_sc_{stamp}.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = sc_root()
    years = args.years or sorted(
        int(p.name.split("=")[1]) for p in (root / "pdf").glob("year=*") if any(p.glob("*.pdf"))
    )
    started = time.time()
    for year in years:
        summary = process_year(year, args.workers, args.force)
        if summary is None:
            print(f"[judgments] {year} up to date")
            continue
        q = dict(summary["quarantined"])
        print(
            f"[judgments] {year}: {summary['ok']}/{summary['pdfs']} ok, {summary['chunks']} chunks, "
            f"quarantined {q or 0}, dup {summary['duplicates']}, invalid chunks {summary['invalid_chunks']} "
            f"({time.time() - started:.0f}s)"
        )
    dedup = dedupe_across_years(years)
    print(f"[judgments] cross-year duplicates dropped: {dedup['dropped_docs']} (years rewritten: {len(dedup['years_rewritten'])})")
    report = qa_report(years)
    print("[judgments] QA:", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
