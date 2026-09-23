#!/usr/bin/env python3
"""
Download Supreme Court of India judgments (English) from the public AWS open-data
bucket into the corpus drive, verified and resumable.

Layout written under the corpus root:
  judgments/sc/pdf/year=YYYY/*.pdf          raw judgments (immutable)
  judgments/sc/metadata/year=YYYY.parquet   the bucket's per-year metadata
  judgments/sc/manifest/year=YYYY.jsonl     name, size, sha256 for every PDF
  judgments/sc/download_ledger.jsonl        one line per completed year
  quarantine/download/sc/year=YYYY/         files that failed validation + reason

Verification per year: tar size == HTTP Content-Length == the bucket's own
english.index.json total_size, member count == its file_count, every PDF starts
with %PDF.
A killed run resumes: partial tars continue via HTTP Range, finished years are
skipped when the ledger and the on-disk file count agree.

Usage (from server/):
    python -m app.ingest.download_sc                     # every available year
    python -m app.ingest.download_sc --years 1950 1951   # subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from app.ingest.paths import corpus_path

BUCKET = "https://indian-supreme-court-judgments.s3.amazonaws.com"
CHUNK = 1 << 20
RETRIES = 6


class VerificationError(RuntimeError):
    pass


def sc_root() -> Path:
    return corpus_path("judgments", "sc")


def list_years(client: httpx.Client) -> List[int]:
    resp = client.get(f"{BUCKET}/", params={"list-type": "2", "prefix": "data/tar/", "delimiter": "/"})
    resp.raise_for_status()
    return sorted(int(y) for y in re.findall(r"data/tar/year=(\d{4})/", resp.text))


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except (httpx.HTTPError, OSError) as e:
            if attempt == RETRIES:
                raise
            wait = 2**attempt
            print(f"  {what}: {type(e).__name__} ({e}); retry {attempt}/{RETRIES} in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def download_file(client: httpx.Client, url: str, dest: Path) -> int:
    """Stream url to dest, resuming a partial `<dest>.part`; returns final size."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    def attempt() -> int:
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 416:  # already complete
                return have
            resp.raise_for_status()
            mode = "ab" if resp.status_code == 206 else "wb"
            with open(part, mode) as f:
                for block in resp.iter_bytes(CHUNK):
                    f.write(block)
        return part.stat().st_size

    size = _retry(attempt, dest.name)
    part.rename(dest)
    return size


def validate_pdf_head(data: bytes) -> Optional[str]:
    """Reason a downloaded 'PDF' is bad, or None."""
    if not data:
        return "empty file"
    if not data.lstrip()[:5] == b"%PDF-":
        return "missing %PDF magic (HTML error page or corrupt download)"
    return None


def extract_verified(tar_path: Path, index: Dict, pdf_dir: Path, quarantine_dir: Path) -> List[Dict]:
    """Extract PDFs from the tar with per-file checks; returns manifest rows.

    Raises VerificationError if the member count disagrees with the bucket's own
    index, so a truncated or wrong archive is never accepted.
    """
    rows: List[Dict] = []
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for m in members:
            name = Path(m.name).name  # flatten; refuses path traversal by construction
            handle = tar.extractfile(m)
            data = handle.read() if handle else b""
            problem = validate_pdf_head(data) if name.lower().endswith(".pdf") else None
            if problem:
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                (quarantine_dir / name).write_bytes(data)
                (quarantine_dir / f"{name}.reason.json").write_text(json.dumps({"reason": problem}))
                continue
            (pdf_dir / name).write_bytes(data)
            rows.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if len(members) != index.get("file_count"):
        raise VerificationError(
            f"tar has {len(members)} files; index says {index.get('file_count')}"
        )
    return rows


def year_complete(root: Path, year: int) -> bool:
    manifest = root / "manifest" / f"year={year}.jsonl"
    ledger = root / "download_ledger.jsonl"
    if not manifest.exists() or not ledger.exists():
        return False
    pdf_dir = root / "pdf" / f"year={year}"
    listed = sum(1 for _ in manifest.open())
    on_disk = sum(1 for _ in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else -1
    done = any(
        json.loads(line).get("year") == year and json.loads(line).get("status") == "ok"
        for line in ledger.open()
    )
    return done and listed == on_disk


def process_year(year: int, keep_tar: bool = False) -> Dict:
    root = sc_root()
    if year_complete(root, year):
        return {"year": year, "status": "skipped"}

    with httpx.Client(timeout=httpx.Timeout(60, read=120), follow_redirects=True) as client:
        base = f"{BUCKET}/data/tar/year={year}/english"
        index = _retry(lambda: client.get(f"{base}/english.index.json"), "index")
        if index.status_code == 404:
            return {"year": year, "status": "no-english"}
        index.raise_for_status()
        index = index.json()

        tar_path = root / "_downloads" / f"year={year}.tar"
        head = _retry(lambda: client.head(f"{base}/english.tar"), "head")
        head.raise_for_status()
        expected = int(head.headers["content-length"])
        size = download_file(client, f"{base}/english.tar", tar_path)
        if size != expected or size != index.get("total_size"):
            tar_path.unlink(missing_ok=True)
            raise VerificationError(
                f"year {year}: tar is {size} bytes; server says {expected}, "
                f"index says {index.get('total_size')}"
            )

        pdf_dir = root / "pdf" / f"year={year}"
        quarantine = corpus_path("quarantine", "download", "sc", f"year={year}")
        if pdf_dir.exists():
            shutil.rmtree(pdf_dir)  # never mix a stale partial extraction with a fresh one
        rows = extract_verified(tar_path, index, pdf_dir, quarantine)

        meta = _retry(
            lambda: client.get(f"{BUCKET}/metadata/parquet/year={year}/metadata.parquet"), "metadata"
        )
        if meta.status_code == 200:
            (root / "metadata").mkdir(parents=True, exist_ok=True)
            (root / "metadata" / f"year={year}.parquet").write_bytes(meta.content)

    (root / "manifest").mkdir(parents=True, exist_ok=True)
    (root / "manifest" / f"year={year}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    if not keep_tar:
        tar_path.unlink(missing_ok=True)
    entry = {
        "year": year,
        "status": "ok",
        "files": len(rows),
        "tar_bytes": size,
        "metadata": meta.status_code == 200,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(root / "download_ledger.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--years", type=int, nargs="*", help="default: every year the bucket has")
    parser.add_argument("--workers", type=int, default=3, help="years downloaded in parallel")
    parser.add_argument("--keep-tar", action="store_true")
    args = parser.parse_args()

    with httpx.Client(timeout=60) as client:
        years = args.years or list_years(client)
    print(f"[sc] {len(years)} year(s) to process")

    failures = []

    def run(year: int):
        try:
            result = process_year(year, keep_tar=args.keep_tar)
        except Exception as e:  # keep going; failed years are listed at the end
            failures.append((year, f"{type(e).__name__}: {e}"))
            print(f"[sc] {year} FAILED — {e}")
            return
        print(f"[sc] {year} {result['status']}" + (f" ({result.get('files')} files)" if "files" in result else ""))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, years))

    if failures:
        print(f"[sc] {len(failures)} year(s) failed:")
        for year, why in failures:
            print(f"  {year}: {why}")
        sys.exit(1)
    print("[sc] all years verified")


if __name__ == "__main__":
    main()
