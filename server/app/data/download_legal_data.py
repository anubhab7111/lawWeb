#!/usr/bin/env python3
"""
Download curated essential Indian legal documents for RAG pipeline.
~23 core PDFs organised into bare_acts/, notifications/, explanatory/.
Target: 1-1.5 GB vectorisable English legal text.

All URLs are direct PDF links with fallbacks — no scraping required.
Sources: MHA, India Code bitstreams, Legislative Dashboard.
"""

import logging
import time
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf;q=0.8,*/*;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(client: httpx.Client, url: str, dest: Path, retries: int = 3) -> bool:
    """Stream-download *url* to *dest* with retries.  Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1024:
        log.info(f"  [SKIP] Already exists: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
                # India Code returns 302 + HTML body for invalid bitstream URLs
                ct = r.headers.get("content-type", "")
                if "text/html" in ct:
                    log.warning(f"  [WARN] Got HTML instead of PDF — skipping: {url}")
                    return False
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            size_kb = dest.stat().st_size / 1024
            if size_kb < 1:
                log.warning(
                    f"  [WARN] File too small ({size_kb:.1f} KB), discarding: {dest.name}"
                )
                dest.unlink(missing_ok=True)
                return False

            log.info(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
            return True

        except Exception as e:
            log.warning(f"  [RETRY {attempt}/{retries}] {url}: {e}")
            dest.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)

    log.error(f"  [FAIL] Could not download: {url}")
    return False


def download_with_fallbacks(client: httpx.Client, urls: list[str], dest: Path) -> bool:
    """Try each URL in order until one succeeds."""
    for url in urls:
        if download_file(client, url, dest):
            return True
    return False


# ── Document catalogue ────────────────────────────────────────────
# Tuple schema: (sub_directory, filename, [primary_url, fallback_urls...])
#
# India Code bitstream pattern:
#   https://www.indiacode.nic.in/bitstream/123456789/{ITEM_ID}/1/{FILENAME}
# Legislative Dashboard pattern:
#   https://lddashboard.legislative.gov.in/sites/default/files/A{YEAR}-{ACTNO}.pdf

DOCUMENTS: list[tuple[str, str, list[str]]] = [
    # ═══════════════════════════════════════════════════════════════
    # CRIMINAL — New Codes (2023)
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/criminal",
        "Bharatiya_Nyaya_Sanhita_BNS_2023.pdf",
        [
            "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Bharatiya_Nagarik_Suraksha_Sanhita_BNSS_2023.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/21544/1/the_bharatiya_nagarik_suraksha_sanhita%2C_2023.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Bharatiya_Sakshya_Adhiniyam_BSA_2023.pdf",
        [
            "https://www.mha.gov.in/sites/default/files/BharatiyaSakshyaAdhiniyam_24022024.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # CRIMINAL — Legacy
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/criminal",
        "Indian_Penal_Code_1860.pdf",
        [
            # Act 45 of 1860 — state handle (MP)
            "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Code_of_Criminal_Procedure_1973.pdf",
        [
            # Act 2 of 1974 — state handle (CH)
            "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure%2C_1973.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Indian_Evidence_Act_1872.pdf",
        [
            # Act 1 of 1872 — state handle (MP)
            "https://www.indiacode.nic.in/bitstream/123456789/4218/1/THE-INDIAN-EVIDENCE-ACT-1872.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # CONSTITUTION
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/constitutional",
        "Constitution_of_India.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/16124/1/the_constitution_of_india.pdf",
            "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # GENERAL LAW (Civil)
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/civil",
        "Indian_Contract_Act_1872.pdf",
        [
            # Act 9 of 1872 — central handle 2187, version 2
            "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Specific_Relief_Act_1963.pdf",
        [
            # Act 47 of 1963 — central handle 1583, version 7
            "https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Transfer_of_Property_Act_1882.pdf",
        [
            # Act 4 of 1882 — central handle 2338
            "https://www.indiacode.nic.in/bitstream/123456789/2338/1/A1882-04.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Limitation_Act_1963.pdf",
        [
            # Act 36 of 1963 — central handle 1565, version 5
            "https://www.indiacode.nic.in/bitstream/123456789/1565/5/A1963-36.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Negotiable_Instruments_Act_1881.pdf",
        [
            # Act 26 of 1881 — central handle 2189
            "https://www.indiacode.nic.in/bitstream/123456789/2189/1/a1881-26.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # FAMILY LAW
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/family",
        "Hindu_Marriage_Act_1955.pdf",
        [
            # Act 25 of 1955 — central handle 1560
            "https://www.indiacode.nic.in/bitstream/123456789/1560/1/A1955-25.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Special_Marriage_Act_1954.pdf",
        [
            # Act 43 of 1954 — central handle 1387
            "https://www.indiacode.nic.in/bitstream/123456789/1387/1/A195443.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Hindu_Succession_Act_1956.pdf",
        [
            # Act 30 of 1956 — central handle 1713
            "https://www.indiacode.nic.in/bitstream/123456789/1713/1/AAA1956suc___30.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # JUSTICE / PROCEDURE
    # ═══════════════════════════════════════════════════════════════
    (
        "bare_acts/civil",
        "Protection_of_Women_from_Domestic_Violence_Act_2005.pdf",
        [
            # Act 43 of 2005 — central handle 2021, version 5
            "https://www.indiacode.nic.in/bitstream/123456789/2021/5/A2005-43.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Legal_Services_Authorities_Act_1987.pdf",
        [
            # Act 39 of 1987 — central handle 1925
            "https://www.indiacode.nic.in/bitstream/123456789/1925/1/198739.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Right_to_Information_Act_2005.pdf",
        [
            # Act 22 of 2005 — central handle 2065
            "https://www.indiacode.nic.in/bitstream/123456789/2065/1/aa2005.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Code_of_Civil_Procedure_1908.pdf",
        [
            # CPC 1908 — MCRHRDI hosted PDF
            "https://www.mcrhrdi.gov.in/FC2020/week14/Civil%20Procedure%20Code.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Contempt_of_Courts_Act_1971.pdf",
        [
            # Act 70 of 1971 — central handle 1514
            "https://www.indiacode.nic.in/bitstream/123456789/1514/1/A1971-70.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # EXPLANATORY / REFERENCE
    # ═══════════════════════════════════════════════════════════════
    (
        "explanatory",
        "NJA_Organizational_Structure_and_Jurisdiction.pdf",
        [
            # National Judicial Academy — judiciary structure overview
            "https://nja.gov.in/Concluded_Programmes/2018-19/SE-05_2019_PPTs/6.Organizational%20Structure%20and%20Jurisdiction.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════
    # NOTIFICATIONS — Enforcement / Advisory
    # ═══════════════════════════════════════════════════════════════
    (
        "notifications",
        "BNSS_Advisory_Enforcement.pdf",
        [
            "https://www.mha.gov.in/sites/default/files/Advisory(BNSS)_17102024.pdf",
        ],
    ),
    (
        "notifications",
        "BNS_BSA_Enforcement_Notification.pdf",
        [
            "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
        ],
    ),
]


# ── Main ──────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Curated Indian Legal Documents Downloader")
    log.info("  Target: ~20 PDFs for RAG pipeline (English only)")
    log.info("=" * 60)

    success = 0
    failed = 0
    total = len(DOCUMENTS)

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120.0) as client:
        for i, (subdir, filename, urls) in enumerate(DOCUMENTS, 1):
            dest_dir = ensure_dir(BASE_DIR / subdir)
            dest = dest_dir / filename
            log.info(f"\n[{i}/{total}] {subdir}/{filename}")
            if download_with_fallbacks(client, urls, dest):
                success += 1
            else:
                failed += 1
            time.sleep(1)

    # ── Summary ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  DONE  ✓ {success} succeeded · ✗ {failed} failed")
    log.info(f"  Data root: {BASE_DIR}")
    log.info("=" * 60)

    total_mb = 0.0
    for pdf in sorted(BASE_DIR.rglob("*.pdf")):
        rel = pdf.relative_to(BASE_DIR)
        size_mb = pdf.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        log.info(f"  {rel}  ({size_mb:.1f} MB)")
    log.info(f"\n  Total PDF size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
