#!/usr/bin/env python3
"""
Supplementary Indian Legal Data Downloader
==========================================

Downloads and VERIFIES the corpus-expansion data beyond bare acts:

  mappings/    Old-to-new criminal code correspondence tables
               (IPC↔BNS, CrPC↔BNSS, IEA↔BSA) from BPRD / UP Police.
               These let the RAG answer "what is IPC 420 under BNS?".
  rules/       Statutory Rules (procedure layer): POCSO Rules 2020,
               IT Intermediary Rules 2021, DPDP Rules 2025, E-Commerce
               Rules 2020, POSH Rules 2013, Notaries Rules 1956, CMV
               Rules 1989.
  guides/      Plain-language citizen guides (NALSA schemes, victim
               compensation).
  bare_acts/   Statutes new or missing from the main corpus: the three
               2020 labour codes, Income-tax Act 2025 (replaces the
               1961 Act from 1 Apr 2026), Mediation Act 2023,
               Telecommunications Act 2023, Immigration & Foreigners
               Act 2025, and the two biggest state rent acts
               (Maharashtra 1999, Delhi 1958) under property/.

NOTE: only files under bare_acts/ are picked up by the unified RAG
index today. mappings/, rules/ and guides/ need their own ingestion
(ideally the mapping tables as structured JSON) — they are downloaded
here so that work has verified source material.

Every download is validated before it is kept:
  1. response must start with the %PDF magic bytes,
  2. the PDF must parse and meet a per-document minimum page count,
  3. the extracted text of the first pages must contain a per-document
     required phrase (so a notification/slideshow can't masquerade as
     an act — this exact failure happened with BSA 2023 and POCSO),
  4. a mid-document page must have a real text layer (not scanned).

Usage:  python download_supplementary_data.py
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.5,*/*;q=0.4",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES_PER_URL = 3  # India Code intermittently 302s to an HTML session page


def _thc_rule(name: str) -> str:
    return "https://thc.nic.in/Central%20Governmental%20Rules/" + name.replace(" ", "%20")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentSpec:
    """One document to download, with its validation contract."""

    subdir: str            # destination directory under app/data/
    filename: str
    min_pages: int         # reject anything shorter (catches notifications)
    must_contain: str      # phrase required in first-pages text (lowercase)
    urls: list[str] = field(default_factory=list)  # tried in order
    # Accept documents whose later pages are scanned as long as the front
    # matter has a text layer (e.g. rules whose annexure forms are scans).
    allow_partial: bool = False

    @property
    def dest(self) -> Path:
        return BASE_DIR / self.subdir / self.filename


DOCUMENTS: list[DocumentSpec] = [
    # ══════════════════════════════════════════════════════════════════
    # 1. OLD↔NEW CRIMINAL CODE MAPPING TABLES
    # ══════════════════════════════════════════════════════════════════
    # WB Land Reforms officers' assoc. tables are the reliably reachable
    # mirrors; the official BPRD / UP Police copies are kept as fallbacks
    # (both hosts time out from some networks).
    DocumentSpec(
        "mappings", "IPC_to_BNS_Comparative_Table.pdf", 30, "bharatiya nyaya sanhita",
        [
            "https://wbllroa.in/wp-content/uploads/2024/07/COMPARATIVE-TABLE-OF-IPC-1860-BNS-2023-ADV-GURENDER-RANA.pdf",
            "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202406281710564823BNS_IPC_Comparative.pdf",
        ],
    ),
    DocumentSpec(
        "mappings", "CrPC_to_BNSS_Comparative_Table.pdf", 20, "nagarik suraksha",
        [
            "https://wbllroa.in/wp-content/uploads/2024/07/COMPARATIVE-TABLE-OF-CRPC-1973-BHARTIYA-NAGARIK-SURAKSHA-SANHITA-2023-ADV-GURENDER-RANA.pdf",
            "https://bprd.nic.in/uploads/pdf/Comparison%20summary%20BNSS%20to%20CrPC.pdf",
        ],
    ),
    DocumentSpec(
        "mappings", "IEA_to_BSA_Comparative_Table.pdf", 10, "sakshya",
        [
            "https://wbllroa.in/wp-content/uploads/2024/07/COMPARATIVE-TABLE-OF-EVIDENCE-ACT-BHARTIYA-SAKSHYA-ADHINIYAM-2023-ADV-GURENDER-RANA.pdf",
            "https://bprd.nic.in/uploads/pdf/Comparison%20Summary%20BSA%20to%20IEA.pdf",
            "https://keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BSA_to_IEA.pdf",
        ],
    ),
    DocumentSpec(
        # CDTI Hyderabad (BPRD/MHA training institute) compendium of the
        # three amended criminal laws.
        "mappings", "Amended_Criminal_Laws_Compendium_CDTI.pdf", 50, "sanhita",
        ["https://cdtihyd.gov.in/static/download/LatestLaws/amended_criminal_laws.pdf"],
    ),
    # ══════════════════════════════════════════════════════════════════
    # 2. STATUTORY RULES (the procedure layer under the acts)
    # ══════════════════════════════════════════════════════════════════
    DocumentSpec(
        "rules", "POCSO_Rules_2020.pdf", 6, "children from sexual",
        [
            "https://gsja.nic.in/gsjanew/files/20201005163659_POCSO_Rules_2020.pdf",
            "https://wcd.nic.in/sites/default/files/POCSO%20Rules%20merged_2.pdf",
        ],
        # Only the title page of the gsja copy has a text layer — the body
        # is scanned. A tesseract OCR of the full text is committed next to
        # it as POCSO_Rules_2020.txt; use that for ingestion.
        allow_partial=True,
    ),
    DocumentSpec(
        "rules", "IT_Intermediary_Guidelines_Rules_2021.pdf", 15, "intermediary",
        [
            "https://www.meity.gov.in/static/uploads/2024/02/Information-Technology-Intermediary-Guidelines-and-Digital-Media-Ethics-Code-Rules-2021-updated-06.04.2023-.pdf",
            "https://www.meity.gov.in/static/uploads/2024/02/IT-Intermediary-Rules-2021-updated-on-28.10.2022-2.pdf",
        ],
    ),
    DocumentSpec(
        "rules", "DPDP_Rules_2025.pdf", 15, "digital personal data protection",
        [
            _thc_rule("Digital Personal Data Protection Rules, 2025.pdf"),
            "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf",
            "https://www.dpdpa.com/DPDP_Rules_2025_English_only.pdf",
        ],
    ),
    # NOTE: Consumer Protection (E-Commerce) Rules 2020 and the Central
    # Motor Vehicles Rules 1989 are deliberately absent: every official PDF
    # found (thc.nic.in scans, morth chapter pages) has no text layer, and
    # the reachable copies elsewhere are commentary, not the rules. Ingest
    # them from an HTML source (e.g. India Code's rules viewer) or OCR the
    # thc.nic.in scans if needed.
    DocumentSpec(
        "rules", "POSH_Act_and_Rules_2013.pdf", 5, "sexual harassment",
        [
            # IIT Bhubaneswar compilation of the POSH Act + Rules 2013
            "https://www.iitbbs.ac.in/notice/sexual-harrassment-of-women-act-and-rules-2013.pdf",
            _thc_rule(
                "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Rules, 2013.pdf"
            ),
            "https://patnahighcourt.gov.in/bslsa/pdf/ActsRules/94.pdf",
        ],
    ),
    DocumentSpec(
        "rules", "Notaries_Rules_1956.pdf", 6, "notaries",
        [
            _thc_rule("Notaries Rules, 1956.pdf"),
            "https://upload.indiacode.nic.in/showfile?actid=AC_CEN_3_46_00007_195253_1517807328336&type=rule&filename=notaries-rules-1956.pdf",
        ],
    ),
    # ══════════════════════════════════════════════════════════════════
    # 3. CITIZEN GUIDES (plain-language, official)
    # ══════════════════════════════════════════════════════════════════
    # NOTE: NALSA's Compendium of Schemes and the individual scheme PDFs on
    # nalsa.gov.in are image scans (no text layer) — skipped for the same
    # reason as the rules above. Only the VC scheme below has real text.
    DocumentSpec(
        "guides", "NALSA_Victim_Compensation_Scheme_2018.pdf", 5, "compensation",
        [
            "https://cdnbbsr.s3waas.gov.in/s32e45f93088c7db59767efef516b306aa/uploads/2025/04/202504081255775595.pdf",
            "https://kslsa.kar.nic.in/pdfs/guidelines/NALSAs_Compensation_Scheme_for_Women_Victims-2018.pdf",
            "https://wcd.nic.in/sites/default/files/Final%20VC%20Sheme_0.pdf",
        ],
    ),
    # ══════════════════════════════════════════════════════════════════
    # 4. NEW / UPDATED STATUTES  (indexed by the unified RAG)
    # ══════════════════════════════════════════════════════════════════
    DocumentSpec(
        "bare_acts/labour", "Industrial_Relations_Code_2020.pdf", 25, "industrial relations",
        ["https://www.indiacode.nic.in/bitstream/123456789/22040/1/A2020-35.pdf"],
    ),
    DocumentSpec(
        "bare_acts/labour", "Code_on_Social_Security_2020.pdf", 40, "social security",
        ["https://www.indiacode.nic.in/bitstream/123456789/16823/1/aA2020-36.pdf"],
    ),
    DocumentSpec(
        "bare_acts/labour", "Occupational_Safety_Health_Code_2020.pdf", 30, "occupational safety",
        ["https://www.indiacode.nic.in/bitstream/123456789/22041/1/a2020-37.pdf"],
    ),
    DocumentSpec(
        # Replaces the Income-tax Act 1961 with effect from 1 April 2026.
        # The 1961 PDF is kept for historical queries.
        "bare_acts/indirect_tax", "Income_Tax_Act_2025.pdf", 300, "income-tax",
        [
            # Official gazette of Act 30 of 2025; incometaxindia.gov.in
            # sits behind Akamai and 403s non-browser clients.
            "https://egazette.gov.in/WriteReadData/2025/265620.pdf",
            "https://resource.cdn.icai.org/87647dtc-aps2139-inceome-tax-act-2025.pdf",
            "https://www.incometaxindia.gov.in/Documents/Act/Income-tax-Act-2025.pdf",
        ],
    ),
    DocumentSpec(
        "bare_acts/civil", "Mediation_Act_2023.pdf", 20, "mediation",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/19637/1/aA2023-32.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/19637/1/A2023-32.pdf",
        ],
    ),
    DocumentSpec(
        "bare_acts/consumer_cyber_ip", "Telecommunications_Act_2023.pdf", 20, "telecommunication",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/20101/1/A2023-44.pdf",
            "https://egazette.gov.in/WriteReadData/2023/250880.pdf",
        ],
    ),
    DocumentSpec(
        "bare_acts/criminal", "Immigration_and_Foreigners_Act_2025.pdf", 15, "immigration",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/21918/1/A2025-13.pdf",
            "https://prsindia.org/files/bills_acts/acts_parliament/2025/The_Immigration_and_Foreigners_Act,_2025.pdf",
        ],
    ),
    # State rent acts live under property/ (an indexed domain); the doc
    # validator's DOCUMENT_LAW_MAP cites "Rent Control Acts (state-specific)".
    DocumentSpec(
        "bare_acts/property", "Maharashtra_Rent_Control_Act_1999.pdf", 30, "rent",
        [
            # consolidated as modified up to Feb 2016; the Housing Dept CDN
            # copy is an image scan, so it is not listed here
            "https://indiacode.nic.in/bitstream/123456789/7848/1/maha._rent_control_act%2C_1999_%2818_of_2000%29_%28modi._8_feb._2016%29.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15817/1/the_maharashtra_rent_control_act,_1999.pdf",
            "https://upload.indiacode.nic.in/showfile?actid=AC_MH_166_1465_00004_00004_1609153355090&filename=the_maharashtra_rent_control_act%2C_1999.pdf&type=actfile",
            "https://www.indiacode.nic.in/bitstream/123456789/15817/3/eng_maharashtra_rent_control_ac.pdf",
        ],
    ),
    DocumentSpec(
        "bare_acts/property", "Delhi_Rent_Control_Act_1958.pdf", 25, "rent",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/19223/1/a1958-59.pdf",
            "https://www.mohua.gov.in/upload/uploadfiles/files/drc_1958.pdf",
        ],
    ),
]


# ── Validation ────────────────────────────────────────────────────────────────
def validate_pdf(data: bytes, spec: DocumentSpec) -> tuple[bool, str]:
    """Full validation contract for a downloaded document (see module docstring)."""
    if not data.startswith(b"%PDF"):
        return False, "response is not a PDF (%PDF header missing)"
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        return False, f"unparseable PDF: {exc}"
    n = len(reader.pages)
    if n < spec.min_pages:
        return False, f"only {n} pages (expected >= {spec.min_pages})"
    head = " ".join(
        (reader.pages[i].extract_text() or "") for i in range(min(5, n))
    ).lower()
    if spec.must_contain.lower() not in head:
        return False, f"required phrase {spec.must_contain!r} not found in first pages"
    mid = (reader.pages[n // 2].extract_text() or "").strip()
    if len(mid) < 100 and not spec.allow_partial:
        return False, "mid-document page has no real text layer (scanned?)"
    return True, f"{n} pages, text layer OK"


# ── Download ──────────────────────────────────────────────────────────────────
def fetch(client: httpx.Client, spec: DocumentSpec) -> tuple[bool, str]:
    """Try each candidate URL (with retries) until one passes validation."""
    if spec.dest.exists():
        ok, why = validate_pdf(spec.dest.read_bytes(), spec)
        if ok:
            return True, f"already present ({why})"
        log.warning(f"  existing file INVALID ({why}) — re-downloading")

    for url in spec.urls:
        for attempt in range(1, MAX_RETRIES_PER_URL + 1):
            try:
                resp = client.get(url)
                ctype = resp.headers.get("content-type", "")
                if resp.status_code != 200 or "text/html" in ctype:
                    log.info(f"  miss  [{attempt}/{MAX_RETRIES_PER_URL}] HTTP {resp.status_code} {ctype}  {url}")
                    time.sleep(2 * attempt)
                    continue
                ok, why = validate_pdf(resp.content, spec)
                if not ok:
                    log.info(f"  reject {why}  {url}")
                    break  # same bytes on retry — move to next URL
                spec.dest.parent.mkdir(parents=True, exist_ok=True)
                spec.dest.write_bytes(resp.content)
                return True, f"{why}  <- {url}"
            except Exception as exc:
                log.info(f"  error [{attempt}/{MAX_RETRIES_PER_URL}] {type(exc).__name__}  {url}")
                time.sleep(2 * attempt)
    return False, "no candidate URL yielded a valid document"


def main() -> int:
    log.info("=" * 70)
    log.info("  Supplementary Legal Data Downloader")
    log.info(f"  Destination : {BASE_DIR}")
    log.info(f"  Documents   : {len(DOCUMENTS)}")
    log.info("=" * 70)

    results: list[tuple[DocumentSpec, bool, str]] = []
    timeout = httpx.Timeout(120.0, connect=15.0)  # fail fast on dead gov hosts
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as client:
        for i, spec in enumerate(DOCUMENTS, 1):
            log.info(f"[{i}/{len(DOCUMENTS)}] {spec.subdir}/{spec.filename}")
            ok, detail = fetch(client, spec)
            results.append((spec, ok, detail))
            log.info(("  [OK]   " if ok else "  [FAIL] ") + detail)
            time.sleep(0.8)

    ok_n = sum(1 for _, ok, _ in results if ok)
    log.info("=" * 70)
    log.info(f"  VALIDATION REPORT — {ok_n}/{len(results)} documents verified")
    log.info("=" * 70)
    for spec, ok, detail in results:
        log.info(f"  {'✓' if ok else '✗'} {spec.subdir}/{spec.filename}: {detail}")
    if ok_n < len(results):
        log.warning("Some documents failed — find them manually and re-run "
                    "(existing valid files are skipped).")
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
