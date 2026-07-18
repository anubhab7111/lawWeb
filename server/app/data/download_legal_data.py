#!/usr/bin/env python3
"""
Unified Indian Legal Documents Downloader
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

MAX_RETRIES = 3
MIN_FILE_SIZE_BYTES = 1024  # files smaller than 1 KB are treated as corrupt/empty


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(client: httpx.Client, url: str, dest: Path) -> bool:
    """
    Attempt to download *url* into *dest*.
    Returns True on success, False on any failure (HTML redirect, bad status,
    network error, or suspiciously small file).
    Already-present files are silently skipped and return True.
    """
    if dest.exists() and dest.stat().st_size > MIN_FILE_SIZE_BYTES:
        log.info(f"  [SKIP] Already exists: {dest.name}")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with client.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
                ct = r.headers.get("content-type", "")
                if "text/html" in ct:
                    # India Code / gov servers redirect invalid bitstream URLs
                    # to an HTML session page — not a real PDF.
                    log.warning(f"  [WARN] Server returned HTML (not PDF): {url}")
                    return False
                r.raise_for_status()
                first_chunk = True
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        if first_chunk:
                            if not chunk.startswith(b"%PDF"):
                                log.warning(
                                    f"  [WARN] Response is not a PDF "
                                    f"(missing %PDF header): {url}"
                                )
                                f.close()
                                dest.unlink(missing_ok=True)
                                return False
                            first_chunk = False
                        f.write(chunk)

            size_kb = dest.stat().st_size / 1024
            if dest.stat().st_size < MIN_FILE_SIZE_BYTES:
                log.warning(
                    f"  [WARN] File too small ({size_kb:.1f} KB), discarding: {dest.name}"
                )
                dest.unlink(missing_ok=True)
                return False

            log.info(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
            return True

        except Exception as exc:
            log.warning(f"  [RETRY {attempt}/{MAX_RETRIES}] {url}: {exc}")
            dest.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    log.error(f"  [FAIL] All {MAX_RETRIES} attempts exhausted — skipping: {dest.name}")
    return False


def download_with_fallbacks(client: httpx.Client, urls: list[str], dest: Path) -> bool:
    """Try each URL in order. Return True as soon as one succeeds."""
    for url in urls:
        if download_file(client, url, dest):
            return True
    return False


DOCUMENTS: list[tuple[str, str, list[str]]] = [
    # ═══════════════════════════════════════════════════════════════════════════
    # CRIMINAL — new 2023 codes  (originally from download_legal_data.py)
    # ═══════════════════════════════════════════════════════════════════════════
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
            # NOTE: the old MHA link (BharatiyaSakshyaAdhiniyam_24022024.pdf)
            # was the 2-page enforcement notification, NOT the bare act.
            "https://www.indiacode.nic.in/bitstream/123456789/20063/1/aa202347.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/20063/1/a2023-47.pdf",
            "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CRIMINAL — legacy codes
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/criminal",
        "Indian_Penal_Code_1860.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Code_of_Criminal_Procedure_1973.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure%2C_1973.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Indian_Evidence_Act_1872.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/4218/1/THE-INDIAN-EVIDENCE-ACT-1872.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CRIMINAL — special statutes  (from download_legal_data1.py)
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/criminal",
        "NDPS_Act_1985.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/18974/1/narcotic-drugs-and-psychotropic-substances-act-1985.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Narcotic%20Drugs%20and%20Psychotropic%20Substances%20Act%2C%201985.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Prevention_of_Money_Laundering_Act_PMLA_2002.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15402/1/moneylaunderingact2002.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Prevention%20of%20Money-laundering%20Act%2C%202002.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "POCSO_Act_2012.pdf",
        [
            # NOTE: the old WCD link served a training slideshow, NOT the act.
            "https://www.indiacode.nic.in/bitstream/123456789/2079/1/AA2012-32.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15303/1/pocso_act,_2012.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Protection%20of%20Children%20from%20Sexual%20Offences%20Act%2C%202012.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "SC_ST_Prevention_of_Atrocities_Act_1989.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15338/1/scheduled_castes_and_the_scheduled_tribes.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Scheduled%20Castes%20and%20the%20Scheduled%20Tribes%20%28Prevention%20of%20Atrocities%29%20Act%2C%201989.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSTITUTIONAL
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/constitutional",
        "Constitution_of_India.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/16124/1/the_constitution_of_india.pdf",
            "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
        ],
    ),
    (
        "bare_acts/constitutional",
        "Representation_of_the_People_Act_1951.pdf",
        [
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Representation%20of%20the%20People%20Act%2C%201951.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/2110/4/A1951-43.pdf",
        ],
    ),
    (
        "bare_acts/constitutional",
        "Protection_of_Human_Rights_Act_1993.pdf",
        [
            # GOV — NHRC
            "https://nhrc.nic.in/sites/default/files/PHRA%201993.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Protection%20of%20Human%20Rights%20Act%2C%201993.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15497/1/the_protection_of_human_rights_act,_1993.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CIVIL — core procedure & general
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/civil",
        "Code_of_Civil_Procedure_1908.pdf",
        [
            "https://www.mcrhrdi.gov.in/FC2020/week14/Civil%20Procedure%20Code.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Indian_Contract_Act_1872.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Specific_Relief_Act_1963.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Transfer_of_Property_Act_1882.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2338/1/A1882-04.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Limitation_Act_1963.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1565/5/A1963-36.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Negotiable_Instruments_Act_1881.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2189/1/a1881-26.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Protection_of_Women_from_Domestic_Violence_Act_2005.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2021/5/A2005-43.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Legal_Services_Authorities_Act_1987.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1925/1/198739.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Right_to_Information_Act_2005.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2065/1/aa2005.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Contempt_of_Courts_Act_1971.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1514/1/A1971-70.pdf",
        ],
    ),
    # CIVIL — additions from supplement
    (
        "bare_acts/civil",
        "Arbitration_and_Conciliation_Act_1996.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/21922/1/the_arbitration_and_conciliation_act,_1996_act_no._26_of_1996.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Arbitration%20and%20Conciliation%20Act%2C%201996.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Registration_Act_1908.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15937/1/the_registration_act,1908.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/13236/1/the_registration_act,_1908.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Court_Fees_Act_1870.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/9181/1/the_court-fees_act1870.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Court-fees%20Act%2C%201870.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # FAMILY LAW
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/family",
        "Hindu_Marriage_Act_1955.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1560/1/A1955-25.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Special_Marriage_Act_1954.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1387/1/A195443.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Hindu_Succession_Act_1956.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1713/1/AAA1956suc___30.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Muslim_Personal_Law_Shariat_Application_Act_1937.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2303/1/A1937-26.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Muslim%20Personal%20Law%20%28Shariat%29%20Application%20Act%2C%201937.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Guardians_and_Wards_Act_1890.pdf",
        [
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Guardians%20and%20Wards%20Act%2C%201890.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/11553/1/guardians_and_wards_act,_1890.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/2264/3/A1890-08.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Dowry_Prohibition_Act_1961.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15364/1/the_dowry_prohibition_act,_1961.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Dowry%20Prohibition%20Act%2C%201961.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Maintenance_and_Welfare_of_Parents_and_Senior_Citizens_Act_2007.pdf",
        [
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Maintenance%20and%20Welfare%20of%20Parents%20and%20Senior%20Citizens%20Act%2C%202007.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/2102/5/A2007-56.pdf",
            "https://socialjustice.gov.in/public/uploads/2024-03/1710742424_0.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Muslim_Women_Protection_of_Rights_on_Divorce_Act_1986.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15353/1/muslim_women_(protection_of_rights_on_divorce)_act,_1986.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Muslim%20Women%20%28Protection%20of%20Rights%20on%20Divorce%29%20Act%2C%201986.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Muslim_Women_Protection_of_Rights_on_Marriage_Act_2019.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/11564/1/a2019-20.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Muslim%20Women%20%28Protection%20of%20Rights%20on%20Marriage%29%20Act%2C%202019.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CORPORATE & COMMERCIAL
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/corporate",
        "Companies_Act_2013.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2114/5/A2013-18.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Companies%20Act%2C%202013.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "Insolvency_and_Bankruptcy_Code_IBC_2016.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15479/1/the_insolvency_and_bankruptcy_code,_2016.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Insolvency%20and%20Bankruptcy%20Code%2C%202016.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "SEBI_Act_1992.pdf",
        [
            # GOV — SEBI official
            "https://www.sebi.gov.in/sebi_data/commondocs/april-2018/sebiact_p.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Securities%20and%20Exchange%20Board%20of%20India%20Act%2C%201992.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "Competition_Act_2002.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2010/5/a2003-12.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Competition%20Act%2C%202002.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "Income_Tax_Act_1961.pdf",
        [
            # GOV — Income Tax India portal (~30 MB, allow extra time)
            "https://incometaxindia.gov.in/Documents/income-tax-act-1961-as-amended-by-finance-act-2024.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Income-tax%20Act%2C%201961.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "Central_Goods_and_Services_Tax_CGST_Act_2017.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15246/1/the_central_goods_and_services_tax_act,_2017.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Central%20Goods%20and%20Services%20Tax%20Act%2C%202017.pdf",
            "https://cbic-gst.gov.in/pdf/central-goods-service-tax-act-2017.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # LABOUR & EMPLOYMENT
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/labour",
        "Industrial_Disputes_Act_1947.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/17112/1/the_industrial_disputes_act.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/11102/1/industrial-disputes-act-1947.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Industrial%20Disputes%20Act%2C%201947.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Factories_Act_1948.pdf",
        [
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Factories%20Act%2C%201948.pdf",
            # GOV — Chief Labour Commissioner
            "https://clc.gov.in/clc/sites/default/files/FactoriesAct1948.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Employees_Provident_Funds_and_MP_Act_1952.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/5437/1/employees_provident_fund_act.pdf",
            # GOV — EPFO official
            "https://www.epfindia.gov.in/site_docs/PDFs/Downloads_PDFs/EPFAct1952.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Employees%27%20Provident%20Funds%20and%20Miscellaneous%20Provisions%20Act%2C%201952.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Maternity_Benefit_Act_1961.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/17115/1/maternity_benefit.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/20954/1/the_maternity_benefit_act,_1961.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Maternity%20Benefit%20Act%2C%201961.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Code_on_Wages_2019.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15793/1/A2019-29.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Code%20on%20Wages%2C%202019.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSUMER, CYBER & INTELLECTUAL PROPERTY
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/consumer_cyber_ip",
        "Consumer_Protection_Act_2019.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15256/1/a2019-35.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Consumer%20Protection%20Act%2C%202019.pdf",
        ],
    ),
    (
        "bare_acts/consumer_cyber_ip",
        "Information_Technology_Act_2000.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Information%20Technology%20Act%2C%202000.pdf",
        ],
    ),
    (
        "bare_acts/consumer_cyber_ip",
        "Digital_Personal_Data_Protection_DPDP_Act_2023.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/22037/1/digital_personal_data_protection_act,_2023.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Digital%20Personal%20Data%20Protection%20Act%2C%202023.pdf",
            # GOV — MEITY
            "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf",
        ],
    ),
    (
        "bare_acts/consumer_cyber_ip",
        "Patents_Act_1970.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1392/1/A1970-39.pdf",
            # GOV — IP India (CGPDTM)
            "https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_69_1_patent-act-1970-11march2015.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Patents%20Act%2C%201970.pdf",
        ],
    ),
    (
        "bare_acts/consumer_cyber_ip",
        "Copyright_Act_1957.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1367/5/a1957-14.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Copyright%20Act%2C%201957.pdf",
        ],
    ),
    (
        "bare_acts/consumer_cyber_ip",
        "Trade_Marks_Act_1999.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/15427/1/the_trade_marks_act,_1999.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Trade%20Marks%20Act%2C%201999.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # PROPERTY & REAL ESTATE
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/property",
        "RERA_Real_Estate_Regulation_and_Development_Act_2016.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2158/3/A2016-16.pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Real%20Estate%20%28Regulation%20and%20Development%29%20Act%2C%202016.pdf",
        ],
    ),
    (
        "bare_acts/property",
        "Right_to_Fair_Compensation_Land_Acquisition_Act_2013.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/19895/1/the_right_to_fair_compensation_and_transparency_in_land_acquisition,_rehabilitation_and_resettlement_act,_2013..pdf",
            "https://lddashboard.legislative.gov.in/sites/default/files/The%20Right%20to%20Fair%20Compensation%20and%20Transparency%20in%20Land%20Acquisition%2C%20Rehabilitation%20and%20Resettlement%20Act%2C%202013.pdf",
        ],
    ),
    (
        "bare_acts/property",
        "Indian_Stamp_Act_1899.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/13205/1/the_indian_stamp_act,_1899.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/20095/1/the_indian_stamp_act,_1899.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # NOTIFICATIONS & EXPLANATORY  (from original download_legal_data.py)
    # ═══════════════════════════════════════════════════════════════════════════
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
    (
        "explanatory",
        "NJA_Organizational_Structure_and_Jurisdiction.pdf",
        [
            "https://nja.gov.in/Concluded_Programmes/2018-19/SE-05_2019_PPTs/6.Organizational%20Structure%20and%20Jurisdiction.pdf",
        ],
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CORPUS EXPANSION (2026-07) — acts referenced by the document validator
    # (Succession, Partnership, PoA, Notaries, Oaths) plus high-traffic statutes.
    # All URLs verified to serve the actual bare-act text with a clean text layer.
    # ═══════════════════════════════════════════════════════════════════════════
    (
        "bare_acts/family",
        "Indian_Succession_Act_1925.pdf",
        [
            "https://thc.nic.in/Central%20Governmental%20Acts/Indian%20Succession%20Act,%201925.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Hindu_Adoptions_and_Maintenance_Act_1956.pdf",
        [
            "https://thc.nic.in/Central%20Governmental%20Acts/Hindu%20Adoptions%20and%20Maintenance%20Act,%201956.pdf",
        ],
    ),
    (
        "bare_acts/family",
        "Hindu_Minority_and_Guardianship_Act_1956.pdf",
        [
            "https://thc.nic.in/Central%20Governmental%20Acts/Hindu%20Minority%20and%20Guardianship%20Act,%201956.pdf",
        ],
    ),
    (
        "bare_acts/commercial",
        "Indian_Partnership_Act_1932.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/9183/1/the_indian_partnership_act_1932.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/2394/1/A1932-9.pdf",
        ],
    ),
    (
        "bare_acts/property",
        "Powers_of_Attorney_Act_1882.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2360/1/a1882-07.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Notaries_Act_1952.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2172/1/A1952-53.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/17288/1/notaries_act_1952.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Oaths_Act_1969.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1643/1/A1969-44.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "General_Clauses_Act_1897.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2328/1/A1897-10.pdf",
        ],
    ),
    (
        "bare_acts/civil",
        "Motor_Vehicles_Act_1988.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/1798/1/eng.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Prevention_of_Corruption_Act_1988.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1558/1/A1988-49.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Juvenile_Justice_Act_2015.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2148/1/a2016-2.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Arms_Act_1959.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1398/1/A1959_54.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/6791/1/arms_act-1959.pdf",
        ],
    ),
    (
        "bare_acts/criminal",
        "Unlawful_Activities_Prevention_Act_UAPA_1967.pdf",
        [
            # /1470/3/ is the English-only consolidated text; /1470/1/ 302s and
            # the thc.nic.in copy is a bilingual gazette (30% Hindi).
            "https://www.indiacode.nic.in/bitstream/123456789/1470/3/A1967-37.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/6853/1/unlawful_activities_prevention_act1967.pdf",
        ],
    ),
    (
        "bare_acts/corporate",
        "SARFAESI_Act_2002.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2006/1/A2002-54.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "POSH_Sexual_Harassment_at_Workplace_Act_2013.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/2104/1/A2013-14.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Payment_of_Gratuity_Act_1972.pdf",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/20959/2/the_payment_of_gratuity_act,_1972.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15318/1/payment-of-gratuity-act-1972.pdf",
        ],
    ),
    (
        "bare_acts/labour",
        "Minimum_Wages_Act_1948.pdf",
        [
            "https://clc.gov.in/clc/sites/default/files/MinimumWagesact.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/1730/1/A1948-011.pdf",
        ],
    ),
    (
        "bare_acts/environment",
        "Wildlife_Protection_Act_1972.pdf",
        [
            "https://thc.nic.in/Central%20Governmental%20Acts/Wild%20Life%20(Protection)%20Act,%201972.pdf",
        ],
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # De-duplicate by destination path (first entry wins)
    seen: set[Path] = set()
    deduped: list[tuple[str, str, list[str]]] = []
    for entry in DOCUMENTS:
        dest = BASE_DIR / entry[0] / entry[1]
        if dest not in seen:
            seen.add(dest)
            deduped.append(entry)

    total = len(deduped)

    log.info("=" * 65)
    log.info("  Unified Indian Legal Documents Downloader")
    log.info(f"  Saving to : {BASE_DIR}")
    log.info(f"  Documents : {total}  (duplicates removed)")
    log.info(f"  Retries   : {MAX_RETRIES} per URL before skipping")
    log.info("=" * 65)

    success = failed = skipped = 0
    failed_list: list[str] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120.0) as client:
        for i, (subdir, filename, urls) in enumerate(deduped, 1):
            dest_dir = ensure_dir(BASE_DIR / subdir)
            dest = dest_dir / filename

            log.info(f"\n[{i}/{total}] {subdir}/{filename}")

            # Already on disk → count as skipped
            if dest.exists() and dest.stat().st_size > MIN_FILE_SIZE_BYTES:
                log.info("  [SKIP] Already downloaded.")
                skipped += 1
                continue

            if download_with_fallbacks(client, urls, dest):
                success += 1
            else:
                failed += 1
                failed_list.append(f"{subdir}/{filename}")
                log.warning("  [SKIP] Moving to next file.")

            time.sleep(0.8)

    # ── Final report ──────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info(
        f"  DONE  ✓ {success} downloaded  "
        f"↷ {skipped} already present  "
        f"✗ {failed} failed"
    )
    log.info("=" * 65)

    if failed_list:
        log.warning(
            f"\n{len(failed_list)} file(s) could not be downloaded automatically:"
        )
        for f in failed_list:
            log.warning(f"  • {f}")
        log.warning(
            "\nTo fetch these manually, search the act name on:\n"
            "  https://lddashboard.legislative.gov.in\n"
            "  https://www.indiacode.nic.in\n"
            "  Relevant ministry websites (MCA, SEBI, CBIC, EPFO, etc.)"
        )

    log.info("\nFull corpus inventory:")
    total_mb = 0.0
    for pdf in sorted(BASE_DIR.rglob("**/*.pdf")):
        rel = pdf.relative_to(BASE_DIR)
        size_mb = pdf.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        log.info(f"  {rel}  ({size_mb:.1f} MB)")
    log.info(f"\n  Total corpus size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
