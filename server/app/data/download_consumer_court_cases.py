#!/usr/bin/env python3
"""
Consumer Court Cases Downloader (India)

Downloads consumer-related case law from Indian Kanoon
and stores structured JSON for dataset generation.
"""

import json
import logging
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent / "consumer_cases"
BASE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_SEARCH_URL = "https://indiankanoon.org/search/?formInput=consumer+protection"

MAX_PAGES = 20  # increase for more data
DELAY = 1.5


# ─────────────────────────────────────────────
# Fetch page
# ─────────────────────────────────────────────
def get_soup(client, url):
    r = client.get(url)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


# ─────────────────────────────────────────────
# Extract case links
# ─────────────────────────────────────────────
def extract_case_links(soup):
    links = []

    for a in soup.select("a.result_title"):
        href = a.get("href")
        if href:
            links.append("https://indiankanoon.org" + href)

    return links


# ─────────────────────────────────────────────
# Extract case content
# ─────────────────────────────────────────────
def extract_case_data(soup):
    try:
        title = soup.find("title").get_text(strip=True)

        # Main judgment text
        content_div = soup.find("div", {"class": "judgments"})
        paragraphs = content_div.find_all("p") if content_div else []

        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {"title": title, "content": content}

    except Exception as e:
        log.warning(f"Parse error: {e}")
        return None


# ─────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────
def save_case(data, idx):
    path = BASE_DIR / f"case_{idx}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    with httpx.Client(headers=HEADERS, timeout=30.0) as client:

        case_id = 0

        for page in range(MAX_PAGES):
            url = BASE_SEARCH_URL + f"&pagenum={page}"
            log.info(f"Fetching page {page}")

            soup = get_soup(client, url)
            links = extract_case_links(soup)

            log.info(f"Found {len(links)} cases")

            for link in links:
                try:
                    case_soup = get_soup(client, link)
                    data = extract_case_data(case_soup)

                    if not data or len(data["content"]) < 200:
                        continue

                    save_case(data, case_id)
                    case_id += 1

                    log.info(f"[OK] Case {case_id}")

                    time.sleep(DELAY)

                except Exception as e:
                    log.warning(f"Failed: {link} | {e}")

        log.info(f"\nDownloaded {case_id} cases")


if __name__ == "__main__":
    main()
