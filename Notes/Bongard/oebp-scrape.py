#!/usr/bin/env python3
"""
Scrape the On-Line Encyclopedia of Bongard Problems (OEBP) and build
`oebp.csv` containing the columns:

    url, problem_id, description, comments, example, concepts

Features
--------
* Polite headers (realistic User-Agent, explicit Accept fields).
* Random delay **5–10 s** between page requests to avoid hammering the server.
* Progress bar (tqdm) and colourised status messages.
* Graceful skip on HTTP errors (row is not written).

Usage
-----
    pip install beautifulsoup4 requests tqdm colorama
    python oebp_scraper.py             # creates oebp.csv in current dir

Adjust START / END if OEBP adds new problems.
"""
import csv
import random
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style
from tqdm import tqdm

# -----------------------------------------------------------
BASE_URL = "https://oebp.org"
START, END = 1, 1290         # scrape BP1 … BP1290 (inclusive)
OUTPUT_CSV = Path("oebp.csv")

HEADERS = {
    # Emulate a mainstream desktop browser
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

TIMEOUT_S = 20               # request timeout seconds
DELAY_MIN, DELAY_MAX = 5, 10 # random sleep bounds (seconds)

# -----------------------------------------------------------

def clean(text: str) -> str:
    """Collapse runs of whitespace and strip ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_page(html: str, bp_id: str):
    """Return a list matching the CSV columns for one page."""
    soup = BeautifulSoup(html, "html.parser")

    # Description (immediately right of the BP# link)
    link = soup.find("a", href=f"/{bp_id}")
    if not link:
        raise ValueError("BP link not found in page")
    descr_td = link.find_next("td")
    description = clean(descr_td.text)

    def grab(section: str) -> str:
        lab = soup.find(
            "font",
            string=lambda s: s and s.strip().upper() == section.upper(),
        )
        if not lab:
            return ""
        cell = lab.parent.find_next("td", width="600")
        return clean(cell.text)

    comments  = grab("COMMENTS")
    example   = grab("EXAMPLE")
    concepts  = ", ".join(
        [c.text.capitalize() for c in soup.select("a.conceptlink")]
    )

    url = f"{BASE_URL}/{bp_id}"
    return [url, bp_id, description, comments, example, concepts]


# -----------------------------------------------------------

def main():
    if OUTPUT_CSV.exists():
        print(f"{Fore.YELLOW}[!] {OUTPUT_CSV} exists – aborting to avoid overwrite.{Style.RESET_ALL}")
        sys.exit(1)

    session = requests.Session()
    session.headers.update(HEADERS)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "problem_id", "description", "comments", "example", "concepts"])

        for n in tqdm(range(START, END + 1), unit="BP", desc="Scraping OEBP"):
            bp_id = f"BP{n}"
            url = f"{BASE_URL}/{bp_id}"

            try:
                resp = session.get(url, timeout=TIMEOUT_S, allow_redirects=True)
                status = resp.status_code

                if status != 200:
                    tqdm.write(f"{Fore.CYAN}skip {bp_id}: {status}{Style.RESET_ALL}")
                    continue

                row = parse_page(resp.text, bp_id)
                writer.writerow(row)

            except Exception as exc:
                tqdm.write(f"{Fore.RED}error {bp_id}: {exc}{Style.RESET_ALL}")

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    print(f"{Fore.GREEN}✓ Done. CSV saved to {OUTPUT_CSV}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
