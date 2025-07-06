import requests
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime

# Example: Scrape Moneycontrol's top funds page (public, no login required)
MONEYC_URL = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/large-cap-fund.html"
VRO_URL = "https://www.valueresearchonline.com/funds/fundSelector/?category=equity&plan=direct&tab=returns"
ET_URL = "https://economictimes.indiatimes.com/mutual-funds/top-mutual-funds"


def fetch_moneycontrol_top_funds(url=MONEYC_URL):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "responsive"})
    funds = []
    if table:
        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 6:
                fund = {
                    "name": cols[0].get_text(strip=True),
                    "aum": cols[1].get_text(strip=True),
                    "1y_return": cols[2].get_text(strip=True),
                    "3y_return": cols[3].get_text(strip=True),
                    "5y_return": cols[4].get_text(strip=True),
                    "category": "Large Cap",
                    "source": url,
                    "scraped_at": datetime.now().isoformat()
                }
                funds.append(fund)
    return funds


def fetch_vro_top_funds(url=VRO_URL):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    funds = []
    # VRO uses JS for main table, but fallback to scraping visible table if present
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                fund = {
                    "name": cols[0].get_text(strip=True),
                    "aum": cols[1].get_text(strip=True),
                    "1y_return": cols[2].get_text(strip=True),
                    "3y_return": cols[3].get_text(strip=True),
                    "5y_return": cols[4].get_text(strip=True),
                    "category": "Equity",
                    "source": url,
                    "scraped_at": datetime.now().isoformat()
                }
                funds.append(fund)
    return funds


def fetch_et_top_funds(url=ET_URL):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    funds = []
    # ET has multiple tables, try to find the first one
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                fund = {
                    "name": cols[0].get_text(strip=True),
                    "aum": cols[1].get_text(strip=True),
                    "1y_return": cols[2].get_text(strip=True),
                    "3y_return": cols[3].get_text(strip=True),
                    "5y_return": cols[4].get_text(strip=True),
                    "category": "Equity",
                    "source": url,
                    "scraped_at": datetime.now().isoformat()
                }
                funds.append(fund)
    return funds


def save_funds_json(funds, out_path):
    with open(out_path, "w") as f:
        json.dump(funds, f, indent=2)
    print(f"Saved {len(funds)} funds to {out_path}")


def main():
    all_funds = []
    # Moneycontrol Large Cap
    all_funds.extend(fetch_moneycontrol_top_funds())
    # ValueResearchOnline (if table is present)
    all_funds.extend(fetch_vro_top_funds())
    # Economic Times (if table is present)
    all_funds.extend(fetch_et_top_funds())
    # (Add more sources here as needed)
    save_funds_json(all_funds, "data/top_funds_latest.json")

if __name__ == "__main__":
    main() 