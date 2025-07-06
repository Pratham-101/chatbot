import json
import re
import os
from collections import defaultdict

INPUT_JSON = "data/all_funds_selenium.json"
OUTPUT_JSON = "data/all_funds_cleaned.json"

# Helper to normalize fund names
RE_ADS = re.compile(r"Sponsored Adv.*?Invest Now", re.IGNORECASE)
RE_WS = re.compile(r"\s+")

def normalize_fund_name(name):
    if not name:
        return ""
    name = RE_ADS.sub("", name)
    name = name.replace("\n", " ")
    name = RE_WS.sub(" ", name)
    return name.strip()

def clean_fund_record(fund):
    # Normalize keys
    cleaned = {}
    for k, v in fund.items():
        key = k.strip().replace(" ", "_").lower()
        cleaned[key] = v.strip() if isinstance(v, str) else v
    # Normalize fund name
    if "fund_name" in cleaned:
        cleaned["fund_name"] = normalize_fund_name(cleaned["fund_name"])
    elif "scheme_name" in cleaned:
        cleaned["fund_name"] = normalize_fund_name(cleaned["scheme_name"])
    # Remove records with missing/invalid fund name
    if not cleaned.get("fund_name") or len(cleaned["fund_name"]) < 3:
        return None
    return cleaned

def remove_duplicates(funds):
    seen = set()
    unique = []
    for fund in funds:
        key = fund["fund_name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(fund)
    return unique

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Input file not found: {INPUT_JSON}")
        return
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    cleaned = []
    for fund in data:
        c = clean_fund_record(fund)
        if c:
            cleaned.append(c)
    print(f"After cleaning: {len(cleaned)} records.")
    unique = remove_duplicates(cleaned)
    print(f"After deduplication: {len(unique)} records.")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Cleaned data saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main() 