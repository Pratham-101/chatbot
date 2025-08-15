import json
import os
from typing import Optional, Dict, Any, List
import re

class StructuredDataLoader:
    def __init__(self, data_dir: str = "processed_structured_data"):
        self.data_dir = data_dir
        self.data = {}  # fund_name -> list of records
        self._load_all_data()
        # Create a lowercase mapping for case-insensitive lookup
        self.lowercase_map = {k.lower(): k for k in self.data.keys()}

    def _load_all_data(self):
        if not os.path.exists(self.data_dir):
            print(f"Structured data directory {self.data_dir} does not exist.")
            return
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        # Accept a single dict, a list of dicts, or a dict with 'funds'/'data' key
                        if isinstance(records, dict):
                            if 'funds' in records and isinstance(records['funds'], list):
                                records = records['funds']
                            elif 'data' in records and isinstance(records['data'], list):
                                records = records['data']
                            else:
                                # If it's a single fund dict, wrap in a list
                                records = [records]
                        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                            print(f"Warning: {file_path} does not contain a list of dicts. Skipping.")
                            continue
                        for record in records:
                            fund_name = record.get("fund_name")
                            if not fund_name:
                                # Infer fund name from filename if missing
                                base = filename.replace('.json', '')
                                # Remove 'factsheet', months, and years
                                import re
                                base = re.sub(r'factsheet', '', base, flags=re.IGNORECASE)
                                base = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*', '', base, flags=re.IGNORECASE)
                                base = re.sub(r'20\d{2}', '', base)  # Remove years like 2025, 2026
                                base = re.sub(r'\s+', ' ', base).strip()
                                fund_name = base
                                record["fund_name"] = fund_name
                                print(f"[INFO] Inferred fund_name '{fund_name}' from filename {filename}")
                            if fund_name:
                                if fund_name not in self.data:
                                    self.data[fund_name] = []
                                self.data[fund_name].append(record)
                                print(f"Loaded record for fund '{fund_name}' from {filename}")
                except Exception as e:
                    print(f"Error loading structured data from {file_path}: {e}")

    def get_fund_data(self, fund_name: str) -> Optional[List[Dict[str, Any]]]:
        # Case-insensitive exact match
        key = self.lowercase_map.get(fund_name.lower())
        if key:
            return self.data.get(key)
        # Robust fuzzy/token-based match
        def normalize(s):
            return set(re.sub(r'[^a-zA-Z0-9]', ' ', s).lower().split())
        query_tokens = normalize(fund_name)
        for k in self.data.keys():
            k_tokens = normalize(k)
            # If all tokens in query are in k, or vice versa, consider it a match
            if query_tokens and (query_tokens <= k_tokens or k_tokens <= query_tokens):
                print(f"[ROBUST FUZZY MATCH] Matched '{fund_name}' to '{k}'")
                return self.data[k]
        return None

    def get_latest_metric(self, fund_name: str, metric_key: str) -> Optional[Any]:
        records = self.get_fund_data(fund_name)
        if not records:
            return None
        # Sort records by inception_date or other date if available
        def get_date(rec):
            date_str = rec.get("inception_date") or rec.get("date")
            if date_str:
                try:
                    from datetime import datetime
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except Exception:
                    return None
            return None
        records_sorted = sorted(records, key=get_date, reverse=True)
        for rec in records_sorted:
            if metric_key in rec and rec[metric_key] is not None:
                return rec[metric_key]
        return None
