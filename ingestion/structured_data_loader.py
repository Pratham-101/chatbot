import json
import os
from typing import Optional, Dict, Any, List

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
                        for record in records:
                            fund_name = record.get("fund_name")
                            if fund_name:
                                if fund_name not in self.data:
                                    self.data[fund_name] = []
                                self.data[fund_name].append(record)
                except Exception as e:
                    print(f"Error loading structured data from {file_path}: {e}")

    def get_fund_data(self, fund_name: str) -> Optional[List[Dict[str, Any]]]:
        # Case-insensitive lookup
        key = self.lowercase_map.get(fund_name.lower())
        if key:
            return self.data.get(key)
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
