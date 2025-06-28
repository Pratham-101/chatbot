import os
import json
from typing import Dict, Any, Optional

class MutualFundKnowledgeGraph:
    """
    Lightweight in-memory knowledge graph for mutual fund attributes.
    Supports fast lookup and update of fund details.
    """
    def __init__(self, structured_data_dir: str = "processed_structured_data"):
        self.structured_data_dir = structured_data_dir
        self.graph: Dict[str, Dict[str, Any]] = {}
        self._load_from_structured_data()

    def _load_from_structured_data(self):
        if not os.path.exists(self.structured_data_dir):
            print(f"Knowledge graph: structured data dir '{self.structured_data_dir}' not found.")
            return
        for filename in os.listdir(self.structured_data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.structured_data_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        for record in records:
                            fund_name = record.get("fund_name")
                            if fund_name:
                                self.graph[fund_name.lower()] = record
                except Exception as e:
                    print(f"Knowledge graph: error loading {file_path}: {e}")

    def get_fund(self, fund_name: str) -> Optional[Dict[str, Any]]:
        return self.graph.get(fund_name.lower())

    def update_fund(self, fund_name: str, attributes: Dict[str, Any]):
        key = fund_name.lower()
        if key in self.graph:
            self.graph[key].update(attributes)
        else:
            self.graph[key] = attributes

    def all_funds(self):
        return list(self.graph.keys()) 