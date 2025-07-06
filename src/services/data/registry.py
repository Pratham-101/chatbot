from typing import Callable, Dict, List, Any

class DataSource:
    def __init__(self, name: str, fetch_func: Callable[[str], Dict[str, Any]], meta: Dict[str, Any]):
        self.name = name
        self.fetch_func = fetch_func
        self.meta = meta

    def fetch(self, fund_query: str) -> Dict[str, Any]:
        return self.fetch_func(fund_query)

# Registry of all data sources
_DATA_SOURCES: List[DataSource] = []

def register_source(source: DataSource):
    _DATA_SOURCES.append(source)

def get_all_sources() -> List[DataSource]:
    return _DATA_SOURCES

def get_data(fund_query: str) -> List[Dict[str, Any]]:
    results = []
    for source in _DATA_SOURCES:
        try:
            data = source.fetch(fund_query)
            if data:
                results.append({"source": source.name, "data": data, "meta": source.meta})
        except Exception as e:
            results.append({"source": source.name, "error": str(e), "meta": source.meta})
    return results 