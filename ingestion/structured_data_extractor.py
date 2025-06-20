import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional

class StructuredDataExtractor:
    def __init__(self):
        pass

    def extract_from_excel(self, excel_path: str) -> List[Dict]:
        """
        Extract structured key metrics from Excel factsheet.
        Expected to extract:
        - Fund Name
        - Inception Date
        - NAV (latest and historical)
        - Returns (1yr, 3yr, 5yr CAGR if available)
        - Expense Ratio
        - Other key metrics if present
        """
        df = pd.read_excel(excel_path, sheet_name=None)
        extracted_data = []

        for sheet_name, sheet_df in df.items():
            # Attempt to find relevant columns
            fund_name_col = None
            nav_cols = []
            returns_cols = []
            expense_ratio_col = None
            inception_date_col = None

            # Identify columns heuristically
            for col in sheet_df.columns:
                col_lower = col.lower()
                if "fund name" in col_lower or "scheme name" in col_lower:
                    fund_name_col = col
                if "nav" in col_lower:
                    nav_cols.append(col)
                if "return" in col_lower or "cagr" in col_lower:
                    returns_cols.append(col)
                if "expense ratio" in col_lower:
                    expense_ratio_col = col
                if "inception" in col_lower or "launch" in col_lower:
                    inception_date_col = col

            # Process each row to extract structured info
            for idx, row in sheet_df.iterrows():
                fund_name = row[fund_name_col] if fund_name_col and fund_name_col in row else None
                inception_date = row[inception_date_col] if inception_date_col and inception_date_col in row else None
                expense_ratio = row[expense_ratio_col] if expense_ratio_col and expense_ratio_col in row else None

                nav_data = {}
                for nav_col in nav_cols:
                    nav_data[nav_col] = row[nav_col]

                returns_data = {}
                for ret_col in returns_cols:
                    returns_data[ret_col] = row[ret_col]

                extracted_data.append({
                    "fund_name": fund_name,
                    "inception_date": str(inception_date) if inception_date is not None else None,
                    "expense_ratio": expense_ratio,
                    "nav": nav_data,
                    "returns": returns_data,
                    "source_sheet": sheet_name
                })

        return extracted_data

    def save_structured_data(self, data: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_structured_data(self, input_path: str) -> List[Dict]:
        if not os.path.exists(input_path):
            return []
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_cagr(self, start_value: float, end_value: float, years: float) -> Optional[float]:
        try:
            if start_value <= 0 or end_value <= 0 or years <= 0:
                return None
            cagr = (end_value / start_value) ** (1 / years) - 1
            return round(cagr * 100, 2)  # percentage
        except Exception:
            return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python structured_data_extractor.py <excel_file_path> <output_json_path>")
        sys.exit(1)
    excel_file = sys.argv[1]
    output_json = sys.argv[2]
    extractor = StructuredDataExtractor()
    data = extractor.extract_from_excel(excel_file)
    extractor.save_structured_data(data, output_json)
    print(f"Extracted structured data saved to {output_json}")
