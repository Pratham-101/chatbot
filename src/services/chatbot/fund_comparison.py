"""
Fund Comparison Module
Compares two funds side by side with key metrics.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FundComparison:
    fund1_name: str
    fund2_name: str
    fund1_data: Dict
    fund2_data: Dict
    comparison_table: str

class FundComparator:
    """Compares two funds and generates comparison analysis."""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    async def compare_funds(self, fund1_name: str, fund2_name: str) -> FundComparison:
        """Compare two funds and return structured comparison."""
        
        # Fetch fund data
        fund1_data = await self.api_client.get_fund_data(fund1_name)
        fund2_data = await self.api_client.get_fund_data(fund2_name)
        
        # Generate comparison table
        comparison_table = self._generate_comparison_table(fund1_data, fund2_data)
        
        return FundComparison(
            fund1_name=fund1_name,
            fund2_name=fund2_name,
            fund1_data=fund1_data,
            fund2_data=fund2_data,
            comparison_table=comparison_table
        )
    
    def _generate_comparison_table(self, fund1: Dict, fund2: Dict) -> str:
        """Generate markdown comparison table."""
        
        table = """
## Fund Comparison Table

| Metric | Fund 1 | Fund 2 | Difference |
|--------|--------|--------|------------|
"""
        
        metrics = [
            ("NAV", "nav", "₹"),
            ("Expense Ratio", "expense_ratio", "%"),
            ("Risk Category", "sebi_risk_category", ""),
            ("1Y Return", "return_1y", "%"),
            ("3Y Return", "return_3y", "%"),
            ("5Y Return", "return_5y", "%"),
            ("Fund Type", "fund_type", ""),
            ("Fund Subtype", "fund_subtype", "")
        ]
        
        for metric_name, key, unit in metrics:
            val1 = fund1.get(key, "N/A")
            val2 = fund2.get(key, "N/A")
            
            # Calculate difference for numeric values
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val1 - val2
                diff_str = f"{diff:+.2f}{unit}" if unit else f"{diff:+.2f}"
            else:
                diff_str = "N/A"
            
            table += f"| {metric_name} | {val1}{unit} | {val2}{unit} | {diff_str} |\n"
        
        return table 