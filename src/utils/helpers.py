import re
from typing import Optional

def extract_fund_name(query: str) -> Optional[str]:
    """Extract fund name from query."""
    patterns = [
        r'(HDFC.*?Fund)',
        r'(ICICI.*?Fund)',
        r'(SBI.*?Fund)',
        r'(Kotak.*?Fund)',
        r'(Nippon.*?Fund)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
