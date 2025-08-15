import datetime
from services.data.web_scraper import (
    get_top_funds_moneycontrol, get_top_funds_valueresearch, get_top_funds_amfi,
    scrape_moneycontrol_fund_details, scrape_valueresearch_fund_details
)
# Optionally, add more imports for ETMoney, Groww, Mint, etc.

def aggregate_fund_data(fund_name):
    """
    Aggregate mutual fund data from all available free sources for a given fund name.
    Returns a dict with merged fields, each with value, source, and as_of.
    """
    results = []
    # 1. Scrape top lists
    try:
        results.extend(get_top_funds_moneycontrol(20))
    except Exception:
        pass
    try:
        results.extend(get_top_funds_valueresearch(20))
    except Exception:
        pass
    try:
        results.extend(get_top_funds_amfi(20))
    except Exception:
        pass
    # 2. Scrape dedicated fund pages
    try:
        mc = scrape_moneycontrol_fund_details(fund_name)
        if mc: results.append(mc)
    except Exception:
        pass
    try:
        vr = scrape_valueresearch_fund_details(fund_name)
        if vr: results.append(vr)
    except Exception:
        pass
    # 3. Fuzzy match to fund_name
    import difflib
    best = None
    best_score = 0
    for fund in results:
        name = fund.get('scheme_name') or fund.get('fund_name') or ''
        score = difflib.SequenceMatcher(None, fund_name.lower(), name.lower()).ratio()
        if score > best_score:
            best = fund
            best_score = score
    # 4. Merge all fields from all sources
    merged = {}
    fields = ['nav', 'aum', 'expense_ratio', 'benchmark', 'fund_manager', 'manager_tenure', 'top_holdings', 'sector_allocation', 'risk', 'launch_date', 'exit_load', 'as_of', 'fund_link']
    for field in fields:
        # Collect all values for this field
        values = []
        for fund in results:
            val = fund.get(field)
            if val:
                values.append({
                    'value': val,
                    'source': fund.get('source'),
                    'as_of': fund.get('as_of') or fund.get('date')
                })
        if values:
            merged[field] = values
    # 5. Add best fuzzy match as 'primary' if available
    if best:
        merged['primary'] = best
    # 6. Optionally, add news headlines and sentiment (stub for now)
    merged['news'] = []
    merged['sentiment'] = 'neutral'
    return merged 