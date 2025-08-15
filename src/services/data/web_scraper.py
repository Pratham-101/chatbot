import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MutualFundBot/1.0)"}

# --- AMFI ---
def get_top_funds_amfi(limit=10):
    """Fetch top funds from AMFI NAVAll.txt (latest NAVs)."""
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    lines = resp.text.splitlines()
    funds = []
    for line in lines:
        parts = line.split(';')
        if len(parts) >= 6 and parts[0].isdigit():
            fund = {
                'scheme_code': parts[0],
                'isin': parts[1],
                'scheme_name': parts[3],
                'nav': parts[4],
                'date': parts[5],
                'source': 'AMFI',
            }
            funds.append(fund)
    # Sort by NAV (descending) and take top N
    funds = sorted(funds, key=lambda x: float(x['nav']) if x['nav'].replace('.','',1).isdigit() else 0, reverse=True)
    return funds[:limit]

# --- Moneycontrol ---
def get_top_funds_moneycontrol(limit=10):
    """Fetch top funds from Moneycontrol (top equity funds page)."""
    url = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/large-cap-fund.html"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'class': 'responsive'} )
    funds = []
    if table:
        df = pd.read_html(str(table))[0]
        for _, row in df.iterrows():
            fund = {
                'scheme_name': row.get('Scheme Name') or row.get('Scheme'),
                '1y_return': row.get('1-Year'),
                '3y_return': row.get('3-Year'),
                '5y_return': row.get('5-Year'),
                'aum': row.get('AUM (Cr)'),
                'nav': row.get('NAV'),
                'source': 'Moneycontrol',
                'date': datetime.now().strftime('%Y-%m-%d'),
            }
            funds.append(fund)
    return funds[:limit]

# --- Value Research ---
def get_top_funds_valueresearch(limit=10):
    """Fetch top funds from Value Research (top equity funds page)."""
    url = "https://www.valueresearchonline.com/funds/fundSelector/default.asp?category=equity&plan=direct&option=growth"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = pd.read_html(resp.text)
    funds = []
    if tables:
        df = tables[0]
        for _, row in df.iterrows():
            fund = {
                'scheme_name': row.get('Fund Name') or row.get('Scheme'),
                '1y_return': row.get('1-Year'),
                '3y_return': row.get('3-Year'),
                '5y_return': row.get('5-Year'),
                'aum': row.get('AUM (Cr)'),
                'nav': row.get('NAV'),
                'source': 'Value Research',
                'date': datetime.now().strftime('%Y-%m-%d'),
            }
            funds.append(fund)
    return funds[:limit]

def _parse_nav(nav_str):
    """Parse NAV string and return float if in valid range, else None."""
    import re
    nav_str = nav_str.replace(",", "").replace("₹", "").strip()
    try:
        nav = float(re.findall(r"[\d.]+", nav_str)[0])
        if 1 <= nav <= 100000:
            return nav
        else:
            print(f"[DEBUG] Ignoring out-of-range NAV: {nav_str}")
            return None
    except Exception as e:
        print(f"[DEBUG] NAV parse error: {nav_str} ({e})")
        return None

def _parse_aum(aum_str):
    """Parse AUM string and return float in Cr if possible."""
    import re
    aum_str = aum_str.replace(",", "").replace("₹", "").strip().lower()
    try:
        if "cr" in aum_str:
            val = float(re.findall(r"[\d.]+", aum_str)[0])
            return val
        elif "lakh" in aum_str:
            val = float(re.findall(r"[\d.]+", aum_str)[0]) / 100
            return val
        else:
            # Try to guess if it's a raw number (e.g., 1000000000)
            val = float(re.findall(r"[\d.]+", aum_str)[0])
            if val > 1e7:
                return round(val / 1e7, 2)  # Convert to Cr
            return val
    except Exception as e:
        print(f"[DEBUG] AUM parse error: {aum_str} ({e})")
        return None

def _find_nav_candidates(soup):
    """Find all candidate NAV values from the soup for debugging and robust extraction."""
    import re
    candidates = []
    # Look for common NAV labels
    nav_labels = [
        'NAV per unit', 'Net Asset Value', 'NAV', 'NAV (Rs.)', 'NAV (₹)', 'NAV as on', 'NAV as at'
    ]
    for label in nav_labels:
        for tag in soup.find_all(text=re.compile(label, re.I)):
            # Try to find a number in the same tag or nearby
            parent = tag.parent
            text = parent.get_text(" ", strip=True)
            nums = re.findall(r"[\d,.]+", text)
            for n in nums:
                nav = _parse_nav(n)
                if nav:
                    candidates.append((label, nav, text))
            # Also check next siblings
            sib = parent.find_next_sibling()
            if sib:
                nums = re.findall(r"[\d,.]+", sib.get_text(" ", strip=True))
                for n in nums:
                    nav = _parse_nav(n)
                    if nav:
                        candidates.append((label, nav, sib.get_text(" ", strip=True)))
    # Also, as fallback, look for any number in a span with id containing 'nav'
    for span in soup.find_all('span'):
        if 'nav' in (span.get('id') or '').lower():
            nav = _parse_nav(span.get_text(strip=True))
            if nav:
                candidates.append(('span-id', nav, span.get_text(strip=True)))
    print(f"[DEBUG] NAV candidates found: {candidates}")
    return candidates

def scrape_moneycontrol_fund_details(fund_name):
    """
    Scrape Moneycontrol for fund manager(s), tenure, and top holdings for a given fund name.
    Returns a dict with keys: fund_manager, manager_tenure, top_holdings, nav, aum, as_of, source.
    """
    import requests
    from bs4 import BeautifulSoup
    base_url = "https://www.moneycontrol.com"
    search_url = f"https://www.moneycontrol.com/mutual-funds/search/?q={fund_name.replace(' ', '+')}"
    headers = HEADERS
    resp = requests.get(search_url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Find first fund link
    fund_link = None
    for a in soup.find_all('a', href=True):
        if '/mutual-funds/nav/' in a['href']:
            fund_link = a['href']
            break
    if not fund_link:
        return {}
    fund_url = base_url + fund_link
    resp = requests.get(fund_url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Fund Manager(s)
    manager = None
    tenure = None
    for div in soup.find_all('div', class_='fundmanager'):  # class may change
        text = div.get_text(separator=' ', strip=True)
        m = re.search(r'Fund Manager[s]?:\s*([A-Za-z ,.&-]+)', text)
        if m:
            manager = m.group(1)
        t = re.search(r'Tenure: ([^|]+)', text)
        if t:
            tenure = t.group(1)
    # Top Holdings
    holdings = []
    table = soup.find('table', {'class': 'mctable1'})
    if table:
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if cols and len(cols) > 1:
                holdings.append(cols[0].get_text(strip=True))
    # NAV extraction (improved)
    nav_candidates = _find_nav_candidates(soup)
    nav_val = None
    if nav_candidates:
        # Pick the candidate closest to ₹10–₹10,000
        nav_val = min(nav_candidates, key=lambda x: abs(x[1] - 100)) if len(nav_candidates) > 1 else nav_candidates[0][1]
        if isinstance(nav_val, tuple):
            nav_val = nav_val[1]
    # AUM
    aum_raw = None
    aum_val = None
    aum_tag = soup.find('span', {'id': 'aum_val'})
    if aum_tag:
        aum_raw = aum_tag.text.strip()
        aum_val = _parse_aum(aum_raw)
    # as_of
    as_of = None
    date_tag = soup.find(text=re.compile(r'As on'))
    if date_tag:
        as_of = date_tag.parent.get_text(strip=True)
    return {
        'fund_manager': manager,
        'manager_tenure': tenure,
        'top_holdings': ', '.join(holdings) if holdings else None,
        'nav': nav_val,
        'aum': aum_val,
        'as_of': as_of,
        'source': fund_url,
        'fund_link': fund_url
    }

def scrape_valueresearch_fund_details(fund_name):
    """
    Scrape ValueResearchOnline for fund manager(s), tenure, and top holdings for a given fund name.
    Returns a dict with keys: fund_manager, manager_tenure, top_holdings, nav, aum, as_of, source.
    """
    import requests
    from bs4 import BeautifulSoup
    base_url = "https://www.valueresearchonline.com"
    search_url = f"https://www.valueresearchonline.com/funds/search/?q={fund_name.replace(' ', '+')}"
    headers = HEADERS
    resp = requests.get(search_url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Find first fund link
    fund_link = None
    for a in soup.find_all('a', href=True):
        if '/funds/' in a['href']:
            fund_link = base_url + a['href']
            break
    if not fund_link:
        return {}
    fund_resp = requests.get(fund_link, headers=headers, timeout=15)
    fund_soup = BeautifulSoup(fund_resp.text, 'html.parser')
    # Fund Manager(s)
    manager = None
    tenure = None
    manager_tag = fund_soup.find(text=re.compile(r'Fund Manager'))
    if manager_tag:
        manager = manager_tag.parent.find_next('span').get_text(strip=True)
    tenure_tag = fund_soup.find(text=re.compile(r'Tenure'))
    if tenure_tag:
        tenure = tenure_tag.parent.find_next('span').get_text(strip=True)
    # Top Holdings
    holdings = []
    table = fund_soup.find('table', {'class': 'fund-holdings-table'})
    if table:
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if cols and len(cols) > 1:
                holdings.append(cols[0].get_text(strip=True))
    # NAV
    nav_raw = None
    nav_val = None
    nav_tag = fund_soup.find(text=re.compile(r'NAV'))
    if nav_tag:
        nav_val = _parse_nav(nav_tag.find_next('span').get_text(strip=True))
    # AUM
    aum_raw = None
    aum_val = None
    aum_tag = fund_soup.find(text=re.compile(r'AUM'))
    if aum_tag:
        aum_val = _parse_aum(aum_tag.find_next('span').get_text(strip=True))
    # as_of
    as_of = None
    date_tag = fund_soup.find(text=re.compile(r'As on'))
    if date_tag:
        as_of = date_tag.parent.get_text(strip=True)
    return {
        'fund_manager': manager,
        'manager_tenure': tenure,
        'top_holdings': ', '.join(holdings) if holdings else None,
        'nav': nav_val,
        'aum': aum_val,
        'as_of': as_of,
        'source': fund_link,
        'fund_link': fund_link
    } 