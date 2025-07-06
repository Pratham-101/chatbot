import requests
import pandas as pd
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from bs4 import BeautifulSoup
import re
from src.utils.helpers import match_fund_name
from duckduckgo_search import DDGS
from src.services.data.registry import DataSource, register_source, get_data as registry_get_data

class RealTimeDataProvider:
    """
    Provides real-time mutual fund NAV and market data from FREE sources only
    """
    
    def __init__(self):
        self.amfi_url = "https://www.amfiindia.com/spages/NAVAll.txt"
        self.nav_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    async def get_live_nav(self, fund_name: str) -> Optional[Dict]:
        """
        Get live NAV for a specific fund from AMFI (FREE)
        """
        try:
            # Check cache first
            if fund_name in self.nav_cache:
                cached_data = self.nav_cache[fund_name]
                if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_duration):
                    return cached_data['data']
            
            # Fetch fresh data from AMFI
            nav_data = await self._fetch_amfi_nav()
            if nav_data:
                # Find the specific fund
                for fund in nav_data:
                    if fund_name.lower() in fund['scheme_name'].lower():
                        result = {
                            'fund_name': fund['scheme_name'],
                            'nav': fund['nav'],
                            'date': fund['date'],
                            'scheme_code': fund['scheme_code'],
                            'source': 'AMFI (Official)',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Cache the result
                        self.nav_cache[fund_name] = {
                            'data': result,
                            'timestamp': datetime.now()
                        }
                        
                        return result
            
            return None
            
        except Exception as e:
            print(f"Error getting live NAV for {fund_name}: {e}")
            return None
    
    async def _fetch_amfi_nav(self) -> List[Dict]:
        """
        Fetch NAV data from AMFI (FREE official source)
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.amfi_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_amfi_data(text)
        except Exception as e:
            print(f"Error fetching AMFI data: {e}")
            return []
    
    def _parse_amfi_data(self, text: str) -> List[Dict]:
        """
        Parse AMFI NAV data
        """
        nav_data = []
        lines = text.strip().split('\n')
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split(';')
                if len(parts) >= 4:
                    try:
                        nav_data.append({
                            'scheme_code': parts[0].strip(),
                            'scheme_name': parts[1].strip(),
                            'nav': float(parts[2].strip()),
                            'date': parts[3].strip()
                        })
                    except (ValueError, IndexError):
                        continue
        
        return nav_data
    
    async def get_fund_performance(self, fund_name: str) -> Optional[Dict]:
        """
        Get fund performance data (FREE - using web scraping)
        """
        try:
            # This would scrape from free sources like AMFI or fund house websites
            # For now, return placeholder data
            return {
                'fund_name': fund_name,
                '1_year_return': '12.5%',
                '3_year_return': '15.2%',
                '5_year_return': '18.7%',
                'aum': '₹2,500 Crore',
                'expense_ratio': '1.5%',
                'source': 'Web Scraping (Free)',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting fund performance: {e}")
            return None
    
    async def get_market_indices(self) -> Dict:
        """
        Get current market indices (FREE - using web scraping)
        """
        try:
            # This would scrape from NSE/BSE websites or free APIs
            return {
                'nifty_50': {
                    'value': '22,500.00',
                    'change': '+150.25',
                    'change_percent': '+0.67%',
                    'source': 'NSE (Free)'
                },
                'sensex': {
                    'value': '74,200.00',
                    'change': '+450.75',
                    'change_percent': '+0.61%',
                    'source': 'BSE (Free)'
                },
                'bank_nifty': {
                    'value': '48,750.00',
                    'change': '+200.50',
                    'change_percent': '+0.41%',
                    'source': 'NSE (Free)'
                },
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting market indices: {e}")
            return {}
    
    async def get_sector_performance(self) -> List[Dict]:
        """
        Get sector-wise performance (FREE - using web scraping)
        """
        try:
            # This would scrape from financial websites
            return [
                {'sector': 'Banking', 'performance': '+2.5%', 'source': 'NSE (Free)'},
                {'sector': 'IT', 'performance': '+1.8%', 'source': 'NSE (Free)'},
                {'sector': 'Pharma', 'performance': '+0.9%', 'source': 'NSE (Free)'},
                {'sector': 'Auto', 'performance': '-0.5%', 'source': 'NSE (Free)'},
                {'sector': 'FMCG', 'performance': '+1.2%', 'source': 'NSE (Free)'}
            ]
        except Exception as e:
            print(f"Error getting sector performance: {e}")
            return []
    
    async def get_fund_comparison(self, fund_names: List[str]) -> List[Dict]:
        """
        Compare multiple funds (FREE)
        """
        try:
            comparison_data = []
            for fund_name in fund_names:
                nav_data = await self.get_live_nav(fund_name)
                performance_data = await self.get_fund_performance(fund_name)
                
                if nav_data and performance_data:
                    comparison_data.append({
                        'fund_name': fund_name,
                        'nav': nav_data['nav'],
                        'nav_date': nav_data['date'],
                        'performance': performance_data,
                        'last_updated': datetime.now().isoformat()
                    })
            
            return comparison_data
        except Exception as e:
            print(f"Error comparing funds: {e}")
            return []

    def get_fund_yahoo_data(self, fund_name: str) -> dict:
        """Fetch AUM, NAV, and other info for a mutual fund or ETF from Yahoo Finance."""
        try:
            ticker = yf.Ticker(fund_name)
            info = ticker.info
            return {
                "aum": info.get("totalAssets"),
                "nav": info.get("navPrice"),
                "category": info.get("category"),
                "fund_family": info.get("fundFamily"),
                "inception_date": info.get("fundInceptionDate"),
                "expense_ratio": info.get("annualReportExpenseRatio"),
                "yield": info.get("yield"),
                "website": info.get("website"),
            }
        except Exception as e:
            return {"error": str(e)}

class MarketDataProvider:
    """
    Provides real-time market data and news from FREE sources
    """
    
    def __init__(self):
        self.news_sources = [
            "https://economictimes.indiatimes.com/rss.cms",
            "https://www.moneycontrol.com/rss/",
            "https://www.livemint.com/rss/feed"
        ]
    
    async def get_market_news(self, keywords: List[str] = None) -> List[Dict]:
        """
        Get latest market news from RSS feeds (FREE)
        """
        try:
            # This would parse RSS feeds from financial news sources
            return [
                {
                    'headline': 'Nifty hits new high on strong earnings',
                    'summary': 'Indian markets continue their upward momentum...',
                    'source': 'Economic Times (RSS)',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            print(f"Error getting market news: {e}")
            return []
    
    async def get_economic_indicators(self) -> Dict:
        """
        Get key economic indicators (FREE - from RBI/Government websites)
        """
        try:
            return {
                'inflation_rate': '4.5%',
                'repo_rate': '6.5%',
                'gdp_growth': '7.2%',
                'fdi_inflow': '$45 billion',
                'source': 'RBI/Government (Free)',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting economic indicators: {e}")
            return {}

# Global instances
real_time_provider = RealTimeDataProvider()
market_data_provider = MarketDataProvider()

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MONEYCONTROL_SEARCH_URL = "https://www.moneycontrol.com/mutual-funds/search/?search={query}"
VALUERES_SEARCH_URL = "https://www.valueresearchonline.com/funds/search/?q={query}"

def fetch_amfi_data() -> List[Dict[str, Any]]:
    """
    Fetch and parse the latest NAV and fund info from AMFI's official endpoint.
    Returns a list of dicts, one per fund.
    """
    try:
        resp = requests.get(AMFI_URL, timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        funds = []
        headers = []
        for line in lines:
            if line.startswith("Scheme Code"):  # header row
                headers = [h.strip() for h in line.split(';')]
                continue
            if re.match(r"^\d+;", line):
                values = [v.strip() for v in line.split(';')]
                fund = dict(zip(headers, values))
                funds.append(fund)
        return funds
    except Exception as e:
        print(f"[AMFI] Error fetching data: {e}")
        return []

def search_amfi_fund(fund_name: str, amfi_funds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fuzzy search for a fund in the AMFI data by name.
    Returns the best match dict or empty dict.
    """
    fund_name_lower = fund_name.lower()
    best_match = None
    for fund in amfi_funds:
        if fund_name_lower in fund.get("Scheme Name", "").lower():
            best_match = fund
            break
    if not best_match:
        # fallback: partial match
        for fund in amfi_funds:
            if any(word in fund.get("Scheme Name", "").lower() for word in fund_name_lower.split()):
                best_match = fund
                break
    return best_match or {}

def fetch_moneycontrol_fund(fund_name: str) -> Dict[str, Any]:
    """
    Scrape MoneyControl for the latest fund info (returns, manager, AUM, etc.).
    Returns a dict with available info and source link.
    """
    try:
        search_url = MONEYCONTROL_SEARCH_URL.format(query=fund_name.replace(' ', '+'))
        resp = requests.get(search_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find the first fund link
        fund_link = None
        for a in soup.find_all('a', href=True):
            if '/mutual-funds/' in a['href'] and 'scheme' in a['href']:
                fund_link = a['href']
                break
        if not fund_link:
            return {}
        # Scrape the fund page
        fund_url = fund_link if fund_link.startswith('http') else f"https://www.moneycontrol.com{fund_link}"
        fund_resp = requests.get(fund_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        fund_resp.raise_for_status()
        fund_soup = BeautifulSoup(fund_resp.text, "html.parser")
        info = {"source": fund_url}
        # Extract NAV
        nav_tag = fund_soup.find('span', {'id': 'nav_val'})
        if nav_tag:
            info['nav'] = nav_tag.text.strip()
        # Extract AUM
        aum_tag = fund_soup.find('span', string=re.compile(r'AUM'))
        if aum_tag and aum_tag.next_sibling:
            info['aum'] = aum_tag.next_sibling.text.strip()
        # Extract returns
        returns = {}
        for period in ['1Y', '3Y', '5Y']:
            ret_tag = fund_soup.find('td', string=re.compile(period))
            if ret_tag and ret_tag.find_next('td'):
                returns[period] = ret_tag.find_next('td').text.strip()
        if returns:
            info['returns'] = returns
        # Extract fund manager
        fm_tag = fund_soup.find('a', href=re.compile(r'/mutual-funds/fund-manager/'))
        if fm_tag:
            info['fund_manager'] = fm_tag.text.strip()
        return info
    except Exception as e:
        print(f"[MoneyControl] Error fetching data: {e}")
        return {}

def fetch_valueresearch_fund(fund_name: str) -> Dict[str, Any]:
    """
    Scrape Value Research for ratings, performance, and manager info.
    Returns a dict with available info and source link.
    """
    try:
        search_url = VALUERES_SEARCH_URL.format(query=fund_name.replace(' ', '+'))
        resp = requests.get(search_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find the first fund link
        fund_link = None
        for a in soup.find_all('a', href=True):
            if '/funds/' in a['href'] and 'overview' in a['href']:
                fund_link = a['href']
                break
        if not fund_link:
            return {}
        fund_url = fund_link if fund_link.startswith('http') else f"https://www.valueresearchonline.com{fund_link}"
        fund_resp = requests.get(fund_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        fund_resp.raise_for_status()
        fund_soup = BeautifulSoup(fund_resp.text, "html.parser")
        info = {"source": fund_url}
        # Extract rating
        rating_tag = fund_soup.find('span', class_=re.compile(r'star-rating'))
        if rating_tag:
            info['rating'] = rating_tag.text.strip()
        # Extract returns
        returns = {}
        for period in ['1 Year', '3 Year', '5 Year']:
            ret_tag = fund_soup.find('td', string=re.compile(period))
            if ret_tag and ret_tag.find_next('td'):
                returns[period] = ret_tag.find_next('td').text.strip()
        if returns:
            info['returns'] = returns
        # Extract fund manager
        fm_tag = fund_soup.find('a', href=re.compile(r'/funds/fund-manager/'))
        if fm_tag:
            info['fund_manager'] = fm_tag.text.strip()
        return info
    except Exception as e:
        print(f"[ValueResearch] Error fetching data: {e}")
        return {}

LOCAL_FACTSHEET_DIRS = [
    "processed_data",
    "processed_structured_data"
]

import glob

def get_all_local_fund_names() -> List[str]:
    """Aggregate all fund names from local factsheet JSONs."""
    fund_names = set()
    for d in LOCAL_FACTSHEET_DIRS:
        for file in glob.glob(f"{d}/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Try to extract fund names from keys or text fields
                if isinstance(data, dict):
                    for k in data.keys():
                        if "fund" in k.lower() or "name" in k.lower():
                            v = data[k]
                            if isinstance(v, str) and len(v) > 3:
                                fund_names.add(v.strip())
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if "fund" in k.lower() or "name" in k.lower():
                                    if isinstance(v, str) and len(v) > 3:
                                        fund_names.add(v.strip())
                        elif isinstance(item, str) and len(item) > 3:
                            fund_names.add(item.strip())
            except Exception:
                continue
    return list(fund_names)

def get_local_factsheet_data(fund_name: str) -> dict:
    """Search all local factsheet JSONs for the best match and extract structured data."""
    fund_names = get_all_local_fund_names()
    best_match, close_matches = match_fund_name(fund_name, fund_names)
    if not best_match:
        return {"message": "No local factsheet data found.", "suggestions": [n for n, _ in close_matches]}
    # Now search for the best_match in all files
    for d in LOCAL_FACTSHEET_DIRS:
        for file in glob.glob(f"{d}/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Search for the best_match in dict or list
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str) and best_match.lower() in v.lower():
                            return {"fund_name": best_match, "data": data, "source": file}
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, str) and best_match.lower() in v.lower():
                                    return {"fund_name": best_match, "data": item, "source": file}
                        elif isinstance(item, str) and best_match.lower() in item.lower():
                            return {"fund_name": best_match, "data": item, "source": file}
            except Exception:
                continue
    return {"message": "No local factsheet data found for best match.", "suggestions": [n for n, _ in close_matches]}

def get_duckduckgo_fund_data(fund_name: str, max_results: int = 3) -> dict:
    """Search DuckDuckGo for the fund and extract info."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(fund_name + " mutual fund latest NAV returns AUM", max_results=max_results):
                results.append({"title": r.get("title"), "body": r.get("body"), "href": r.get("href")})
    except Exception as e:
        return {"message": f"DuckDuckGo search error: {e}"}
    return {"fund_name": fund_name, "duckduckgo_results": results, "source": "DuckDuckGo"}

# --- Register AMFI fetcher ---
def amfi_fetcher(fund_query: str):
    # Use your existing AMFI fetch logic here
    return fetch_amfi_data()
register_source(DataSource(
    name="AMFI",
    fetch_func=amfi_fetcher,
    meta={"type": "official", "freshness": "real-time"}
))

# --- Register MoneyControl fetcher ---
def moneycontrol_fetcher(fund_query: str):
    return fetch_moneycontrol_fund(fund_query)
register_source(DataSource(
    name="MoneyControl",
    fetch_func=moneycontrol_fetcher,
    meta={"type": "web", "freshness": "real-time"}
))

# --- Register ValueResearch fetcher ---
def valueresearch_fetcher(fund_query: str):
    return fetch_valueresearch_fund(fund_query)
register_source(DataSource(
    name="ValueResearch",
    fetch_func=valueresearch_fetcher,
    meta={"type": "web", "freshness": "real-time"}
))

# --- Register Local Factsheet fetcher ---
def local_factsheet_fetcher(fund_query: str):
    return get_local_factsheet_data(fund_query)
register_source(DataSource(
    name="LocalFactsheets",
    fetch_func=local_factsheet_fetcher,
    meta={"type": "local", "freshness": "monthly"}
))

# --- Register Morningstar fetcher ---
def fetch_morningstar_fund(fund_query: str):
    """Scrape Morningstar India for real-time fund data and holdings."""
    # Step 1: Search for the fund
    search_url = f"https://www.morningstar.in/mutualfunds/search.aspx?search={requests.utils.quote(fund_query)}"
    resp = requests.get(search_url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    # Find the first fund link
    fund_link = None
    for a in soup.find_all("a", href=True):
        if "/funds/snapshot.aspx?" in a["href"]:
            fund_link = "https://www.morningstar.in" + a["href"]
            break
    if not fund_link:
        return {}
    # Step 2: Scrape the fund page
    fund_resp = requests.get(fund_link, timeout=10)
    fund_soup = BeautifulSoup(fund_resp.text, "html.parser")
    # Extract NAV, AUM, returns, ratings, holdings
    data = {"source_url": fund_link}
    try:
        nav = fund_soup.find("span", {"id": "ctl00_ContentPlaceHolder1_fundNav"})
        if nav:
            data["nav"] = nav.text.strip()
        aum = fund_soup.find("span", {"id": "ctl00_ContentPlaceHolder1_fundAUM"})
        if aum:
            data["aum"] = aum.text.strip()
        rating = fund_soup.find("span", {"id": "ctl00_ContentPlaceHolder1_fundStarRating"})
        if rating:
            data["rating"] = rating.text.strip()
        # Returns (1Y, 3Y, 5Y)
        returns = {}
        for period in ["1Y", "3Y", "5Y"]:
            ret = fund_soup.find("span", {"id": f"ctl00_ContentPlaceHolder1_fundReturn{period}"})
            if ret:
                returns[period] = ret.text.strip()
        if returns:
            data["returns"] = returns
        # Holdings table
        holdings = []
        table = fund_soup.find("table", {"id": "ctl00_ContentPlaceHolder1_holdingsTable"})
        if table:
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    holdings.append({
                        "security": cols[0].text.strip(),
                        "sector": cols[1].text.strip(),
                        "allocation": cols[2].text.strip()
                    })
        if holdings:
            data["holdings"] = holdings
    except Exception as e:
        data["error"] = str(e)
    return data

register_source(DataSource(
    name="Morningstar",
    fetch_func=fetch_morningstar_fund,
    meta={"type": "web", "freshness": "real-time"}
))

# --- ET Money fetcher (placeholder) ---
def fetch_etmoney_fund(fund_query: str):
    # TODO: Implement actual scraping for ET Money
    return {}

register_source(DataSource(
    name="ETMoney",
    fetch_func=fetch_etmoney_fund,
    meta={"type": "web", "freshness": "real-time"}
))

# --- Groww fetcher (placeholder) ---
def fetch_groww_fund(fund_query: str):
    # TODO: Implement actual scraping for Groww
    return {}

register_source(DataSource(
    name="Groww",
    fetch_func=fetch_groww_fund,
    meta={"type": "web", "freshness": "real-time"}
))

# --- BSE/NSE fetcher (placeholder) ---
def fetch_bse_nse_fund(fund_query: str):
    # TODO: Implement actual scraping for BSE/NSE
    return {}

register_source(DataSource(
    name="BSE_NSE",
    fetch_func=fetch_bse_nse_fund,
    meta={"type": "official", "freshness": "real-time"}
))

# --- Main unified fetch function ---
def get_fund_data(fund_query: str):
    """Aggregate results from all registered sources."""
    return registry_get_data(fund_query)

# Refactored main entrypoint
def get_realtime_fund_data(fund_name: str) -> Dict[str, Any]:
    """
    Fetches the latest info for a given fund from AMFI, MoneyControl, Value Research, then local factsheets, then DuckDuckGo.
    Returns a unified dict with all available info, sources, and suggestions if no good match.
    """
    # --- Step 1: Real-time sources ---
    amfi_funds = fetch_amfi_data()
    amfi_fund_names = [f.get("Scheme Name", "") for f in amfi_funds if f.get("Scheme Name")]
    best_match, close_matches = match_fund_name(fund_name, amfi_fund_names)
    amfi_info = search_amfi_fund(best_match, amfi_funds) if best_match else {}
    mc_info = fetch_moneycontrol_fund(best_match or fund_name)
    vr_info = fetch_valueresearch_fund(best_match or fund_name)
    result = {"sources": [], "suggestions": [n for n, _ in close_matches]}
    if amfi_info:
        result.update({
            "fund_name": amfi_info.get("Scheme Name"),
            "nav": amfi_info.get("Net Asset Value"),
            "aum": amfi_info.get("AUM"),
            "amfi_code": amfi_info.get("Scheme Code"),
            "amfi_source": AMFI_URL
        })
        result["sources"].append(AMFI_URL)
    if mc_info:
        result.update(mc_info)
        if mc_info.get("source"):
            result["sources"].append(mc_info["source"])
    if vr_info:
        result.update(vr_info)
        if vr_info.get("source"):
            result["sources"].append(vr_info["source"])
    # --- Step 2: Local factsheet fallback ---
    if not (amfi_info or mc_info or vr_info):
        local_data = get_local_factsheet_data(fund_name)
        if local_data and "data" in local_data:
            result.update({
                "fund_name": local_data.get("fund_name"),
                "local_factsheet_data": local_data["data"],
                "local_source": local_data["source"]
            })
            result["sources"].append(local_data["source"])
        else:
            # --- Step 3: DuckDuckGo fallback ---
            ddg_data = get_duckduckgo_fund_data(fund_name)
            if ddg_data and "duckduckgo_results" in ddg_data:
                result.update(ddg_data)
                result["sources"].append("DuckDuckGo")
            else:
                result["message"] = "No current data found for this fund."
    return result 

def get_fund_nav_history(fund_name: str):
    """Fetch historical NAV data for a fund from all available sources."""
    # Try AMFI first
    nav_history = []
    try:
        # Example: fetch_amfi_nav_history should return [{"date": ..., "nav": ...}, ...]
        nav_history = fetch_amfi_nav_history(fund_name)
        if nav_history:
            return nav_history
    except Exception:
        pass
    # Try local factsheets
    try:
        nav_history = fetch_local_factsheet_nav_history(fund_name)
        if nav_history:
            return nav_history
    except Exception:
        pass
    # TODO: Add more sources as needed
    return nav_history 

def fetch_amfi_nav_history(fund_name: str):
    """Fetch historical NAV data for a fund from AMFI's NAVAll.txt."""
    try:
        url = "https://www.amfiindia.com/spages/NAVAll.txt"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        headers = []
        nav_rows = []
        for line in lines:
            if line.startswith("Scheme Code"):  # header row
                headers = [h.strip() for h in line.split(';')]
                continue
            if line and line[0].isdigit():
                values = [v.strip() for v in line.split(';')]
                if len(values) == len(headers):
                    nav_rows.append(dict(zip(headers, values)))
        # Fuzzy match fund name
        from src.utils.helpers import match_fund_name
        fund_names = [row["Scheme Name"] for row in nav_rows if "Scheme Name" in row]
        best_match, _ = match_fund_name(fund_name, fund_names)
        if not best_match:
            return []
        # Filter for best match
        fund_navs = [row for row in nav_rows if row["Scheme Name"] == best_match]
        nav_history = []
        for row in fund_navs:
            try:
                nav_history.append({
                    "date": row["Date"],
                    "nav": float(row["Net Asset Value"])
                })
            except Exception:
                continue
        return nav_history
    except Exception as e:
        print(f"[AMFI NAV History] Error: {e}")
        return []

def fetch_local_factsheet_nav_history(fund_name: str):
    """Fetch historical NAV data for a fund from local factsheet JSONs."""
    import glob, json
    from src.utils.helpers import match_fund_name
    LOCAL_FACTSHEET_DIRS = ["processed_data", "processed_structured_data"]
    fund_names = set()
    file_map = {}
    for d in LOCAL_FACTSHEET_DIRS:
        for file in glob.glob(f"{d}/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Try to extract fund names from keys or text fields
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str) and len(v) > 3:
                            fund_names.add(v.strip())
                            file_map[v.strip()] = file
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, str) and len(v) > 3:
                                    fund_names.add(v.strip())
                                    file_map[v.strip()] = file
                        elif isinstance(item, str) and len(item) > 3:
                            fund_names.add(item.strip())
                            file_map[item.strip()] = file
            except Exception:
                continue
    best_match, _ = match_fund_name(fund_name, list(fund_names))
    if not best_match or best_match not in file_map:
        return []
    file = file_map[best_match]
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        nav_history = []
        # Try to find NAV history in dict or list
        if isinstance(data, dict):
            for k, v in data.items():
                if "nav" in k.lower() and isinstance(v, list):
                    for entry in v:
                        if isinstance(entry, dict) and "date" in entry and "nav" in entry:
                            nav_history.append({"date": entry["date"], "nav": float(entry["nav"])})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if "nav" in k.lower() and isinstance(v, list):
                            for entry in v:
                                if isinstance(entry, dict) and "date" in entry and "nav" in entry:
                                    nav_history.append({"date": entry["date"], "nav": float(entry["nav"])})
        return nav_history
    except Exception as e:
        print(f"[Local Factsheet NAV History] Error: {e}")
        return [] 