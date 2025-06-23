import asyncio
import aiohttp
import json
import re
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import time

class WebSearch:
    def __init__(self):
        self.search_engines = {
            'google': 'https://www.google.com/search?q=',
            'bing': 'https://www.bing.com/search?q='
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def search_mutual_funds(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for mutual fund information on the web"""
        try:
            # Enhance query with mutual fund specific terms
            enhanced_query = f"{query} mutual fund performance NAV latest factsheet"
            
            # Search multiple sources
            results = []
            
            # Search general web
            web_results = await self._search_web(enhanced_query, max_results)
            results.extend(web_results)
            
            # Search specific mutual fund sites
            fund_sites = [
                'moneycontrol.com',
                'valueresearchonline.com',
                'morningstar.in',
                'amfiindia.com',
                'sebi.gov.in'
            ]
            
            for site in fund_sites:
                site_query = f"{query} site:{site}"
                site_results = await self._search_web(site_query, 2)
                results.extend(site_results)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    async def _search_web(self, query: str, max_results: int) -> List[Dict]:
        """Search the web using multiple engines"""
        results = []
        
        for engine_name, base_url in self.search_engines.items():
            try:
                engine_results = await self._search_engine(base_url, query, max_results)
                results.extend(engine_results)
                await asyncio.sleep(1)  # Be respectful to search engines
            except Exception as e:
                print(f"Search engine {engine_name} failed: {e}")
        
        return results
    
    async def _search_engine(self, base_url: str, query: str, max_results: int) -> List[Dict]:
        """Search a specific search engine"""
        try:
            encoded_query = quote_plus(query)
            url = f"{base_url}{encoded_query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_search_results(html, max_results)
                    else:
                        print(f"Search request failed with status {response.status}")
                        return []
                        
        except Exception as e:
            print(f"Search engine request failed: {e}")
            return []
    
    def _parse_search_results(self, html: str, max_results: int) -> List[Dict]:
        """Parse search results from HTML"""
        results = []
        
        # Simple regex patterns to extract search results
        # This is a basic implementation - in production, you'd use proper HTML parsing
        
        # Look for title and snippet patterns
        title_pattern = r'<h3[^>]*>([^<]+)</h3>'
        snippet_pattern = r'<div[^>]*class="[^"]*snippet[^"]*"[^>]*>([^<]+)</div>'
        
        titles = re.findall(title_pattern, html, re.IGNORECASE)
        snippets = re.findall(snippet_pattern, html, re.IGNORECASE)
        
        for i in range(min(len(titles), len(snippets), max_results)):
            results.append({
                'title': titles[i].strip(),
                'snippet': snippets[i].strip(),
                'source': 'web_search'
            })
        
        return results
    
    async def get_fund_performance(self, fund_name: str) -> Optional[Dict]:
        """Get specific fund performance data"""
        try:
            query = f"{fund_name} NAV performance CAGR returns latest"
            results = await self.search_mutual_funds(query, 3)
            
            if results:
                # Extract performance data from results
                performance_data = {
                    'fund_name': fund_name,
                    'source': 'web_search',
                    'data': results,
                    'timestamp': time.time()
                }
                return performance_data
            
        except Exception as e:
            print(f"Error getting fund performance: {e}")
        
        return None
    
    async def get_market_overview(self) -> Optional[Dict]:
        """Get current market overview for mutual funds"""
        try:
            query = "Indian mutual fund market overview latest performance trends"
            results = await self.search_mutual_funds(query, 3)
            
            if results:
                return {
                    'type': 'market_overview',
                    'source': 'web_search',
                    'data': results,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            print(f"Error getting market overview: {e}")
        
        return None
    
    async def get_fund_comparison(self, fund1: str, fund2: str) -> Optional[Dict]:
        """Compare two mutual funds"""
        try:
            query = f"{fund1} vs {fund2} comparison performance returns"
            results = await self.search_mutual_funds(query, 5)
            
            if results:
                return {
                    'fund1': fund1,
                    'fund2': fund2,
                    'source': 'web_search',
                    'data': results,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            print(f"Error comparing funds: {e}")
        
        return None 