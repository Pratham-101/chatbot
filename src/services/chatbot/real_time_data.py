# Stub file for real_time_data.py to allow backend startup

class RealTimeProvider:
    """Stub real-time data provider."""
    
    async def get_live_nav(self, fund_name: str):
        """Get live NAV for a fund."""
        return {
            "fund_name": fund_name,
            "nav": 100.0,
            "date": "2024-07-07",
            "source": "stub"
        }
    
    async def get_fund_performance(self, fund_name: str):
        """Get fund performance data."""
        return {
            "fund_name": fund_name,
            "1y_return": 10.0,
            "3y_return": 30.0,
            "aum": 1000.0,
            "source": "stub"
        }
    
    async def get_fund_news(self, fund_name: str):
        """Get fund news."""
        return []
    
    async def get_market_indices(self):
        """Get market indices."""
        return {
            "Nifty": {"value": 20000, "change_percent": "+1%"},
            "Sensex": {"value": 65000, "change_percent": "+0.8%"},
            "last_updated": "2024-07-07"
        }
    
    async def get_sector_performance(self):
        """Get sector performance."""
        return [
            {"sector": "IT", "performance": "+2%"},
            {"sector": "Banking", "performance": "+1.5%"}
        ]
    
    async def get_economic_indicators(self):
        """Get economic indicators."""
        return {
            "gdp": "7%",
            "inflation": "5%",
            "last_updated": "2024-07-07"
        }
    
    async def get_regulatory_updates(self):
        """Get regulatory updates."""
        return [
            {
                "title": "SEBI regulatory update",
                "summary": "Stub regulatory content",
                "link": "https://example.com",
                "timestamp": "2024-07-07T00:00:00"
            }
        ]
    
    def analyze_sentiment(self, headlines):
        """Analyze sentiment of headlines."""
        return "neutral"

class MarketDataProvider:
    """Stub market data provider."""
    
    async def get_market_overview(self):
        """Get market overview."""
        return {
            "sensex": 75000,
            "nifty": 22500,
            "market_sentiment": "positive",
            "source": "stub"
        }
    
    async def get_regulatory_updates(self):
        """Get regulatory updates."""
        return {
            "updates": [
                {
                    "title": "SEBI updates mutual fund regulations",
                    "summary": "New guidelines for mutual fund investments",
                    "date": "2024-07-07",
                    "impact": "positive",
                    "source": "stub"
                }
            ],
            "source": "stub"
        }
    
    async def get_economic_indicators(self):
        """Get economic indicators."""
        return {
            "indicators": [
                {
                    "name": "GDP Growth",
                    "value": "7.2%",
                    "trend": "positive",
                    "date": "2024-07-07"
                },
                {
                    "name": "Inflation Rate",
                    "value": "4.5%",
                    "trend": "stable",
                    "date": "2024-07-07"
                },
                {
                    "name": "Repo Rate",
                    "value": "6.5%",
                    "trend": "stable",
                    "date": "2024-07-07"
                }
            ],
            "source": "stub"
        }
    
    async def get_sector_performance(self):
        """Get sector performance."""
        return {
            "sectors": [
                {"name": "Technology", "return": 15.0},
                {"name": "Healthcare", "return": 12.0},
                {"name": "Finance", "return": 8.0}
            ],
            "source": "stub"
        }
    
    async def get_market_data(self, query: str):
        """Get market data."""
        return {
            "query": query,
            "data": "Stub market data",
            "source": "stub"
        }

# Create instances
real_time_provider = RealTimeProvider()
market_data_provider = MarketDataProvider() 