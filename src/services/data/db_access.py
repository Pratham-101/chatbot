import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MutualFundDB:
    """Database access layer for mutual fund data"""
    
    def __init__(self):
        # Database connection parameters
        self.db_params = {
            'host': '34.57.196.130',
            'database': 'mutualfundpro',
            'user': 'postgres',
            'password': 'mutual@fund@pro12',
            'port': 5432
        }
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_params)
            logger.info("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results as list of dictionaries"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Rollback on error
            self.connection.rollback()
            return []
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        results = self.execute_query(query)
        return [row['table_name'] for row in results]
    
    def search_fund_by_name(self, fund_name: str, limit: int = 5) -> List[Dict]:
        """Search for funds by name using fuzzy matching"""
        # Clean and normalize fund name
        clean_name = re.sub(r'[^\w\s]', '', fund_name.lower())
        
        query = """
        SELECT DISTINCT 
            f.scheme_name as fund_name,
            f.isin as fund_code,
            f.amc as amc_name,
            f.sub_category as category,
            f.sub_category,
            f.scheme_type as fund_type
        FROM mf_factsheet f
        WHERE LOWER(f.scheme_name) LIKE %s
        OR LOWER(f.scheme_name) LIKE %s
        LIMIT %s;
        """
        
        # Try exact match first
        results = self.execute_query(query, (f'%{clean_name}%', f'%{fund_name}%', limit))
        
        if not results:
            # Try broader search
            words = clean_name.split()
            if len(words) > 1:
                query = """
                SELECT DISTINCT 
                    f.scheme_name as fund_name,
                    f.isin as fund_code,
                    f.amc as amc_name,
                    f.sub_category as category,
                    f.sub_category,
                    f.scheme_type as fund_type
                FROM mf_factsheet f
                WHERE """
                conditions = []
                params = []
                for word in words[:3]:  # Use first 3 words
                    conditions.append("LOWER(f.scheme_name) LIKE %s")
                    params.append(f'%{word}%')
                
                query += " OR ".join(conditions) + " LIMIT %s;"
                params.append(limit)
                results = self.execute_query(query, tuple(params))
        
        return results
    
    def get_fund_details(self, fund_code: str = None, fund_name: str = None) -> Dict:
        """Get comprehensive fund details"""
        if not fund_code and not fund_name:
            return {}
        
        if fund_name and not fund_code:
            # Find fund code by name
            funds = self.search_fund_by_name(fund_name, 1)
            if funds:
                fund_code = funds[0]['fund_code']
            else:
                return {}
        
        # Get basic fund info
        query = """
        SELECT * FROM mf_factsheet 
        WHERE isin = %s;
        """
        factsheet = self.execute_query(query, (fund_code,))
        
        if not factsheet:
            return {}
        
        fund_info = factsheet[0]
        
        # Get latest NAV
        nav_query = """
        SELECT date as nav_date, nav as nav_value 
        FROM mf_nav_history 
        WHERE isin = %s 
        ORDER BY date DESC 
        LIMIT 1;
        """
        nav_data = self.execute_query(nav_query, (fund_code,))
        
        # Get fund ratings (if table exists and has data)
        rating_query = """
        SELECT * FROM mf_fund_ratings 
        WHERE isin = %s;
        """
        ratings = self.execute_query(rating_query, (fund_code,))
        
        # Get latest returns
        returns_query = """
        SELECT * FROM mf_returns 
        WHERE isin = %s 
        ORDER BY last_updated DESC 
        LIMIT 1;
        """
        returns = self.execute_query(returns_query, (fund_code,))
        
        # Get analytics (if table exists and has data)
        analytics_query = """
        SELECT * FROM mf_fund_analytics 
        WHERE isin = %s;
        """
        analytics = self.execute_query(analytics_query, (fund_code,))
        
        # Compile results
        result = {
            'fund_info': fund_info,
            'latest_nav': nav_data[0] if nav_data else None,
            'ratings': ratings,
            'returns': returns[0] if returns else None,
            'analytics': analytics[0] if analytics else None,
            'source': 'PostgreSQL Database'
        }
        
        return result
    
    def get_nav_history(self, fund_code: str, days: int = 365) -> List[Dict]:
        """Get NAV history for a fund"""
        query = """
        SELECT nav_date, nav_value 
        FROM mf_nav_history 
        WHERE fund_code = %s 
        AND nav_date >= %s
        ORDER BY nav_date DESC;
        """
        
        start_date = datetime.now() - timedelta(days=days)
        return self.execute_query(query, (fund_code, start_date))
    
    def get_fund_holdings(self, fund_code: str) -> List[Dict]:
        """Get portfolio holdings for a fund"""
        query = """
        SELECT * FROM mf_portfolio_holdings 
        WHERE fund_code = %s 
        ORDER BY weight DESC;
        """
        return self.execute_query(query, (fund_code,))
    
    def get_fund_comparison(self, fund_codes: List[str]) -> Dict:
        """Compare multiple funds"""
        if len(fund_codes) < 2:
            return {}
        
        # Get basic info for all funds
        placeholders = ','.join(['%s'] * len(fund_codes))
        query = f"""
        SELECT * FROM mf_factsheet 
        WHERE fund_code IN ({placeholders});
        """
        funds_info = self.execute_query(query, tuple(fund_codes))
        
        # Get latest NAV for all funds
        nav_query = f"""
        SELECT DISTINCT ON (fund_code) 
            fund_code, nav_date, nav_value 
        FROM mf_nav_history 
        WHERE fund_code IN ({placeholders})
        ORDER BY fund_code, nav_date DESC;
        """
        navs = self.execute_query(nav_query, tuple(fund_codes))
        
        # Get returns for all funds
        returns_query = f"""
        SELECT DISTINCT ON (fund_code) 
            fund_code, * 
        FROM mf_returns 
        WHERE fund_code IN ({placeholders})
        ORDER BY fund_code, date DESC;
        """
        returns = self.execute_query(returns_query, tuple(fund_codes))
        
        # Get analytics for all funds
        analytics_query = f"""
        SELECT * FROM mf_fund_analytics 
        WHERE fund_code IN ({placeholders});
        """
        analytics = self.execute_query(analytics_query, tuple(fund_codes))
        
        return {
            'funds_info': funds_info,
            'navs': navs,
            'returns': returns,
            'analytics': analytics,
            'source': 'PostgreSQL Database'
        }
    
    def get_top_funds(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Get top performing funds by category"""
        if category:
            query = """
            SELECT f.*, a.*, r.*
            FROM mf_factsheet f
            LEFT JOIN mf_fund_analytics a ON f.fund_code = a.fund_code
            LEFT JOIN mf_returns r ON f.fund_code = r.fund_code
            WHERE f.category = %s
            ORDER BY a.sharpe_ratio DESC NULLS LAST
            LIMIT %s;
            """
            return self.execute_query(query, (category, limit))
        else:
            query = """
            SELECT f.*, a.*, r.*
            FROM mf_factsheet f
            LEFT JOIN mf_fund_analytics a ON f.fund_code = a.fund_code
            LEFT JOIN mf_returns r ON f.fund_code = r.fund_code
            ORDER BY a.sharpe_ratio DESC NULLS LAST
            LIMIT %s;
            """
            return self.execute_query(query, (limit,))
    
    def get_categories(self) -> List[str]:
        """Get all fund categories"""
        query = """
        SELECT DISTINCT category 
        FROM mf_factsheet 
        WHERE category IS NOT NULL 
        ORDER BY category;
        """
        results = self.execute_query(query)
        return [row['category'] for row in results]
    
    def get_amcs(self) -> List[str]:
        """Get all AMC names"""
        query = """
        SELECT DISTINCT amc_name 
        FROM mf_factsheet 
        WHERE amc_name IS NOT NULL 
        ORDER BY amc_name;
        """
        results = self.execute_query(query)
        return [row['amc_name'] for row in results]

    def get_all_fund_names(self) -> list:
        """Return a list of all fund names in the database."""
        query = "SELECT scheme_name FROM mf_factsheet;"
        results = self.execute_query(query)
        return [row['scheme_name'] for row in results]

    def fuzzy_search_fund_by_name(self, user_query: str, limit: int = 3, score_cutoff: int = 70) -> list:
        """Fuzzy search for fund names using rapidfuzz. Returns a list of (fund_name, score) tuples."""
        all_funds = self.get_all_fund_names()
        matches = process.extract(user_query, all_funds, scorer=fuzz.token_sort_ratio, limit=limit)
        # Only return matches above the score_cutoff
        return [(name, score) for name, score, _ in matches if score >= score_cutoff]

# Global database instance
db_instance = None

def get_db_instance():
    """Get or create database instance"""
    global db_instance
    if db_instance is None:
        db_instance = MutualFundDB()
        if not db_instance.connect():
            logger.error("Failed to initialize database connection")
            return None
    return db_instance

def test_connection():
    """Test database connection and list tables"""
    db = get_db_instance()
    if not db:
        return False
    
    try:
        tables = db.get_all_tables()
        logger.info(f"✅ Database connected. Found {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table}")
        return True
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the database connection
    if test_connection():
        print("Database connection successful!")
        
        # Test fund search
        db = get_db_instance()
        funds = db.search_fund_by_name("HDFC")
        print(f"\nFound {len(funds)} HDFC funds:")
        for fund in funds[:3]:
            print(f"  - {fund['fund_name']} ({fund['fund_code']})")
    else:
        print("Database connection failed!") 