#!/usr/bin/env python3
"""
Daily Mutual Fund Data Updater
Fetches fresh data from multiple sources and updates PostgreSQL database
"""

import requests
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import logging
import json
import time
import os
from typing import Dict, List, Optional
import schedule
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyDataUpdater:
    """Daily updater for mutual fund data"""
    
    def __init__(self):
        self.db_params = {
            'host': 'localhost',
            'database': 'mutualfundpro',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        self.connection = None
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_params)
            logger.info("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect_db(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def fetch_amfi_nav_data(self) -> pd.DataFrame:
        """Fetch latest NAV data from AMFI"""
        try:
            # AMFI NAV All file URL (updated daily)
            amfi_url = "https://www.amfiindia.com/spages/NAVAll.txt"
            
            logger.info("📥 Fetching NAV data from AMFI...")
            response = requests.get(amfi_url, timeout=30)
            response.raise_for_status()
            
            # Parse AMFI NAV data
            lines = response.text.strip().split('\n')
            data = []
            
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(';')
                    if len(parts) >= 4:
                        # Try to parse NAV value, skip if invalid
                        try:
                            nav_value = float(parts[2].strip()) if parts[2].strip() else None
                            if nav_value is not None and nav_value > 0:
                                data.append({
                                    'scheme_code': parts[0].strip(),
                                    'scheme_name': parts[1].strip(),
                                    'nav': nav_value,
                                    'nav_date': parts[3].strip(),
                                    'amc_name': parts[4].strip() if len(parts) > 4 else None
                                })
                        except (ValueError, TypeError):
                            # Skip invalid NAV values
                            continue
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Fetched {len(df)} NAV records from AMFI")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching AMFI data: {e}")
            return pd.DataFrame()
    
    def fetch_moneycontrol_data(self, fund_name: str) -> Dict:
        """Fetch fund data from MoneyControl"""
        try:
            # Clean fund name for search
            search_name = fund_name.replace(' ', '+')
            search_url = f"https://www.moneycontrol.com/mutual-funds/search/?search={search_name}"
            
            # This would require proper web scraping with BeautifulSoup
            # For now, return empty dict as placeholder
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching MoneyControl data: {e}")
            return {}
    
    def fetch_valueresearch_data(self, fund_name: str) -> Dict:
        """Fetch fund data from Value Research"""
        try:
            # Clean fund name for search
            search_name = fund_name.replace(' ', '+')
            search_url = f"https://www.valueresearchonline.com/funds/search/?q={search_name}"
            
            # This would require proper web scraping with BeautifulSoup
            # For now, return empty dict as placeholder
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching Value Research data: {e}")
            return {}
    
    def update_nav_history(self, nav_data: pd.DataFrame):
        """Update NAV history in database"""
        if nav_data.empty:
            logger.warning("⚠️ No NAV data to update")
            return
        
        try:
            with self.connection.cursor() as cursor:
                # Get existing fund codes
                cursor.execute("SELECT fund_code FROM mf_factsheet")
                existing_funds = {row[0] for row in cursor.fetchall()}
                
                updated_count = 0
                for _, row in nav_data.iterrows():
                    # Try to match fund by name
                    fund_name = row['scheme_name']
                    nav_value = row['nav']
                    nav_date = datetime.strptime(row['nav_date'], '%d-%m-%Y').date()
                    
                    # Find matching fund code
                    cursor.execute("""
                        SELECT fund_code FROM mf_factsheet 
                        WHERE LOWER(fund_name) LIKE LOWER(%s)
                        LIMIT 1
                    """, (f'%{fund_name}%',))
                    
                    result = cursor.fetchone()
                    if result:
                        fund_code = result[0]
                        
                        # Insert/update NAV
                        cursor.execute("""
                            INSERT INTO mf_nav_history (fund_code, nav_date, nav_value)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (fund_code, nav_date) 
                            DO UPDATE SET nav_value = EXCLUDED.nav_value
                        """, (fund_code, nav_date, nav_value))
                        
                        updated_count += 1
                
                self.connection.commit()
                logger.info(f"✅ Updated NAV for {updated_count} funds")
                
        except Exception as e:
            logger.error(f"❌ Error updating NAV history: {e}")
            self.connection.rollback()
    
    def update_fund_returns(self):
        """Update fund returns data"""
        try:
            with self.connection.cursor() as cursor:
                # Get funds with NAV history
                cursor.execute("""
                    SELECT DISTINCT fund_code FROM mf_nav_history 
                    WHERE nav_date >= CURRENT_DATE - INTERVAL '1 year'
                """)
                funds = [row[0] for row in cursor.fetchall()]
                
                for fund_code in funds:
                    # Calculate returns from NAV history
                    cursor.execute("""
                        SELECT nav_date, nav_value 
                        FROM mf_nav_history 
                        WHERE fund_code = %s 
                        ORDER BY nav_date DESC
                    """, (fund_code,))
                    
                    nav_history = cursor.fetchall()
                    if len(nav_history) < 2:
                        continue
                    
                    # Calculate returns
                    latest_nav = nav_history[0][1]
                    returns = {}
                    
                    # Find NAVs for different periods
                    periods = {
                        '1m': 30, '3m': 90, '6m': 180, 
                        '1y': 365, '3y': 1095, '5y': 1825
                    }
                    
                    for period, days in periods.items():
                        target_date = datetime.now().date() - timedelta(days=days)
                        old_nav = None
                        
                        for nav_date, nav_value in nav_history:
                            if nav_date <= target_date:
                                old_nav = nav_value
                                break
                        
                        if old_nav and old_nav > 0:
                            returns[f'returns_{period}'] = ((latest_nav - old_nav) / old_nav) * 100
                    
                    # Update returns
                    if returns:
                        placeholders = ', '.join([f'{k} = %s' for k in returns.keys()])
                        query = f"""
                            INSERT INTO mf_returns (fund_code, date, {', '.join(returns.keys())})
                            VALUES (%s, %s, {', '.join(['%s'] * len(returns))})
                            ON CONFLICT (fund_code, date) 
                            DO UPDATE SET {placeholders}
                        """
                        
                        values = [fund_code, datetime.now().date()] + list(returns.values())
                        cursor.execute(query, values)
                
                self.connection.commit()
                logger.info(f"✅ Updated returns for {len(funds)} funds")
                
        except Exception as e:
            logger.error(f"❌ Error updating returns: {e}")
            self.connection.rollback()
    
    def update_fund_analytics(self):
        """Update fund analytics (Sharpe ratio, volatility, etc.)"""
        try:
            with self.connection.cursor() as cursor:
                # Get funds with sufficient NAV history
                cursor.execute("""
                    SELECT DISTINCT fund_code FROM mf_nav_history 
                    WHERE nav_date >= CURRENT_DATE - INTERVAL '2 years'
                """)
                funds = [row[0] for row in cursor.fetchall()]
                
                for fund_code in funds:
                    # Get NAV history for calculations
                    cursor.execute("""
                        SELECT nav_date, nav_value 
                        FROM mf_nav_history 
                        WHERE fund_code = %s 
                        ORDER BY nav_date
                    """, (fund_code,))
                    
                    nav_history = cursor.fetchall()
                    if len(nav_history) < 60:  # Need at least 60 days
                        continue
                    
                    # Calculate analytics
                    analytics = self.calculate_fund_analytics(nav_history)
                    
                    if analytics:
                        placeholders = ', '.join([f'{k} = %s' for k in analytics.keys()])
                        query = f"""
                            INSERT INTO mf_fund_analytics (fund_code, {', '.join(analytics.keys())})
                            VALUES (%s, {', '.join(['%s'] * len(analytics))})
                            ON CONFLICT (fund_code) 
                            DO UPDATE SET {placeholders}
                        """
                        
                        values = [fund_code] + list(analytics.values())
                        cursor.execute(query, values)
                
                self.connection.commit()
                logger.info(f"✅ Updated analytics for {len(funds)} funds")
                
        except Exception as e:
            logger.error(f"❌ Error updating analytics: {e}")
            self.connection.rollback()
    
    def calculate_fund_analytics(self, nav_history: List) -> Dict:
        """Calculate fund analytics from NAV history"""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            dates = [row[0] for row in nav_history]
            navs = [row[1] for row in nav_history]
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(navs)):
                if navs[i-1] > 0:
                    daily_return = (navs[i] - navs[i-1]) / navs[i-1]
                    daily_returns.append(daily_return)
            
            if len(daily_returns) < 30:
                return {}
            
            daily_returns = np.array(daily_returns)
            
            # Calculate analytics
            volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
            avg_return = np.mean(daily_returns) * 252 * 100  # Annualized
            
            # Sharpe ratio (assuming risk-free rate of 6%)
            risk_free_rate = 0.06
            sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Beta (simplified - would need benchmark data)
            beta = 1.0  # Placeholder
            
            # Alpha
            alpha = avg_return - (risk_free_rate + beta * (avg_return - risk_free_rate))
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            
            return {
                'sharpe_ratio': round(sharpe_ratio, 4),
                'volatility': round(volatility, 4),
                'beta': round(beta, 4),
                'alpha': round(alpha, 4),
                'max_drawdown': round(max_drawdown, 4),
                'tracking_error': 0.0,  # Would need benchmark data
                'information_ratio': 0.0,  # Would need benchmark data
                'sortino_ratio': 0.0,  # Would need downside deviation
                'calmar_ratio': 0.0  # Would need average annual return
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating analytics: {e}")
            return {}
    
    def run_daily_update(self):
        """Run complete daily update"""
        logger.info("🚀 Starting daily mutual fund data update...")
        
        if not self.connect_db():
            return
        
        try:
            # 1. Fetch latest NAV data from AMFI
            nav_data = self.fetch_amfi_nav_data()
            if not nav_data.empty:
                self.update_nav_history(nav_data)
            
            # 2. Update fund returns
            self.update_fund_returns()
            
            # 3. Update fund analytics
            self.update_fund_analytics()
            
            logger.info("✅ Daily update completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Daily update failed: {e}")
        finally:
            self.disconnect_db()

def setup_cron_job():
    """Setup cron job for daily updates"""
    # This would typically be done via system cron
    # For development, we can use Python schedule library
    
    updater = DailyDataUpdater()
    
    # Schedule daily update at 6 PM
    schedule.every().day.at("18:00").do(updater.run_daily_update)
    
    logger.info("⏰ Scheduled daily update for 6:00 PM")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def run_manual_update():
    """Run manual update (for testing)"""
    updater = DailyDataUpdater()
    updater.run_daily_update()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Mutual Fund Data Updater')
    parser.add_argument('--manual', action='store_true', help='Run manual update')
    parser.add_argument('--schedule', action='store_true', help='Start scheduled updates')
    
    args = parser.parse_args()
    
    if args.manual:
        run_manual_update()
    elif args.schedule:
        setup_cron_job()
    else:
        print("Usage: python daily_updater.py --manual or --schedule") 