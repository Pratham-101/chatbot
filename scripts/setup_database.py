#!/usr/bin/env python3
"""
Setup script for PostgreSQL mutual fund database
Creates tables and populates with sample data
"""

import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import random
import json
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.data.db_access import MutualFundDB

def create_tables(db):
    """Create all mutual fund tables"""
    
    # Create mf_factsheet table
    factsheet_sql = """
    CREATE TABLE IF NOT EXISTS mf_factsheet (
        fund_code VARCHAR(20) PRIMARY KEY,
        fund_name VARCHAR(255) NOT NULL,
        amc_name VARCHAR(100),
        category VARCHAR(50),
        sub_category VARCHAR(50),
        fund_type VARCHAR(20),
        launch_date DATE,
        aum DECIMAL(15,2),
        expense_ratio DECIMAL(5,4),
        min_investment DECIMAL(10,2),
        fund_manager VARCHAR(100),
        benchmark VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create mf_nav_history table
    nav_sql = """
    CREATE TABLE IF NOT EXISTS mf_nav_history (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        nav_date DATE NOT NULL,
        nav_value DECIMAL(10,4) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(fund_code, nav_date)
    );
    """
    
    # Create mf_fund_ratings table
    ratings_sql = """
    CREATE TABLE IF NOT EXISTS mf_fund_ratings (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        rating_agency VARCHAR(50),
        rating VARCHAR(10),
        rating_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create mf_returns table
    returns_sql = """
    CREATE TABLE IF NOT EXISTS mf_returns (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        date DATE NOT NULL,
        returns_1m DECIMAL(8,4),
        returns_3m DECIMAL(8,4),
        returns_6m DECIMAL(8,4),
        returns_1y DECIMAL(8,4),
        returns_3y DECIMAL(8,4),
        returns_5y DECIMAL(8,4),
        returns_since_inception DECIMAL(8,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create mf_portfolio_holdings table
    holdings_sql = """
    CREATE TABLE IF NOT EXISTS mf_portfolio_holdings (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        security_name VARCHAR(255),
        security_type VARCHAR(50),
        weight DECIMAL(5,2),
        market_value DECIMAL(15,2),
        quantity DECIMAL(15,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create mf_fund_analytics table
    analytics_sql = """
    CREATE TABLE IF NOT EXISTS mf_fund_analytics (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        sharpe_ratio DECIMAL(8,4),
        volatility DECIMAL(8,4),
        beta DECIMAL(8,4),
        alpha DECIMAL(8,4),
        max_drawdown DECIMAL(8,4),
        tracking_error DECIMAL(8,4),
        information_ratio DECIMAL(8,4),
        sortino_ratio DECIMAL(8,4),
        calmar_ratio DECIMAL(8,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create mf_fund_holdings table (alternative name)
    fund_holdings_sql = """
    CREATE TABLE IF NOT EXISTS mf_fund_holdings (
        id SERIAL PRIMARY KEY,
        fund_code VARCHAR(20) REFERENCES mf_factsheet(fund_code),
        holding_name VARCHAR(255),
        holding_type VARCHAR(50),
        weight DECIMAL(5,2),
        market_value DECIMAL(15,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    tables_sql = [
        factsheet_sql,
        nav_sql,
        ratings_sql,
        returns_sql,
        holdings_sql,
        analytics_sql,
        fund_holdings_sql
    ]
    
    for sql in tables_sql:
        try:
            with db.connection.cursor() as cursor:
                cursor.execute(sql)
            db.connection.commit()
            print(f"✅ Table created successfully")
        except Exception as e:
            print(f"❌ Error creating table: {e}")

def populate_sample_data(db):
    """Populate tables with sample mutual fund data"""
    
    # Sample fund data
    sample_funds = [
        {
            'fund_code': 'HDFC001',
            'fund_name': 'HDFC Flexi Cap Fund - Growth Option',
            'amc_name': 'HDFC Mutual Fund',
            'category': 'Equity',
            'sub_category': 'Flexi Cap',
            'fund_type': 'Growth',
            'launch_date': '2018-01-15',
            'aum': 15000.50,
            'expense_ratio': 0.0150,
            'min_investment': 5000.00,
            'fund_manager': 'Prashant Jain',
            'benchmark': 'NIFTY 500'
        },
        {
            'fund_code': 'AXIS001',
            'fund_name': 'Axis Bluechip Fund - Growth Option',
            'amc_name': 'Axis Mutual Fund',
            'category': 'Equity',
            'sub_category': 'Large Cap',
            'fund_type': 'Growth',
            'launch_date': '2017-06-01',
            'aum': 8500.25,
            'expense_ratio': 0.0175,
            'min_investment': 5000.00,
            'fund_manager': 'Shreyash Devalkar',
            'benchmark': 'NIFTY 100'
        },
        {
            'fund_code': 'MIRAE001',
            'fund_name': 'Mirae Asset Large Cap Fund - Growth',
            'amc_name': 'Mirae Asset Mutual Fund',
            'category': 'Equity',
            'sub_category': 'Large Cap',
            'fund_type': 'Growth',
            'launch_date': '2016-04-15',
            'aum': 12000.75,
            'expense_ratio': 0.0160,
            'min_investment': 5000.00,
            'fund_manager': 'Neelesh Surana',
            'benchmark': 'NIFTY 100'
        },
        {
            'fund_code': 'SBI001',
            'fund_name': 'SBI Bluechip Fund - Growth Option',
            'amc_name': 'SBI Mutual Fund',
            'category': 'Equity',
            'sub_category': 'Large Cap',
            'fund_type': 'Growth',
            'launch_date': '2015-03-10',
            'aum': 9500.30,
            'expense_ratio': 0.0180,
            'min_investment': 5000.00,
            'fund_manager': 'Sohini Andani',
            'benchmark': 'NIFTY 100'
        },
        {
            'fund_code': 'ICICI001',
            'fund_name': 'ICICI Prudential Bluechip Fund - Growth',
            'amc_name': 'ICICI Prudential Mutual Fund',
            'category': 'Equity',
            'sub_category': 'Large Cap',
            'fund_type': 'Growth',
            'launch_date': '2016-08-20',
            'aum': 11000.45,
            'expense_ratio': 0.0165,
            'min_investment': 5000.00,
            'fund_manager': 'Sankaran Naren',
            'benchmark': 'NIFTY 100'
        }
    ]
    
    # Insert fund factsheet data
    for fund in sample_funds:
        query = """
        INSERT INTO mf_factsheet 
        (fund_code, fund_name, amc_name, category, sub_category, fund_type, 
         launch_date, aum, expense_ratio, min_investment, fund_manager, benchmark)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (fund_code) DO NOTHING;
        """
        
        values = (
            fund['fund_code'], fund['fund_name'], fund['amc_name'],
            fund['category'], fund['sub_category'], fund['fund_type'],
            fund['launch_date'], fund['aum'], fund['expense_ratio'],
            fund['min_investment'], fund['fund_manager'], fund['benchmark']
        )
        
        try:
            with db.connection.cursor() as cursor:
                cursor.execute(query, values)
            db.connection.commit()
            print(f"✅ Inserted fund: {fund['fund_name']}")
        except Exception as e:
            print(f"❌ Error inserting fund {fund['fund_name']}: {e}")
    
    # Generate NAV history for each fund
    for fund in sample_funds:
        fund_code = fund['fund_code']
        base_nav = random.uniform(15.0, 25.0)
        
        # Generate 365 days of NAV data
        for i in range(365):
            date = datetime.now() - timedelta(days=i)
            # Simulate NAV movement with some randomness
            nav_change = random.uniform(-0.02, 0.02)  # -2% to +2% daily change
            nav_value = base_nav * (1 + nav_change)
            base_nav = nav_value
            
            query = """
            INSERT INTO mf_nav_history (fund_code, nav_date, nav_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (fund_code, nav_date) DO NOTHING;
            """
            
            try:
                with db.connection.cursor() as cursor:
                    cursor.execute(query, (fund_code, date.date(), round(nav_value, 4)))
                db.connection.commit()
            except Exception as e:
                print(f"❌ Error inserting NAV for {fund_code}: {e}")
        
        print(f"✅ Generated NAV history for {fund['fund_name']}")
    
    # Generate fund ratings
    rating_agencies = ['CRISIL', 'ICRA', 'Value Research', 'Morningstar']
    ratings = ['AAA', 'AA+', 'AA', 'A+', 'A']
    
    for fund in sample_funds:
        fund_code = fund['fund_code']
        
        for agency in rating_agencies:
            rating = random.choice(ratings)
            rating_date = datetime.now() - timedelta(days=random.randint(30, 180))
            
            query = """
            INSERT INTO mf_fund_ratings (fund_code, rating_agency, rating, rating_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """
            
            try:
                with db.connection.cursor() as cursor:
                    cursor.execute(query, (fund_code, agency, rating, rating_date.date()))
                db.connection.commit()
            except Exception as e:
                print(f"❌ Error inserting rating for {fund_code}: {e}")
        
        print(f"✅ Generated ratings for {fund['fund_name']}")
    
    # Generate returns data
    for fund in sample_funds:
        fund_code = fund['fund_code']
        
        query = """
        INSERT INTO mf_returns 
        (fund_code, date, returns_1m, returns_3m, returns_6m, returns_1y, returns_3y, returns_5y, returns_since_inception)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        
        values = (
            fund_code,
            datetime.now().date(),
            random.uniform(-5.0, 8.0),  # 1 month returns
            random.uniform(-8.0, 15.0),  # 3 month returns
            random.uniform(-10.0, 20.0),  # 6 month returns
            random.uniform(-15.0, 35.0),  # 1 year returns
            random.uniform(8.0, 25.0),   # 3 year returns
            random.uniform(12.0, 30.0),  # 5 year returns
            random.uniform(15.0, 40.0)   # Since inception
        )
        
        try:
            with db.connection.cursor() as cursor:
                cursor.execute(query, values)
            db.connection.commit()
            print(f"✅ Generated returns for {fund['fund_name']}")
        except Exception as e:
            print(f"❌ Error inserting returns for {fund_code}: {e}")
    
    # Generate analytics data
    for fund in sample_funds:
        fund_code = fund['fund_code']
        
        query = """
        INSERT INTO mf_fund_analytics 
        (fund_code, sharpe_ratio, volatility, beta, alpha, max_drawdown, tracking_error, information_ratio, sortino_ratio, calmar_ratio)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        
        values = (
            fund_code,
            random.uniform(0.5, 2.0),    # Sharpe ratio
            random.uniform(12.0, 25.0),   # Volatility
            random.uniform(0.8, 1.2),     # Beta
            random.uniform(-2.0, 5.0),    # Alpha
            random.uniform(-25.0, -8.0),  # Max drawdown
            random.uniform(2.0, 8.0),     # Tracking error
            random.uniform(0.1, 1.5),     # Information ratio
            random.uniform(0.8, 2.5),     # Sortino ratio
            random.uniform(0.3, 1.8)      # Calmar ratio
        )
        
        try:
            with db.connection.cursor() as cursor:
                cursor.execute(query, values)
            db.connection.commit()
            print(f"✅ Generated analytics for {fund['fund_name']}")
        except Exception as e:
            print(f"❌ Error inserting analytics for {fund_code}: {e}")
    
    # Generate portfolio holdings
    sample_holdings = [
        'Reliance Industries Ltd', 'TCS Ltd', 'HDFC Bank Ltd', 'Infosys Ltd',
        'ICICI Bank Ltd', 'Hindustan Unilever Ltd', 'ITC Ltd', 'Bharti Airtel Ltd',
        'Kotak Mahindra Bank Ltd', 'Axis Bank Ltd', 'Larsen & Toubro Ltd',
        'Asian Paints Ltd', 'Maruti Suzuki India Ltd', 'HCL Technologies Ltd'
    ]
    
    for fund in sample_funds:
        fund_code = fund['fund_code']
        
        # Generate 8-12 holdings per fund
        num_holdings = random.randint(8, 12)
        selected_holdings = random.sample(sample_holdings, num_holdings)
        
        total_weight = 0
        for i, holding in enumerate(selected_holdings):
            if i == len(selected_holdings) - 1:
                weight = 100 - total_weight  # Last holding gets remaining weight
            else:
                weight = random.uniform(3.0, 15.0)
                total_weight += weight
            
            market_value = random.uniform(100.0, 500.0)
            quantity = random.uniform(1000.0, 10000.0)
            
            query = """
            INSERT INTO mf_portfolio_holdings 
            (fund_code, security_name, security_type, weight, market_value, quantity)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """
            
            values = (
                fund_code,
                holding,
                'Equity',
                round(weight, 2),
                round(market_value, 2),
                round(quantity, 2)
            )
            
            try:
                with db.connection.cursor() as cursor:
                    cursor.execute(query, values)
                db.connection.commit()
            except Exception as e:
                print(f"❌ Error inserting holding for {fund_code}: {e}")
        
        print(f"✅ Generated portfolio holdings for {fund['fund_name']}")

def main():
    """Main setup function"""
    print("🚀 Setting up PostgreSQL mutual fund database...")
    
    # Initialize database connection
    db = MutualFundDB()
    if not db.connect():
        print("❌ Failed to connect to database")
        return
    
    try:
        # Create tables
        print("\n📋 Creating database tables...")
        create_tables(db)
        
        # Populate with sample data
        print("\n📊 Populating with sample data...")
        populate_sample_data(db)
        
        print("\n✅ Database setup completed successfully!")
        
        # Test the setup
        print("\n🧪 Testing database access...")
        tables = db.get_all_tables()
        print(f"Found {len(tables)} tables: {', '.join(tables)}")
        
        # Test fund search
        funds = db.search_fund_by_name("HDFC")
        print(f"\nFound {len(funds)} HDFC funds:")
        for fund in funds:
            print(f"  - {fund['fund_name']} ({fund['fund_code']})")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    main() 