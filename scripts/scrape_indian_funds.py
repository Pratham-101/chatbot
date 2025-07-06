import os
import json
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import re
import requests
import csv

OUTPUT_JSON = "data/all_indian_funds_cleaned.json"

# --- Helper functions ---
def setup_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Firefox(options=options)
    driver.set_page_load_timeout(30)
    return driver

def normalize_fund_name(name):
    if not name:
        return ""
    name = re.sub(r"Sponsored Adv.*?Invest Now", "", name, flags=re.IGNORECASE)
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def clean_fund_record(fund):
    cleaned = {}
    for k, v in fund.items():
        key = k.strip().replace(" ", "_").lower()
        cleaned[key] = v.strip() if isinstance(v, str) else v
    if "fund_name" in cleaned:
        cleaned["fund_name"] = normalize_fund_name(cleaned["fund_name"])
    elif "scheme_name" in cleaned:
        cleaned["fund_name"] = normalize_fund_name(cleaned["scheme_name"])
    if not cleaned.get("fund_name") or len(cleaned["fund_name"]) < 3:
        return None
    return cleaned

def remove_duplicates(funds):
    seen = set()
    unique = []
    for fund in funds:
        key = fund["fund_name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(fund)
    return unique

# --- AMFI India Scraper (CSV) ---
def scrape_amfi(driver=None):
    print("Scraping AMFI India (CSV)...")
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    funds = []
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        lines = response.text.splitlines()
        reader = csv.reader(lines, delimiter=';')
        headers = next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            scheme_code, isin_div_payout, isin_div_reinvest, scheme_name, nav, repurchase_price, sale_price, date = (row + [None]*8)[:8]
            if scheme_name and nav:
                funds.append({
                    "fund_name": scheme_name.strip(),
                    "nav": nav.strip(),
                    "date": date.strip() if date else None,
                    "source": url,
                    "platform": "AMFI",
                    "scraped_at": datetime.now().isoformat()
                })
        print(f"  Found {len(funds)} funds in AMFI CSV.")
    except Exception as e:
        print(f"  Error downloading/parsing AMFI CSV: {e}")
    return funds

# --- Groww Scraper (Enhanced) ---
def scrape_groww(driver):
    print("Scraping Groww (Enhanced)...")
    url = "https://groww.in/mutual-funds/explore"
    driver.get(url)
    time.sleep(5)
    
    funds = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    # Scroll to load all funds (infinite scroll)
    scroll_attempts = 0
    max_scrolls = 20
    
    while scroll_attempts < max_scrolls:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1
        print(f"  Scroll {scroll_attempts}: Loaded more funds...")
    
    # Now extract all fund data
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Look for fund cards/containers
    fund_containers = soup.find_all(["div", "article"], class_=re.compile(r"fund|card|item"))
    
    for container in fund_containers:
        try:
            # Extract fund name
            name_elem = container.find(["h1", "h2", "h3", "h4", "h5", "h6", "a", "span", "div"], 
                                     class_=re.compile(r"name|title|fund-name"))
            if not name_elem:
                # Try to find any text that looks like a fund name
                text_content = container.get_text(strip=True)
                if len(text_content) > 10 and len(text_content) < 200:
                    # Check if it contains fund-like keywords
                    if any(keyword in text_content.lower() for keyword in 
                          ["fund", "direct", "growth", "dividend", "equity", "debt", "hybrid"]):
                        fund_name = text_content.split('\n')[0][:100]  # Take first line
                    else:
                        continue
                else:
                    continue
            else:
                fund_name = name_elem.get_text(strip=True)
            
            if not fund_name or len(fund_name) < 3:
                continue
                
            fund_data = {
                "fund_name": fund_name,
                "source": url,
                "platform": "Groww",
                "scraped_at": datetime.now().isoformat()
            }
            
            # Try to extract additional details
            # Category
            category_elem = container.find(["span", "div"], class_=re.compile(r"category|type"))
            if category_elem:
                fund_data["category"] = category_elem.get_text(strip=True)
            
            # Returns
            returns_elem = container.find(["span", "div"], class_=re.compile(r"return|performance"))
            if returns_elem:
                fund_data["returns"] = returns_elem.get_text(strip=True)
            
            # AUM
            aum_elem = container.find(["span", "div"], class_=re.compile(r"aum|assets"))
            if aum_elem:
                fund_data["aum"] = aum_elem.get_text(strip=True)
            
            # Rating
            rating_elem = container.find(["span", "div"], class_=re.compile(r"rating|star"))
            if rating_elem:
                fund_data["rating"] = rating_elem.get_text(strip=True)
            
            funds.append(fund_data)
            
        except Exception as e:
            continue
    
    # If we didn't find structured data, try the original approach
    if len(funds) < 10:
        print("  Falling back to link-based extraction...")
        for a in soup.find_all("a", href=re.compile(r"^/mutual-funds/")):
            name = a.get_text(strip=True)
            if name and len(name) > 3:
                funds.append({
                    "fund_name": name,
                    "source": url,
                    "platform": "Groww",
                    "scraped_at": datetime.now().isoformat()
                })
    
    print(f"  Found {len(funds)} funds on Groww.")
    return funds

# --- Morningstar India Scraper (stub) ---
def scrape_morningstar(driver):
    print("Scraping Morningstar India (stub)...")
    # TODO: Implement
    return []

# --- ET Money Scraper (stub) ---
def scrape_etmoney(driver):
    print("Scraping ET Money (stub)...")
    # TODO: Implement
    return []

# --- Value Research Online Scraper (Implemented) ---
def scrape_valueresearch(driver):
    print("Scraping Value Research Online...")
    funds = []
    
    # Value Research fund categories to scrape
    categories = [
        "equity",
        "debt", 
        "hybrid",
        "solution-oriented",
        "international"
    ]
    
    for category in categories:
        try:
            url = f"https://www.valueresearchonline.com/funds/fundSelector/?category={category}&plan=direct&tab=returns"
            print(f"  Scraping {category} funds...")
            driver.get(url)
            time.sleep(5)
            
            # Wait for table to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
                )
            except TimeoutException:
                print(f"    No table found for {category}")
                continue
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Find fund tables
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue
                
                # Extract headers
                headers = []
                header_row = rows[0]
                for th in header_row.find_all(["th", "td"]):
                    headers.append(th.get_text(strip=True))
                
                # Process data rows
                for row in rows[1:]:
                    cols = row.find_all(["td", "th"])
                    if len(cols) >= 3:
                        fund_data = {
                            "source": url,
                            "platform": "ValueResearch",
                            "category": category,
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        for i, col in enumerate(cols):
                            if i < len(headers):
                                key = headers[i].lower().replace(" ", "_")
                                fund_data[key] = col.get_text(strip=True)
                            else:
                                fund_data[f"col_{i}"] = col.get_text(strip=True)
                        
                        # Extract fund name from first column
                        if cols[0]:
                            fund_name = cols[0].get_text(strip=True)
                            if fund_name and len(fund_name) > 3:
                                fund_data["fund_name"] = fund_name
                                funds.append(fund_data)
            
            print(f"    Found {len([f for f in funds if f.get('category') == category])} {category} funds")
            
        except Exception as e:
            print(f"    Error scraping {category}: {e}")
            continue
    
    print(f"  Total Value Research funds: {len(funds)}")
    return funds

# --- News and Trends Scraping ---
def scrape_financial_news(driver):
    print("Scraping Financial News and Trends...")
    news_data = []
    
    # Economic Times Mutual Fund News
    try:
        print("  Scraping ET Mutual Fund News...")
        url = "https://economictimes.indiatimes.com/mutual-funds"
        driver.get(url)
        time.sleep(5)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find news articles
        articles = soup.find_all(["article", "div"], class_=re.compile(r"article|news|story"))
        
        for article in articles[:20]:  # Limit to 20 articles
            try:
                # Extract title
                title_elem = article.find(["h1", "h2", "h3", "h4", "h5", "h6", "a"])
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                # Extract summary/description
                summary_elem = article.find(["p", "div"], class_=re.compile(r"summary|description|excerpt"))
                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                
                # Extract date
                date_elem = article.find(["span", "time"], class_=re.compile(r"date|time"))
                date = date_elem.get_text(strip=True) if date_elem else ""
                
                news_item = {
                    "title": title,
                    "summary": summary,
                    "date": date,
                    "source": url,
                    "platform": "EconomicTimes",
                    "type": "news",
                    "scraped_at": datetime.now().isoformat()
                }
                
                news_data.append(news_item)
                
            except Exception as e:
                continue
        
        print(f"    Found {len([n for n in news_data if n.get('platform') == 'EconomicTimes'])} ET news articles")
        
    except Exception as e:
        print(f"    Error scraping ET news: {e}")
    
    # Value Research News and Analysis
    try:
        print("  Scraping Value Research Analysis...")
        url = "https://www.valueresearchonline.com/funds/news/"
        driver.get(url)
        time.sleep(5)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find news articles
        articles = soup.find_all(["article", "div"], class_=re.compile(r"article|news|story"))
        
        for article in articles[:15]:  # Limit to 15 articles
            try:
                # Extract title
                title_elem = article.find(["h1", "h2", "h3", "h4", "h5", "h6", "a"])
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                # Extract summary
                summary_elem = article.find(["p", "div"], class_=re.compile(r"summary|description|excerpt"))
                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                
                # Extract date
                date_elem = article.find(["span", "time"], class_=re.compile(r"date|time"))
                date = date_elem.get_text(strip=True) if date_elem else ""
                
                news_item = {
                    "title": title,
                    "summary": summary,
                    "date": date,
                    "source": url,
                    "platform": "ValueResearch",
                    "type": "analysis",
                    "scraped_at": datetime.now().isoformat()
                }
                
                news_data.append(news_item)
                
            except Exception as e:
                continue
        
        print(f"    Found {len([n for n in news_data if n.get('platform') == 'ValueResearch'])} Value Research articles")
        
    except Exception as e:
        print(f"    Error scraping Value Research news: {e}")
    
    print(f"  Total news/trends items: {len(news_data)}")
    return news_data

# --- Economic Times Markets Scraper (Enhanced) ---
def scrape_etmarkets(driver):
    print("Scraping Economic Times Markets...")
    url = "https://economictimes.indiatimes.com/mutual-funds/top-mutual-funds"
    driver.get(url)
    time.sleep(5)
    
    funds = []
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Look for fund listings
    fund_elements = soup.find_all(["div", "article"], class_=re.compile(r"fund|mf|mutual"))
    
    for element in fund_elements:
        try:
            # Extract fund name
            name_elem = element.find(["h1", "h2", "h3", "h4", "h5", "h6", "a", "span"])
            if not name_elem:
                continue
                
            fund_name = name_elem.get_text(strip=True)
            if not fund_name or len(fund_name) < 3:
                continue
            
            fund_data = {
                "fund_name": fund_name,
                "source": url,
                "platform": "EconomicTimes",
                "scraped_at": datetime.now().isoformat()
            }
            
            # Try to extract additional details
            text_content = element.get_text(strip=True)
            
            # Look for returns
            returns_match = re.search(r'(\d+\.?\d*)\s*%', text_content)
            if returns_match:
                fund_data["returns"] = returns_match.group(1) + "%"
            
            # Look for AUM
            aum_match = re.search(r'₹?\s*(\d+(?:,\d+)*)\s*(?:Cr|Lakh|Crore)', text_content)
            if aum_match:
                fund_data["aum"] = aum_match.group(0)
            
            funds.append(fund_data)
            
        except Exception as e:
            continue
    
    print(f"  Found {len(funds)} funds on ET Markets.")
    return funds

# --- Main orchestrator ---
def main():
    driver = setup_driver(headless=True)
    all_funds = []
    all_news = []
    
    try:
        # Scrape funds from all sources
        all_funds.extend(scrape_amfi(driver))
        all_funds.extend(scrape_groww(driver))
        all_funds.extend(scrape_morningstar(driver))
        all_funds.extend(scrape_etmoney(driver))
        all_funds.extend(scrape_valueresearch(driver))
        all_funds.extend(scrape_etmarkets(driver))
        
        # Scrape news and trends
        all_news.extend(scrape_financial_news(driver))
        
    finally:
        driver.quit()
    
    print(f"\n=== SCRAPING SUMMARY ===")
    print(f"Total raw funds scraped: {len(all_funds)}")
    print(f"Total news/trends items: {len(all_news)}")
    
    # Clean and deduplicate funds
    cleaned_funds = [clean_fund_record(f) for f in all_funds if f]
    cleaned_funds = [f for f in cleaned_funds if f]
    unique_funds = remove_duplicates(cleaned_funds)
    
    print(f"Total unique funds: {len(unique_funds)}")
    
    # Save funds data
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(unique_funds, f, indent=2)
    print(f"Funds data saved to {OUTPUT_JSON}")
    
    # Save news data
    news_output = "data/financial_news_trends.json"
    with open(news_output, "w") as f:
        json.dump(all_news, f, indent=2)
    print(f"News/trends data saved to {news_output}")
    
    # Show sample data
    if unique_funds:
        print(f"\nSample fund data:")
        print(json.dumps(unique_funds[0], indent=2))
    
    if all_news:
        print(f"\nSample news data:")
        print(json.dumps(all_news[0], indent=2))
    
    print(f"\n=== READY FOR INGESTION ===")
    print(f"1. Funds: {OUTPUT_JSON}")
    print(f"2. News/Trends: {news_output}")
    print(f"Use these files to populate your vector store for comprehensive chatbot coverage!")

if __name__ == "__main__":
    main() 