from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime

TARGET_URLS = [
    # MoneyControl - Large Cap Funds
    "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/large-cap-fund.html",
    # MoneyControl - Mid Cap Funds  
    "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/mid-cap-fund.html",
    # MoneyControl - Small Cap Funds
    "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/small-cap-fund.html",
    # Value Research - Equity Funds
    "https://www.valueresearchonline.com/funds/fundSelector/?category=equity&plan=direct&tab=returns",
    # Economic Times - Top Mutual Funds
    "https://economictimes.indiatimes.com/mutual-funds/top-mutual-funds"
]

OUTPUT_JSON = "data/all_funds_selenium.json"

def setup_driver(headless=False):
    """Setup Firefox driver with appropriate options"""
    options = Options()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    try:
        driver = webdriver.Firefox(options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        print(f"Failed to setup Firefox driver: {e}")
        print("Trying Chrome as fallback...")
        try:
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            chrome_options = ChromeOptions()
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e2:
            print(f"Failed to setup Chrome driver: {e2}")
            return None

def wait_for_page_load(driver, timeout=10):
    """Wait for page to load completely"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        return True
    except TimeoutException:
        print("Page load timeout")
        return False

def wait_for_elements(driver, selector, timeout=10):
    """Wait for specific elements to appear"""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return True
    except TimeoutException:
        print(f"Timeout waiting for elements: {selector}")
        return False

def debug_page_content(driver, url):
    """Debug function to see what's on the page"""
    print(f"\n=== DEBUG: {url} ===")
    
    # Get page title
    title = driver.title
    print(f"Page title: {title}")
    
    # Check for common table selectors
    selectors_to_check = [
        "table",
        ".table",
        ".table4", 
        ".fund-table",
        ".mf-table",
        "[class*='table']",
        "[class*='fund']",
        "[class*='mutual']"
    ]
    
    for selector in selectors_to_check:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            print(f"Found {len(elements)} elements with selector: {selector}")
            for i, elem in enumerate(elements[:3]):  # Show first 3
                try:
                    text = elem.text[:100] if elem.text else "No text"
                    print(f"  Element {i+1}: {text}...")
                except:
                    print(f"  Element {i+1}: Could not get text")
    
    # Check for any text content
    body_text = driver.find_element(By.TAG_NAME, "body").text
    if body_text:
        print(f"Body text length: {len(body_text)}")
        print(f"First 200 chars: {body_text[:200]}...")
    
    print("=== END DEBUG ===\n")

def extract_moneycontrol_funds(html, source_url):
    """Extract funds from MoneyControl pages"""
    soup = BeautifulSoup(html, "html.parser")
    funds = []
    
    # Look for fund tables with various selectors
    table_selectors = [
        "table",
        ".table4", 
        ".table",
        ".fund-table",
        ".mf-table",
        "[class*='table']",
        "[class*='fund']"
    ]
    
    for selector in table_selectors:
        tables = soup.select(selector)
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
                
            # Extract headers from first row
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))
            
            # Process data rows
            for row in rows[1:]:
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    fund = {
                        "source": source_url,
                        "scraped_at": datetime.now().isoformat(),
                        "platform": "MoneyControl"
                    }
                    
                    for i, col in enumerate(cols):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        fund[key] = col.get_text(strip=True)
                    
                    # Look for fund name in first column
                    if cols and cols[0]:
                        fund_name = cols[0].get_text(strip=True)
                        if fund_name and len(fund_name) > 2:
                            fund["fund_name"] = fund_name
                            funds.append(fund)
    
    return funds

def extract_valueresearch_funds(html, source_url):
    """Extract funds from Value Research pages"""
    soup = BeautifulSoup(html, "html.parser")
    funds = []
    
    # Look for fund data in various formats
    fund_selectors = [
        ".fund-item",
        ".fund-row", 
        ".fund-data",
        "[class*='fund']",
        "tr",
        ".list-item"
    ]
    
    for selector in fund_selectors:
        elements = soup.select(selector)
        for element in elements:
            fund = {
                "source": source_url,
                "scraped_at": datetime.now().isoformat(),
                "platform": "ValueResearch"
            }
            
            # Extract text content
            text_content = element.get_text(strip=True)
            if text_content and len(text_content) > 10:
                fund["raw_content"] = text_content
                
                # Try to extract fund name
                fund_name_elem = element.find(["a", "span", "div"], class_=["fund-name", "name"])
                if fund_name_elem:
                    fund["fund_name"] = fund_name_elem.get_text(strip=True)
                
                funds.append(fund)
    
    return funds

def extract_economictimes_funds(html, source_url):
    """Extract funds from Economic Times pages"""
    soup = BeautifulSoup(html, "html.parser")
    funds = []
    
    # Look for fund tables and lists
    table_selectors = [
        "table",
        ".table",
        ".fund-table",
        ".mf-table"
    ]
    
    for selector in table_selectors:
        tables = soup.select(selector)
        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:  # Skip header
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    fund = {
                        "source": source_url,
                        "scraped_at": datetime.now().isoformat(),
                        "platform": "EconomicTimes"
                    }
                    
                    for i, col in enumerate(cols):
                        fund[f"col_{i}"] = col.get_text(strip=True)
                    
                    if cols[0]:
                        fund["fund_name"] = cols[0].get_text(strip=True)
                        funds.append(fund)
    
    return funds

def extract_tables_from_html(html, source_url):
    """Extract fund data based on the source URL"""
    if "moneycontrol.com" in source_url:
        return extract_moneycontrol_funds(html, source_url)
    elif "valueresearchonline.com" in source_url:
        return extract_valueresearch_funds(html, source_url)
    elif "economictimes.indiatimes.com" in source_url:
        return extract_economictimes_funds(html, source_url)
    else:
        # Generic extraction for unknown sources
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        all_funds = []
        
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
                
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))
            
            for row in rows[1:]:
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    fund = {
                        "source": source_url,
                        "scraped_at": datetime.now().isoformat(),
                        "platform": "Generic"
                    }
                    
                    for i, col in enumerate(cols):
                        key = headers[i] if i < len(headers) else f"col_{i}"
                        fund[key] = col.get_text(strip=True)
                    
                    all_funds.append(fund)
        
        return all_funds

def selenium_scrape_all_funds(urls=TARGET_URLS, output_json=OUTPUT_JSON, headless=False):
    """Main scraping function"""
    driver = setup_driver(headless=headless)
    if not driver:
        print("Failed to setup web driver")
        return
    
    all_funds = []
    
    try:
        for url in urls:
            print(f"\nScraping: {url}")
            try:
                driver.get(url)
                
                # Wait for page to load
                if not wait_for_page_load(driver):
                    print(f"  Warning: Page load timeout for {url}")
                
                # Additional wait for dynamic content
                time.sleep(5)
                
                # Debug page content
                debug_page_content(driver, url)
                
                html = driver.page_source
                funds = extract_tables_from_html(html, url)
                
                print(f"  Found {len(funds)} funds on this page")
                all_funds.extend(funds)
                
            except Exception as e:
                print(f"  Error scraping {url}: {e}")
                continue
                
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        if not headless:
            input("Press Enter to close browser...")
        driver.quit()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save results
    with open(output_json, "w") as f:
        json.dump(all_funds, f, indent=2)
    
    print(f"\nScraping completed!")
    print(f"Total funds scraped: {len(all_funds)}")
    print(f"Results saved to: {output_json}")
    
    # Show sample of scraped data
    if all_funds:
        print(f"\nSample fund data:")
        print(json.dumps(all_funds[0], indent=2))

def main():
    # Run in non-headless mode for debugging
    selenium_scrape_all_funds(headless=False)

if __name__ == "__main__":
    main() 