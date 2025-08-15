import requests
from bs4 import BeautifulSoup
import redis
import json
import datetime
import psycopg2
import re
from ingestion.vector_store import VectorStore

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_KEY = "top_funds_cache"
CACHE_TTL = 60 * 60 * 24  # 24 hours

POSTGRES_PARAMS = {
    'host': 'localhost',
    'database': 'mutualfundpro',
    'user': 'postgres',
    'password': 'mutual@fund@pro12',
    'port': 5432
}

# --- Redis Setup ---
try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None

# --- PostgreSQL Snapshot Storage (stub) ---
def store_snapshot_in_postgres(funds, as_of_date):
    try:
        conn = psycopg2.connect(**POSTGRES_PARAMS)
        cur = conn.cursor()
        for f in funds:
            cur.execute("""
                INSERT INTO fund_snapshots (fund_name, category, return_1y, return_5y, expense_ratio, aum, as_of_date, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fund_name, as_of_date) DO NOTHING;
            """, (
                f.get('name'), f.get('category'), f.get('1y_return'), f.get('5y_return'),
                f.get('expense_ratio'), f.get('aum'), as_of_date, f.get('source')
            ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        pass  # For now, ignore DB errors

# --- Scraper Functions ---
def fetch_moneycontrol_top_funds(category=None):
    url = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/"
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "responsive"})
    funds = []
    if not table:
        return funds
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        name = cols[0].get_text(strip=True)
        cat = cols[1].get_text(strip=True)
        returns_1y = cols[2].get_text(strip=True)
        returns_3y = cols[3].get_text(strip=True)
        returns_5y = cols[4].get_text(strip=True)
        aum = cols[5].get_text(strip=True)
        # Expense ratio is not always available; set as N/A
        expense_ratio = "N/A"
        if category and category.lower() not in cat.lower():
            continue
        funds.append({
            "name": name,
            "category": cat,
            "1y_return": returns_1y,
            "3y_return": returns_3y,
            "5y_return": returns_5y,
            "aum": aum,
            "expense_ratio": expense_ratio,
            "source": "Moneycontrol"
        })
    return funds[:10]

def fetch_morningstar_top_funds(category=None):
    url = "https://www.morningstar.in/tools/top-mutual-funds.aspx"
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "ctl00_ctl00_cphMain_cphMain_grdTopFunds"})
    funds = []
    if not table:
        return funds
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        name = cols[0].get_text(strip=True)
        cat = cols[1].get_text(strip=True)
        returns_1y = cols[2].get_text(strip=True)
        returns_3y = cols[3].get_text(strip=True)
        returns_5y = cols[4].get_text(strip=True)
        aum = cols[5].get_text(strip=True)
        expense_ratio = "N/A"
        if category and category.lower() not in cat.lower():
            continue
        funds.append({
            "name": name,
            "category": cat,
            "1y_return": returns_1y,
            "3y_return": returns_3y,
            "5y_return": returns_5y,
            "aum": aum,
            "expense_ratio": expense_ratio,
            "source": "Morningstar"
        })
    return funds[:10]

def deduplicate_funds(funds):
    seen = set()
    deduped = []
    for f in funds:
        key = (f["name"].lower(), f["category"].lower())
        if key not in seen:
            deduped.append(f)
            seen.add(key)
    return deduped

def get_top_funds(category=None):
    """
    Returns a deduped, merged list of top funds from all sources, with Redis caching and fallback.
    """
    # Try Redis cache first
    if redis_client:
        cached = redis_client.get(CACHE_KEY + (":" + category if category else ""))
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
    # Live scrape from both sources
    try:
        funds_mc = fetch_moneycontrol_top_funds(category)
        funds_ms = fetch_morningstar_top_funds(category)
        all_funds = funds_mc + funds_ms
        deduped = deduplicate_funds(all_funds)
        top5 = deduped[:5]
        # Cache in Redis
        if redis_client:
            try:
                redis_client.set(CACHE_KEY + (":" + category if category else ""), json.dumps(top5), ex=CACHE_TTL)
            except Exception:
                pass
        # Store snapshot in Postgres
        as_of_date = datetime.datetime.now().strftime("%Y-%m-%d")
        store_snapshot_in_postgres(top5, as_of_date)
        return top5
    except Exception as e:
        # Fallback to last cache if available
        if redis_client:
            cached = redis_client.get(CACHE_KEY + (":" + category if category else ""))
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        return None

def get_top_funds_response(query: str = '', category: str = None) -> str:
    """
    Returns a user-friendly, ranked list of the top 5 mutual funds in India, as of today, with explanations and advice.
    If all fails, returns a friendly fallback message.
    """
    funds = get_top_funds(category)
    today = datetime.datetime.now().strftime("%B %d, %Y")
    if not funds:
        return "_Sorry, I'm unable to fetch the latest fund rankings right now. Please try again later._"
    # Optionally, use query/category to filter or highlight
    response = f"**Top 5 Mutual Funds in India as of {today}:**\n\n"
    for idx, f in enumerate(funds, 1):
        # 2-sentence explanation and advice
        explanation = (
            f"{f['name']} is a leading {f['category']} fund with a 1Y return of {f['1y_return']} and 5Y return of {f['5y_return']}. "
            f"AUM: {f['aum']}, Expense Ratio: {f['expense_ratio']}. "
            f"Consider this fund if you are looking for {f['category'].lower()} exposure, but always review the latest factsheet and your risk profile."
        )
        response += f"{idx}. **{f['name']}** ({f['category']})\n   - 1Y Return: {f['1y_return']} | 5Y Return: {f['5y_return']} | AUM: {f['aum']} | Expense Ratio: {f['expense_ratio']}\n   - {explanation}\n\n"
    return response 