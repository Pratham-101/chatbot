import json
import os
from datetime import datetime

# Import your existing vector store utilities
try:
    from ingestion.vector_store import VectorStore
    from ingestion.structured_data_loader import StructuredDataLoader
except ImportError:
    print("Warning: Could not import vector store modules. Please ensure they exist.")
    VectorStore = None
    StructuredDataLoader = None

FUNDS_JSON = "data/all_indian_funds_cleaned.json"
NEWS_JSON = "data/financial_news_trends.json"

def load_json_data(file_path):
    """Load JSON data from file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {file_path}")
    return data

def format_fund_for_ingestion(fund):
    """Format fund data for vector store ingestion"""
    # Create a comprehensive text representation
    text_parts = []
    
    # Basic fund info
    text_parts.append(f"Fund Name: {fund.get('fund_name', 'N/A')}")
    
    # NAV information
    if fund.get('nav'):
        text_parts.append(f"NAV: ₹{fund.get('nav')}")
    
    # Platform and source
    text_parts.append(f"Source: {fund.get('platform', 'N/A')}")
    
    # Additional details
    if fund.get('category'):
        text_parts.append(f"Category: {fund.get('category')}")
    if fund.get('returns'):
        text_parts.append(f"Returns: {fund.get('returns')}")
    if fund.get('aum'):
        text_parts.append(f"AUM: {fund.get('aum')}")
    if fund.get('rating'):
        text_parts.append(f"Rating: {fund.get('rating')}")
    
    # Date information
    if fund.get('date'):
        text_parts.append(f"Date: {fund.get('date')}")
    
    text_parts.append(f"Last Updated: {fund.get('scraped_at', 'N/A')}")
    
    return {
        "text": " | ".join(text_parts),
        "metadata": {
            "type": "mutual_fund",
            "fund_name": fund.get('fund_name', ''),
            "platform": fund.get('platform', ''),
            "category": fund.get('category', ''),
            "nav": fund.get('nav', ''),
            "returns": fund.get('returns', ''),
            "aum": fund.get('aum', ''),
            "rating": fund.get('rating', ''),
            "source": fund.get('source', ''),
            "scraped_at": fund.get('scraped_at', '')
        }
    }

def format_news_for_ingestion(news_item):
    """Format news data for vector store ingestion"""
    text_parts = []
    
    # Title
    text_parts.append(f"Title: {news_item.get('title', 'N/A')}")
    
    # Summary
    if news_item.get('summary'):
        text_parts.append(f"Summary: {news_item.get('summary')}")
    
    # Date
    if news_item.get('date'):
        text_parts.append(f"Date: {news_item.get('date')}")
    
    # Platform and type
    text_parts.append(f"Source: {news_item.get('platform', 'N/A')}")
    text_parts.append(f"Type: {news_item.get('type', 'N/A')}")
    
    text_parts.append(f"Published: {news_item.get('scraped_at', 'N/A')}")
    
    return {
        "text": " | ".join(text_parts),
        "metadata": {
            "type": "financial_news",
            "title": news_item.get('title', ''),
            "summary": news_item.get('summary', ''),
            "platform": news_item.get('platform', ''),
            "news_type": news_item.get('type', ''),
            "date": news_item.get('date', ''),
            "source": news_item.get('source', ''),
            "scraped_at": news_item.get('scraped_at', '')
        }
    }

def ingest_to_vector_store():
    """Main ingestion function"""
    print("=== COMPREHENSIVE DATA INGESTION ===")
    
    # Load funds data
    print("\n1. Loading funds data...")
    funds_data = load_json_data(FUNDS_JSON)
    
    # Load news data
    print("\n2. Loading news/trends data...")
    news_data = load_json_data(NEWS_JSON)
    
    if not funds_data and not news_data:
        print("No data to ingest!")
        return
    
    # Format data for ingestion
    print("\n3. Formatting data for ingestion...")
    formatted_data = []
    
    # Format funds
    for fund in funds_data:
        formatted_data.append(format_fund_for_ingestion(fund))
    
    # Format news
    for news_item in news_data:
        formatted_data.append(format_news_for_ingestion(news_item))
    
    print(f"Formatted {len(formatted_data)} items for ingestion")
    
    # Ingest to vector store
    print("\n4. Ingesting to vector store...")
    
    if VectorStore and StructuredDataLoader:
        try:
            # Initialize vector store
            vector_store = VectorStore()
            
            # Ingest formatted data
            for item in formatted_data:
                vector_store.add_texts(
                    texts=[item["text"]],
                    metadatas=[item["metadata"]]
                )
            
            print(f"Successfully ingested {len(formatted_data)} items to vector store!")
            
        except Exception as e:
            print(f"Error during ingestion: {e}")
            print("Please check your vector store configuration.")
    else:
        print("Vector store modules not available.")
        print("Please ensure your vector store is properly configured.")
        print("\nFormatted data sample:")
        if formatted_data:
            print(json.dumps(formatted_data[0], indent=2))
    
    print("\n=== INGESTION COMPLETE ===")
    print(f"Total items processed: {len(formatted_data)}")
    print("Your chatbot now has access to comprehensive Indian mutual fund data!")

def main():
    ingest_to_vector_store()

if __name__ == "__main__":
    main() 