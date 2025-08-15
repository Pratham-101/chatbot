import asyncio
import json
import time
import re
import os
from typing import List, Dict, Optional, Tuple
import httpx
from duckduckgo_search import DDGS
from groq import Groq, APIError
import threading
import datetime
import logging
from services.chatbot.fund_rankings import get_top_funds
from ingestion.vector_store import VectorStore
from services.chatbot.real_time_data import real_time_provider, market_data_provider
from services.chatbot.response_quality import response_evaluator, structured_generator, ResponseQuality, StructuredResponse
import spacy
from services.chatbot.knowledge_graph import MutualFundKnowledgeGraph
from services.chatbot.web_search import WebSearch
from ingestion.structured_data_loader import StructuredDataLoader
from pipeline import MutualFundPipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
from services.data.web_search import WebSearch as DataWebSearch
from services.data.web_scraper import get_top_funds_moneycontrol, get_top_funds_valueresearch, get_top_funds_amfi, scrape_moneycontrol_fund_details, scrape_valueresearch_fund_details
from services.data.fund_aggregator import aggregate_fund_data

from services.chatbot.intent_classifier import intent_classifier, QueryIntent

# Production API configuration
PRODUCTION_API_BASE = "http://34.122.133.139:4000"

# --- Start of GroqClient Definition ---
class GroqClient:
    def __init__(self, model: str):
        self.model = model
        try:
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        except KeyError:
            print("ERROR: GROQ_API_KEY environment variable not set.")
            self.client = None

    async def generate(self, prompt: str) -> str:
        """Generate a response from the Groq model."""
        if not self.client:
            return self._fallback_response(prompt)
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7,
                max_tokens=2048,
                stream=False,
            )
            return chat_completion.choices[0].message.content or ""
        except APIError as e:
            print(f"Groq API error: {e}")
            return self._fallback_response(prompt)
        except Exception as e:
            print(f"An unexpected error occurred with Groq: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """Provide a fallback response when the LLM is not available."""
        # This fallback logic is triggered if the API key is missing or the call fails.
        web_results = ""
        if "Source 2: Real-Time Web Search Results" in prompt:
            web_start = prompt.find("---", prompt.find("Source 2: Real-Time Web Search Results")) + 3
            web_end = prompt.find("====================", web_start)
            web_results = prompt[web_start:web_end].strip()

        if web_results and "Snippet:" in web_results:
            response = "I couldn't connect to the advanced model, but here's what I found on the web:\n\n"
            snippets = [line.replace("Snippet: ", "") for line in web_results.split('\n') if line.startswith("Snippet:")]
            for i, snippet in enumerate(snippets[:3], 1):
                response += f"{i}. {snippet}\n\n"
            response += "For more details, I recommend visiting the source links or consulting a financial advisor."
        else:
            response = "I am currently unable to process your request. Please try again later."
            
        return response
# --- End of GroqClient Definition ---

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

USER_SESSION_FILE = "user_sessions.json"
USER_SESSION_LOCK = threading.Lock()

def load_user_session(user_id: str) -> dict:
    if not os.path.exists(USER_SESSION_FILE):
        return {}
    with USER_SESSION_LOCK, open(USER_SESSION_FILE, "r", encoding="utf-8") as f:
        try:
            sessions = json.load(f)
        except Exception:
            return {}
    return sessions.get(user_id, {})

def save_user_session(user_id: str, session: dict):
    sessions = {}
    if os.path.exists(USER_SESSION_FILE):
        with USER_SESSION_LOCK, open(USER_SESSION_FILE, "r", encoding="utf-8") as f:
            try:
                sessions = json.load(f)
            except Exception:
                sessions = {}
    sessions[user_id] = session
    with USER_SESSION_LOCK, open(USER_SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

def is_generic_answer(answer):
    return not answer or len(answer.strip()) < 10

class EnhancedMutualFundChatbot:
    """
    A chatbot that answers queries about mutual funds by combining information
    from the production API and real-time web search for enhanced context.
    """
    def __init__(self, model_name="llama3-8b-8192"):
        self.client = GroqClient(model=model_name)
        self.pipeline = MutualFundPipeline()
        self.vector_store: Optional[VectorStore] = None
        self.web_search_tool = None 
        self.knowledge_graph = MutualFundKnowledgeGraph()
        self.structured_data_loader = StructuredDataLoader()

    def set_vector_store(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def set_web_search_tool(self, tool):
        self.web_search_tool = tool

    async def _get_fund_from_api(self, fund_name: str) -> Dict:
        """
        Query the production API for fund data using fund name or ISIN.
        Returns comprehensive fund data including NAV, AUM, performance, holdings, etc.
        """
        print(f"[DEBUG] Searching API for fund: '{fund_name}'")
        try:
            async with httpx.AsyncClient() as client:
                # Search for funds by name across multiple pages
                all_funds = []
                page = 1
                while page <= 5:  # Check first 5 pages
                    response = await client.get(f"{PRODUCTION_API_BASE}/api/funds/", params={"page": page, "limit": 100})
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "success":
                            funds = data.get("data", [])
                            all_funds.extend(funds)
                            print(f"[DEBUG] Found {len(funds)} funds on page {page}")
                            if len(funds) < 100:  # Last page
                                break
                        page += 1
                    else:
                        print(f"[DEBUG] API request failed with status {response.status_code}")
                        break
                
                print(f"[DEBUG] Total funds found: {len(all_funds)}")
                
                # Find matching fund by name (case-insensitive search)
                matching_funds = []
                fund_name_lower = fund_name.lower()
                for fund in all_funds:
                    scheme_name = fund.get("scheme_name", "").lower()
                    if fund_name_lower in scheme_name or scheme_name in fund_name_lower:
                        matching_funds.append(fund)
                        print(f"[DEBUG] Found matching fund: {fund.get('scheme_name')}")
                
                if matching_funds:
                    # Return the first matching fund with all its data
                    fund_data = matching_funds[0]
                    print(f"[DEBUG] Selected fund: {fund_data.get('scheme_name')}")
                    print(f"[DEBUG] Fund data: NAV={fund_data.get('nav')}, Expense Ratio={fund_data.get('expense_ratio')}, Manager={fund_data.get('fund_manager')}")
                    return fund_data
                else:
                    print(f"[DEBUG] No matching funds found for '{fund_name}'")
                    # Show some available funds for debugging
                    print(f"[DEBUG] Sample available funds:")
                    for i, fund in enumerate(all_funds[:5]):
                        print(f"  {i+1}. {fund.get('scheme_name')}")
                
                # If not found by name, try direct ISIN lookup if fund_name looks like an ISIN
                if len(fund_name) == 12 and fund_name.isalnum():
                    print(f"[DEBUG] Trying ISIN lookup for: {fund_name}")
                    response = await client.get(f"{PRODUCTION_API_BASE}/api/funds/{fund_name}/")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "success":
                            return data.get("data", {})
                        
        except Exception as e:
            print(f"[DEBUG] API query error for '{fund_name}': {e}")
        
        return {}

    async def _get_fund_data_robust(self, fund_query: str) -> dict:
        """
        Get fund data from the production API, with web search as enhancement.
        """
        print(f"[DEBUG] Getting data for: {fund_query}")
        
        # Extract fund name from query (remove common phrases)
        fund_name = fund_query
        common_phrases = [
            "tell me about", "what is", "show me", "give me information about",
            "details of", "information about", "analysis of", "overview of"
        ]
        for phrase in common_phrases:
            if phrase.lower() in fund_query.lower():
                fund_name = fund_query.lower().replace(phrase.lower(), "").strip()
                break
        
        print(f"[DEBUG] Extracted fund name: '{fund_name}'")
        
        # 1. Try to get data from the production API
        api_data = await self._get_fund_from_api(fund_name)
        print(f"[DEBUG] API data result: {bool(api_data)}")
        
        # 2. Get web search results for additional context
        web_data = await self._perform_web_search(fund_query, api_data)
        print(f"[DEBUG] Web data result: {bool(web_data)}")
        
        # 3. Combine API data with web search for comprehensive context
        combined_data = {
            'api_data': api_data,
            'web_data': web_data,
            'fund_query': fund_query,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return combined_data

    async def _handle_comparison_query(self, query: str) -> dict:
        """Handle fund comparison queries."""
        print(f"[DEBUG] Handling comparison query: {query}")
        
        # Extract fund names from comparison query
        fund_names = await self._extract_fund_names_with_spacy(query)
        
        if len(fund_names) < 2:
            return {
                "answer": "Please specify two funds to compare. Example: 'Compare Axis Bluechip Fund and HDFC Flexicap Fund'",
                "quality_score": 5.0,
                "sources": [],
                "response_time": 0.1,
                "quality_metrics": {"accuracy": 0.5, "completeness": 0.5, "clarity": 1.0, "relevance": 0.5, "overall_score": 5.0, "feedback": "Need two fund names"},
                "structured_data": {"summary": "Comparison query needs two fund names"}
            }
        
        fund1_name = fund_names[0]
        fund2_name = fund_names[1]
        
        # Get data for both funds
        fund1_data = await self._get_fund_from_api(fund1_name)
        fund2_data = await self._get_fund_from_api(fund2_name)
        
        if not fund1_data or not fund2_data:
            return {
                "answer": f"One or both funds not found in database. Found: {bool(fund1_data)} and {bool(fund2_data)}",
                "quality_score": 5.0,
                "sources": [],
                "response_time": 0.1,
                "quality_metrics": {"accuracy": 0.5, "completeness": 0.5, "clarity": 1.0, "relevance": 0.5, "overall_score": 5.0, "feedback": "Funds not found"},
                "structured_data": {"summary": "Funds not found in database"}
            }
        
        # Generate comparison response
        comparison_response = f"""
# Fund Comparison: {fund1_data.get('scheme_name', fund1_name)} vs {fund2_data.get('scheme_name', fund2_name)}

## Comparison Table

| Metric | {fund1_data.get('scheme_name', fund1_name)[:30]} | {fund2_data.get('scheme_name', fund2_name)[:30]} |
|--------|--------------------------------|--------------------------------|
| **NAV** | ₹{fund1_data.get('nav', 'N/A')} | ₹{fund2_data.get('nav', 'N/A')} |
| **Expense Ratio** | {fund1_data.get('expense_ratio', 'N/A')}% | {fund2_data.get('expense_ratio', 'N/A')}% |
| **Risk Category** | {fund1_data.get('sebi_risk_category', 'N/A')} | {fund2_data.get('sebi_risk_category', 'N/A')} |
| **1Y Return** | {fund1_data.get('return_1y', 'N/A')}% | {fund2_data.get('return_1y', 'N/A')}% |
| **3Y Return** | {fund1_data.get('return_3y', 'N/A')}% | {fund2_data.get('return_3y', 'N/A')}% |
| **5Y Return** | {fund1_data.get('return_5y', 'N/A')}% | {fund2_data.get('return_5y', 'N/A')}% |
| **Fund Type** | {fund1_data.get('fund_type', 'N/A')} | {fund2_data.get('fund_type', 'N/A')} |
| **Fund Manager** | {fund1_data.get('fund_manager', 'N/A')} | {fund2_data.get('fund_manager', 'N/A')} |

## Key Differences

**Risk Profile**: {fund1_data.get('sebi_risk_category', 'N/A')} vs {fund2_data.get('sebi_risk_category', 'N/A')}

**Expense Ratio**: {fund1_data.get('expense_ratio', 'N/A')}% vs {fund2_data.get('expense_ratio', 'N/A')}% 
*Lower expense ratio means more returns for investors*

**Performance**: Compare the 1Y, 3Y, and 5Y returns to see which fund has performed better over different time periods.

*Note: Past performance doesn't guarantee future returns. Always check the latest factsheets for current information.*
"""
        
        return {
            "answer": comparison_response,
            "quality_score": 10.0,
            "sources": [f"Production API: {fund1_data.get('scheme_name', fund1_name)}", f"Production API: {fund2_data.get('scheme_name', fund2_name)}"],
            "response_time": 0.5,
            "quality_metrics": {"accuracy": 1.0, "completeness": 1.0, "clarity": 1.0, "relevance": 1.0, "overall_score": 10.0, "feedback": "Fund comparison successful"},
            "structured_data": {"summary": comparison_response}
        }

    def _build_api_enhanced_context(self, fund_data: dict, query: str) -> str:
        """
        Build context string from API data and web search results.
        """
        api_data = fund_data.get('api_data', {})
        web_data = fund_data.get('web_data', '')
        
        context_parts = []
        
        # Add API data with correct field names
        if api_data:
            context_parts.append("=== PRODUCTION API DATA ===")
            context_parts.append(f"Fund Name: {api_data.get('scheme_name', 'N/A')}")
            context_parts.append(f"AMC: {api_data.get('amc_name', 'N/A')}")
            context_parts.append(f"ISIN: {api_data.get('isin', 'N/A')}")
            context_parts.append(f"NAV: ₹{api_data.get('nav', 'N/A')}")
            context_parts.append(f"Expense Ratio: {api_data.get('expense_ratio', 'N/A')}%")
            context_parts.append(f"Fund Type: {api_data.get('fund_type', 'N/A')}")
            context_parts.append(f"Fund Subtype: {api_data.get('fund_subtype', 'N/A')}")
            context_parts.append(f"Risk Category: {api_data.get('sebi_risk_category', 'N/A')}")
            context_parts.append(f"Plan: {api_data.get('plan', 'N/A')}")
            
            # Fund Manager
            fund_manager = api_data.get('fund_manager', 'N/A')
            context_parts.append(f"Fund Manager(s): {fund_manager}")
            
            # Performance data
            context_parts.append("Performance:")
            context_parts.append(f"  1 Year Return: {api_data.get('return_1y', 'N/A')}%")
            context_parts.append(f"  3 Year Return: {api_data.get('return_3y', 'N/A')}%")
            context_parts.append(f"  5 Year Return: {api_data.get('return_5y', 'N/A')}%")
        
        # Add web search context
        if web_data:
            context_parts.append("\n=== WEB SEARCH CONTEXT ===")
            context_parts.append(web_data)
        
        return "\n".join(context_parts)

    async def _perform_web_search(self, query: str, api_data: dict) -> str:
        """
        Performs a comprehensive web search for fund information including official websites and detailed analysis.
        """
        print(f"[DEBUG] Entering _perform_web_search for: '{query}'")
        
        try:
            from duckduckgo_search import DDGS
            
            print(f"[DEBUG] Starting enhanced web search for: '{query}'")
            
            def do_search():
                with DDGS() as ddgs:
                    # Search for fund-specific information using the actual fund name
                    fund_results = list(ddgs.text(f"{query} mutual fund holdings performance", max_results=3))
                    
                    # Search for official fund website using AMC name
                    amc_name = api_data.get('amc_name', '') if api_data else ''
                    website_results = list(ddgs.text(f"{amc_name} mutual fund official website", max_results=2))
                    
                    # Search for fund analysis and reviews
                    analysis_results = list(ddgs.text(f"{query} fund analysis review", max_results=2))
                    
                    return {
                        'fund_info': fund_results,
                        'websites': website_results,
                        'analysis': analysis_results
                    }
            
            # Run the search
            results = await asyncio.get_event_loop().run_in_executor(None, do_search)
            
            if results:
                formatted_results = []
                
                # Format fund information with clickable links
                if results.get('fund_info'):
                    formatted_results.append("**📊 Fund Information & Analysis:**")
                    for i, result in enumerate(results['fund_info'], 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', 'No content')
                        link = result.get('link', 'No link')
                        formatted_results.append(f"{i}. **{title}**")
                        formatted_results.append(f"   {body}")
                        formatted_results.append(f"   🔗 [Read More]({link})")
                        formatted_results.append("")
                
                # Format official websites with clickable links
                if results.get('websites'):
                    formatted_results.append("**🏢 Official Fund Websites:**")
                    for i, result in enumerate(results['websites'], 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', 'No content')
                        link = result.get('link', 'No link')
                        formatted_results.append(f"{i}. **{title}**")
                        formatted_results.append(f"   {body}")
                        formatted_results.append(f"   🔗 [Visit Website]({link})")
                        formatted_results.append("")
                
                # Format analysis and reviews with clickable links
                if results.get('analysis'):
                    formatted_results.append("**📈 Fund Analysis & Reviews:**")
                    for i, result in enumerate(results['analysis'], 1):
                        title = result.get('title', 'No title')
                        body = result.get('body', 'No content')
                        link = result.get('link', 'No link')
                        formatted_results.append(f"{i}. **{title}**")
                        formatted_results.append(f"   {body}")
                        formatted_results.append(f"   🔗 [Read Analysis]({link})")
                        formatted_results.append("")
                
                return "\n".join(formatted_results)
            else:
                return "No web search results found."
                
        except Exception as e:
            print(f"[DEBUG] Error during web search for '{query}': {e}")
            # Fallback: Provide useful fund information even without web search
            fund_type = api_data.get('fund_type', 'Equity') if api_data else 'Equity'
            fund_subtype = api_data.get('fund_subtype', 'Sectoral/Thematic') if api_data else 'Sectoral/Thematic'
            risk_category = api_data.get('sebi_risk_category', 'Very High') if api_data else 'Very High'
            amc_name = api_data.get('amc_name', 'AMC') if api_data else 'AMC'
            
            fallback_info = f"""
**Fund Analysis (Based on Fund Type):**

**Investment Strategy**: This is a {fund_type} scheme with {fund_subtype} focus. The fund follows a bottom-up stock selection approach, focusing on companies with strong fundamentals, competitive advantages, and growth potential.

**Investment Approach**: 
- Focuses on companies with strong market position and competitive moats
- Emphasizes companies with robust financials and sustainable growth
- Considers government policies and economic cycles affecting the sector
- Maintains a diversified portfolio to manage sector-specific risks

**Risk Factors**:
- Sector concentration risk due to focus on {fund_subtype.lower()}
- Economic cycle sensitivity
- Policy and regulatory changes
- Market volatility and sector-specific risks

**Suitable For**: Investors seeking exposure to {fund_subtype.lower()} growth with {risk_category.lower()} risk tolerance and long-term investment horizon (5+ years).

**Official Resources**: 
- Visit {amc_name} website for latest NAV, AUM, and detailed holdings
- Check SEBI website for regulatory disclosures
- Download latest factsheet for comprehensive fund analysis
"""
            
            return fallback_info

    async def compare_funds(self, fund1_name: str, fund2_name: str) -> str:
        """Compare two funds side by side."""
        print(f"[DEBUG] Comparing funds: {fund1_name} vs {fund2_name}")
        
        # Get data for both funds
        fund1_data = await self._get_fund_from_api(fund1_name)
        fund2_data = await self._get_fund_from_api(fund2_name)
        
        if not fund1_data or not fund2_data:
            return "One or both funds not found in database."
        
        # Generate comparison table
        comparison = f"""
# Fund Comparison: {fund1_data.get('scheme_name', fund1_name)} vs {fund2_data.get('scheme_name', fund2_name)}

## Comparison Table

| Metric | {fund1_data.get('scheme_name', fund1_name)[:30]} | {fund2_data.get('scheme_name', fund2_name)[:30]} |
|--------|--------------------------------|--------------------------------|
| **NAV** | ₹{fund1_data.get('nav', 'N/A')} | ₹{fund2_data.get('nav', 'N/A')} |
| **Expense Ratio** | {fund1_data.get('expense_ratio', 'N/A')}% | {fund2_data.get('expense_ratio', 'N/A')}% |
| **Risk Category** | {fund1_data.get('sebi_risk_category', 'N/A')} | {fund2_data.get('sebi_risk_category', 'N/A')} |
| **1Y Return** | {fund1_data.get('return_1y', 'N/A')}% | {fund2_data.get('return_1y', 'N/A')}% |
| **3Y Return** | {fund1_data.get('return_3y', 'N/A')}% | {fund2_data.get('return_3y', 'N/A')}% |
| **5Y Return** | {fund1_data.get('return_5y', 'N/A')}% | {fund2_data.get('return_5y', 'N/A')}% |
| **Fund Type** | {fund1_data.get('fund_type', 'N/A')} | {fund2_data.get('fund_type', 'N/A')} |
| **Fund Manager** | {fund1_data.get('fund_manager', 'N/A')} | {fund2_data.get('fund_manager', 'N/A')} |

## Key Differences

**Risk Profile**: {fund1_data.get('sebi_risk_category', 'N/A')} vs {fund2_data.get('sebi_risk_category', 'N/A')}

**Expense Ratio**: {fund1_data.get('expense_ratio', 'N/A')}% vs {fund2_data.get('expense_ratio', 'N/A')}% 
*Lower expense ratio means more returns for investors*

**Performance**: Compare the 1Y, 3Y, and 5Y returns to see which fund has performed better over different time periods.

*Note: Past performance doesn't guarantee future returns. Always check the latest factsheets for current information.*
"""
        
        return comparison

    async def process_query(self, query: str, force_web: bool = False, user_id: str = "default") -> dict:
        """
        Process a user query using intent classification and route to the correct handler.
        """
        start_time = time.time()
        print(f"[Chatbot] Processing query: '{query}' for user {user_id} (force_web={force_web})")

        # 1. Intent classification
        from services.chatbot.intent_classifier import intent_classifier, QueryIntent
        analysis = intent_classifier.classify_intent(query)
        print(f"[DEBUG] Detected intent: {analysis.intent}")
        print(f"[DEBUG] Extracted entities: {analysis.entities}")

        # 2. Route to the correct handler
        if analysis.intent == QueryIntent.COMPARE_FUNDS:
            fund_names = analysis.entities.fund_names
            if len(fund_names) < 2:
                return {
                    "answer": "Please specify two funds to compare. Example: 'Compare Axis Bluechip Fund and HDFC Flexicap Fund'",
                    "quality_score": 5.0,
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "quality_metrics": {"accuracy": 0.5, "completeness": 0.5, "clarity": 1.0, "relevance": 0.5, "overall_score": 5.0, "feedback": "Need two fund names"},
                    "structured_data": {"summary": "Comparison query needs two fund names"}
                }
            fund1, fund2 = fund_names[:2]
            return await self._handle_comparison_query(f"{fund1} vs {fund2}")

        elif analysis.intent == QueryIntent.FUND_ANALYSIS or analysis.intent == QueryIntent.FUND_SUITABILITY:
            # Use the first fund name found
            fund_name = analysis.entities.fund_names[0] if analysis.entities.fund_names else query
            fund_data = await self._get_fund_data_robust(fund_name)
            context = self._build_api_enhanced_context(fund_data, query)
            print(f"[DEBUG] API data available: {bool(fund_data.get('api_data'))}")
            if fund_data.get('api_data'):
                api_data = fund_data['api_data']
                response = self._generate_fund_analysis_response(api_data, fund_data, query)
                return {
                    "answer": response,
                    "quality_score": 10.0,
                    "sources": [f"Production API: {api_data.get('scheme_name', 'Fund Data')}"],
                    "response_time": time.time() - start_time,
                    "quality_metrics": {
                        "accuracy": 1.0,
                        "completeness": 1.0,
                        "clarity": 1.0,
                        "relevance": 1.0,
                        "overall_score": 10.0,
                        "feedback": "API data used"
                    },
                    "structured_data": {
                        "summary": response
                    }
                }
            else:
                fallback_msg = "This fund is not available in our database. Please check the official AMC website for current information."
                return {
                    "answer": fallback_msg,
                    "quality_score": 5.0,
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "quality_metrics": {
                        "accuracy": 0.0,
                        "completeness": 0.0,
                        "clarity": 1.0,
                        "relevance": 0.0,
                        "overall_score": 5.0,
                        "feedback": "No API data"
                    },
                    "structured_data": {
                        "summary": fallback_msg
                    }
                }

        # 3. General finance Q&A fallback (web search or static FAQ)
        else:
            print("[DEBUG] Routing to general finance Q&A fallback.")
            web_data = await self._perform_web_search(query, api_data={})
            if web_data and 'No web search results found' not in web_data:
                answer = f"**General Finance Answer:**\n\n{web_data}"
            else:
                answer = "Sorry, I couldn't find an answer to your question. Please try rephrasing or ask about a specific mutual fund."
            return {
                "answer": answer,
                "quality_score": 6.0,
                "sources": ["Web Search"],
                "response_time": time.time() - start_time,
                "quality_metrics": {
                    "accuracy": 0.5,
                    "completeness": 0.5,
                    "clarity": 1.0,
                    "relevance": 0.5,
                    "overall_score": 6.0,
                    "feedback": "General finance fallback"
                },
                "structured_data": {
                    "summary": answer
                }
            }

    def _generate_fund_analysis_response(self, api_data, fund_data, query):
        """
        Generate a detailed, ChatGPT-style answer for a mutual fund.
        """
        # Compose summary
        scheme_name = api_data.get("scheme_name", "Unknown Fund")
        amc_name = api_data.get("amc_name", "Unknown AMC")
        nav = api_data.get("nav", "N/A")
        expense_ratio = api_data.get("expense_ratio", "N/A")
        manager = api_data.get("fund_manager", "N/A")
        fund_type = api_data.get("fund_type", "N/A")
        fund_subtype = api_data.get("fund_subtype", "N/A")
        risk = api_data.get("sebi_risk_category", "N/A")
        returns_1y = api_data.get("return_1y", "N/A")
        returns_3y = api_data.get("return_3y", "N/A")
        returns_5y = api_data.get("return_5y", "N/A")
        plan = api_data.get("plan", "N/A")
        isin = api_data.get("isin", "N/A")

        # Table of key metrics
        metrics_table = f"""
| Metric           | Value                |
|------------------|---------------------|
| NAV              | ₹{nav}              |
| Expense Ratio    | {expense_ratio}%     |
| 1Y Return        | {returns_1y}%        |
| 3Y Return        | {returns_3y}%        |
| 5Y Return        | {returns_5y}%        |
| Risk Category    | {risk}               |
| Fund Manager     | {manager}            |
| Fund Type        | {fund_type}          |
| Fund Subtype     | {fund_subtype}       |
| Plan             | {plan}               |
| ISIN             | {isin}               |
"""

        # Narrative summary
        summary = f"""
**{scheme_name}** is a {fund_type} ({fund_subtype}) offered by **{amc_name}**.
Current NAV: ₹{nav}, Expense Ratio: {expense_ratio}%, Risk Category: {risk}.
Managed by: {manager}. Returns: 1Y: {returns_1y}%, 3Y: {returns_3y}%, 5Y: {returns_5y}%.
"""

        # Investment philosophy and suitability
        suitability = self._format_who_for(api_data, summary)
        disclaimer = "Note: Past performance is not a guarantee of future results. Please review scheme documents and consult a financial advisor before investing."

        # Optionally add web search highlights if available
        web_data = fund_data.get("web_data", "")
        web_highlights = f"\n**Web Highlights:**\n{web_data}" if web_data else ""

        # Compose final answer
        answer = f"""
### {scheme_name} Overview

{summary}

{metrics_table}

**Who It's For:** {suitability}

{web_highlights}

{disclaimer}
"""
        return answer

    def _format_real_time_data(self, real_time_data: Dict) -> str:
        """
        Format real-time data for inclusion in the prompt
        """
        if not real_time_data:
            return "No real-time data available."
        
        formatted_parts = []
        
        # Format fund NAV data
        if 'fund_nav' in real_time_data and real_time_data['fund_nav']:
            nav_str = "**Live NAV Data:**\n"
            for nav in real_time_data['fund_nav']:
                nav_str += f"- {nav['fund_name']}: ₹{nav['nav']} (as of {nav['date']})\n"
            formatted_parts.append(nav_str)
        
        # Format fund performance data
        if 'fund_performance' in real_time_data and real_time_data['fund_performance']:
            perf_str = "**Fund Performance Data:**\n"
            for perf in real_time_data['fund_performance']:
                perf_str += f"- {perf['fund_name']}: 1Y: {perf.get('1y_return', 'N/A')}, 3Y: {perf.get('3y_return', 'N/A')}, AUM: {perf.get('aum', 'N/A')}\n"
            formatted_parts.append(perf_str)
        
        # Format market indices
        if 'market_indices' in real_time_data:
            indices = real_time_data['market_indices']
            indices_str = "**Market Indices:**\n"
            for index_name, data in indices.items():
                if index_name != 'last_updated':
                    indices_str += f"- {index_name.replace('_', ' ').title()}: {data['value']} ({data['change_percent']})\n"
            formatted_parts.append(indices_str)
        
        # Format sector performance
        if 'sector_performance' in real_time_data:
            sector_str = "**Sector Performance:**\n"
            for sector in real_time_data['sector_performance']:
                sector_str += f"- {sector['sector']}: {sector['performance']}\n"
            formatted_parts.append(sector_str)
        
        # Format economic indicators
        if 'economic_indicators' in real_time_data:
            econ_str = "**Economic Indicators:**\n"
            for indicator, value in real_time_data['economic_indicators'].items():
                if indicator != 'last_updated':
                    econ_str += f"- {indicator.replace('_', ' ').title()}: {value}\n"
            formatted_parts.append(econ_str)
        
        return "\n\n".join(formatted_parts) if formatted_parts else "No real-time data available."

    def _format_final_response(self, structured_response: StructuredResponse, 
                             quality_metrics: ResponseQuality, raw_response: str) -> str:
        """
        Format the final response with structured data ONLY (no quality metrics)
        """
        # Get the formatted structured response
        formatted_structured = structured_generator.format_structured_response(structured_response, raw_response)
        return formatted_structured

    async def _extract_fund_names_with_spacy(self, query: str) -> List[str]:
        """Extracts potential fund names using spaCy's named entity recognition."""
        doc = nlp(query)
        fund_names = set()
        
        # Look for entities that are organizations or products
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                fund_names.add(ent.text)

        # A simple fallback to catch fund names spaCy might miss
        # This is a bit naive, but helps catch patterns like "HDFC [anything] Fund"
        pattern = r'\b(HDFC|ICICI|SBI|Kotak|Nippon)\s[\w\s-]*Fund\b'
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            fund_names.add(match.strip())
            
        # If we found specific fund names, add a general query for the company too
        if "HDFC" in query:
            fund_names.add("HDFC")
        if "ICICI" in query:
            fund_names.add("ICICI Prudential")


        if not fund_names:
            print("No specific fund names extracted, using the full query for search.")
            return [query]
            
        print(f"Extracted fund names: {list(fund_names)}")
        return list(fund_names)

    async def _get_web_data(self, query: str) -> str:
        """Simulate web search for current market data"""
        prompt = f"You are a mutual fund expert with current market knowledge. Answer this question: {query}"
        return await self._call_groq(prompt, is_web_search=True)

    async def _call_groq(self, prompt: str, is_web_search: bool = False) -> str:
        """Wrapper for Groq call"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Add a special note for the simulated web search
                if is_web_search:
                    prompt = "Simulating a web search to answer the following: " + prompt
                
                response = await self.client.generate(prompt)
                
                # Basic validation
                if response and isinstance(response, str) and "error" not in response.lower():
                    return response.strip()
                
                print(f"Groq response malformed: {response}")

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Groq connection error, retrying... ({e})")
                    await asyncio.sleep(2)
                else:
                    print(f"Groq connection failed after {max_retries} attempts: {e}")
                    return ""
        return "" # Should not be reached

    def _fallback_response(self, query: str, factsheet_context: List[str], web_data: str) -> str:
        """Fallback if primary generation fails"""
        response_parts = []
        response_parts.append(f"Based on your query about '{query}':\n")
        
        if web_data:
            response_parts.append("Current market information:")
            response_parts.append(web_data)
        
        if factsheet_context:
            response_parts.append("\nFactsheet data:")
            response_parts.append('\n'.join(factsheet_context))
            
        return '\n'.join(response_parts)

    def _format_kg_response(self, fund_data: dict, query: str) -> str:
        """Format a structured response from knowledge graph data."""
        lines = [f"\U0001F4C8 Fund Name: {fund_data.get('fund_name', 'Unknown')}"]
        if 'fund_manager' in fund_data:
            lines.append(f"- Fund Manager: {fund_data['fund_manager']}")
        if 'aum' in fund_data:
            lines.append(f"- AUM: {fund_data['aum']}")
        if 'inception_date' in fund_data:
            lines.append(f"- Inception Date: {fund_data['inception_date']}")
        if 'expense_ratio' in fund_data:
            lines.append(f"- Expense Ratio: {fund_data['expense_ratio']}")
        if 'returns' in fund_data:
            lines.append(f"- Returns: {fund_data['returns']}")
        if 'category' in fund_data:
            lines.append(f"- Category: {fund_data['category']}")
        if 'risk' in fund_data:
            lines.append(f"- Risk: {fund_data['risk']}")
        lines.append("")
        lines.append("(This answer was generated from the knowledge graph. If you need more details, ask for performance, comparison, or latest news.)")
        return "\n".join(lines)

    def update_knowledge_graph(self, fund_name: str, attributes: dict):
        """Update the knowledge graph with new attributes for a fund."""
        self.knowledge_graph.update_fund(fund_name, attributes)

    def _deduplicate_snippets(self, factsheet_chunks, web_results):
        """Remove duplicate snippets between factsheet and web results."""
        seen = set()
        deduped_factsheet = []
        deduped_web = []
        # Deduplicate by normalized text
        for chunk in factsheet_chunks or []:
            norm = chunk.strip().lower()
            if norm and norm not in seen:
                deduped_factsheet.append(chunk)
                seen.add(norm)
        for r in web_results or []:
            snippet = r.get('snippet') if isinstance(r, dict) else r
            norm = (snippet or '').strip().lower()
            if norm and norm not in seen:
                deduped_web.append(r)
                seen.add(norm)
        return deduped_factsheet, deduped_web

    def _extract_all_attributes(self, factsheet_chunks, web_results):
        """Extract key metrics from factsheet and web snippets, with robust top holdings extraction."""
        ws = WebSearch()
        web_snippets = [r.get('snippet', '') if isinstance(r, dict) else r for r in web_results or []]
        web_attrs = ws.extract_fund_attributes(web_snippets)
        factsheet_text = ' '.join(factsheet_chunks or [])
        attrs = dict(web_attrs)
        patterns = {
            'aum': r'AUM[:\s]+([\d,.]+ ?(Cr|crore|billion|lakh|mn|million)?)',
            'nav': r'NAV[:\s]+([\d,.]+)',
            'returns': r'(\d{1,2}\.\d{1,2}% ?(?:CAGR|return|p.a.))',
            'expense_ratio': r'Expense Ratio[:\s]+([\d.]+%)',
            'risk': r'Risk[:\s]+([A-Za-z ]+)',
            'fund_manager': r'Fund Manager[s]?: ([A-Za-z ,.]+)',
        }
        for key, pat in patterns.items():
            match = re.search(pat, factsheet_text, re.IGNORECASE)
            if match:
                attrs[key] = match.group(1).strip()
        # --- Robust Top Holdings Extraction ---
        holdings = []
        # 1. Look for a Top Holdings section/table
        holdings_section = re.search(r"Top Holdings?:?\s*([\s\S]{0,500})", factsheet_text, re.IGNORECASE)
        if holdings_section:
            section = holdings_section.group(1)
            # Try to extract lines that look like holdings (company + % or just company)
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match lines like '1. Reliance Industries 8.5%' or 'Reliance Industries - 8.5%' or just 'Reliance Industries'
                m = re.match(r"(?:\d+\.\s*)?([A-Za-z0-9 &\-\.]+)(?:\s*[-:]?\s*([\d.]+%)?)?", line)
                if m and m.group(1):
                    holding = m.group(1).strip()
                    if holding and holding.lower() not in [h.lower() for h in holdings]:
                        holdings.append(holding)
                if len(holdings) >= 10:
                    break
        # 2. Fallback: Use NER to extract ORG entities from the section
        if not holdings and holdings_section:
            try:
                doc = nlp(holdings_section.group(1))
                for ent in doc.ents:
                    if ent.label_ == "ORG" and ent.text not in holdings:
                        holdings.append(ent.text)
            except Exception:
                pass
        # 3. Fallback: Use web attributes if available
        if not holdings and web_attrs.get('top_holdings'):
            holdings = [h.strip() for h in re.split(r',|;', web_attrs['top_holdings']) if h.strip()]
        if holdings:
            attrs['top_holdings'] = ', '.join(holdings[:10])
        return attrs

    def _format_key_metrics(self, attrs):
        """Format key metrics as bullet points."""
        lines = []
        for label, key in [
            ("AUM", "aum"), ("NAV", "nav"), ("1Y/3Y/5Y Returns", "returns"),
            ("Expense Ratio", "expense_ratio"), ("Risk", "risk"),
            ("Fund Manager", "fund_manager"), ("Top Holdings", "top_holdings")]:
            if attrs.get(key):
                lines.append(f"- **{label}:** {attrs[key]}")
        return '\n'.join(lines)

    def _compose_narrative_summary(self, fund_name, attrs):
        """Compose a narrative summary sentence."""
        summary = f"{fund_name or 'This fund'}"
        if attrs.get('category'):
            summary += f" is a {attrs['category']}"
        summary += " mutual fund"
        if attrs.get('aum'):
            summary += f" with an AUM of {attrs['aum']}"
        if attrs.get('returns'):
            summary += f" and a recent return of {attrs['returns']}"
        if attrs.get('expense_ratio'):
            summary += f". The expense ratio is {attrs['expense_ratio']}"
        if attrs.get('fund_manager'):
            summary += f", managed by {attrs['fund_manager']}"
        if attrs.get('risk'):
            summary += f". Risk level: {attrs['risk']}"
        summary += "."
        return summary

    def _format_metrics_table(self, attrs):
        """Format key metrics as a markdown table."""
        headers = ["Metric", "Value"]
        rows = []
        for label, key in [
            ("AUM", "aum"), ("NAV", "nav"), ("1Y Return", "1_year_return"), ("3Y Return", "3_year_return"), ("5Y Return", "5_year_return"),
            ("Expense Ratio", "expense_ratio"), ("Risk", "risk"), ("Fund Manager", "fund_manager"), ("Top Holdings", "top_holdings")]:
            if attrs.get(key):
                rows.append(f"| **{label}** | {attrs[key]} |")
        if not rows:
            return ""
        table = f"| {' | '.join(headers)} |\n|{'---|'*len(headers)}\n" + '\n'.join(rows)
        return table

    def _format_bullets(self, items, label):
        if not items:
            return ""
        return f"**{label}:**\n" + '\n'.join([f"- {item}" for item in items])

    def _extract_benefits_and_risks(self, factsheet_chunks, web_results):
        """Extract benefits and risks from all sources (simple heuristics)."""
        text = ' '.join(factsheet_chunks or []) + ' ' + ' '.join([r.get('snippet', r) if isinstance(r, dict) else r for r in web_results or []])
        benefits = []
        risks = []
        # Heuristic: look for sentences with 'benefit', 'advantage', 'pro', 'suitable', 'ideal', 'good for', etc.
        for sent in re.split(r'[.\n]', text):
            s = sent.strip()
            if not s:
                continue
            if any(w in s.lower() for w in ['benefit', 'advantage', 'pro', 'suitable', 'ideal', 'good for', 'best for', 'why invest']):
                benefits.append(s)
            if any(w in s.lower() for w in ['risk', 'con', 'drawback', 'volatility', 'downside', 'not ideal', 'caution', 'tax', 'loss']):
                risks.append(s)
        return benefits[:5], risks[:3]

    def _format_who_for(self, attrs, text):
        # Heuristic: try to guess suitability
        if 'risk' in attrs and 'moderate' in attrs['risk'].lower():
            return "Suitable for investors with a medium-term horizon who are comfortable with some volatility."
        if 'risk' in attrs and 'low' in attrs['risk'].lower():
            return "Ideal for conservative investors seeking stable returns."
        if 'risk' in attrs and 'high' in attrs['risk'].lower():
            return "Best for aggressive investors willing to accept higher risk for higher returns."
        # Fallback
        if 'category' in attrs:
            return f"This fund is suitable for investors looking for {attrs['category']} exposure."
        return "Suitable for investors seeking mutual fund exposure in this category."

    def synthesize_fallback_answer(self, fund_name, factsheet_data, web_results, yahoo_data):
        """Generate a ChatGPT-style, narrative answer from available sources."""
        factsheet_chunks = factsheet_data if isinstance(factsheet_data, list) else [factsheet_data] if factsheet_data else []
        deduped_factsheet, deduped_web = self._deduplicate_snippets(factsheet_chunks, web_results)
        attrs = self._extract_all_attributes(deduped_factsheet, deduped_web)
        # Compose summary
        summary = self._compose_narrative_summary(fund_name, attrs)
        table = self._format_metrics_table(attrs)
        benefits, risks = self._extract_benefits_and_risks(deduped_factsheet, deduped_web)
        who_for = self._format_who_for(attrs, summary)
        answer = f"{summary}\n\n"
        if table:
            answer += f"{table}\n\n"
        if benefits:
            answer += self._format_bullets(benefits, "Key Benefits") + "\n\n"
        if risks:
            answer += self._format_bullets(risks, "Risks / Cons") + "\n\n"
        answer += f"**Who It's For:** {who_for}\n\n"
        answer += "**Final Take:** This fund offers a blend of the above features. Please review the details and consult a financial advisor before investing.\n\n"
        # Add deduped factsheet/web info as highlights
        if deduped_factsheet:
            answer += "**Factsheet Highlights:**\n" + '\n'.join(deduped_factsheet[:2]) + "\n"
        if deduped_web:
            answer += "**Web Highlights:**\n" + '\n'.join([r.get('snippet', r) if isinstance(r, dict) else r for r in deduped_web[:2]]) + "\n"
        answer += "\n_Sources: Factsheet, Web search_"
        return answer

    async def generate_llm_answer(self, query: str, context_chunks: list, web_results: list, yahoo_data: dict) -> str:
        """Generate a ChatGPT-style, narrative answer using all available data."""
        fund_name = None
        if query:
            match = re.search(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
            if match:
                fund_name = match.group(1)
        factsheet_chunks = context_chunks if isinstance(context_chunks, list) else [context_chunks] if context_chunks else []
        deduped_factsheet, deduped_web = self._deduplicate_snippets(factsheet_chunks, web_results)
        attrs = self._extract_all_attributes(deduped_factsheet, deduped_web)
        summary = self._compose_narrative_summary(fund_name, attrs)
        table = self._format_metrics_table(attrs)
        benefits, risks = self._extract_benefits_and_risks(deduped_factsheet, deduped_web)
        who_for = self._format_who_for(attrs, summary)
        # Compose a detailed prompt for the LLM
        prompt = f"""
You are a mutual fund expert. Using the following extracted data, answer the user's question in a detailed, ChatGPT-style, narrative format. Start with a summary, then a markdown table of key metrics, then list key benefits and risks, then a 'Who It's For' section, and end with a 'Final Take'. Use markdown formatting. Cite sources at the end. If any data is missing, say 'Data not available'.

User question: {query}

Extracted attributes:
{json.dumps(attrs, indent=2)}

Factsheet highlights:
{json.dumps(deduped_factsheet[:2], indent=2)}

Web highlights:
{json.dumps([r.get('snippet', r) if isinstance(r, dict) else r for r in deduped_web[:2]], indent=2)}

---
Now generate the answer as described above.
"""
        # If LLM is available, call it; else fallback
        try:
            if self.llm_available():
                # Properly await the async call
                return await self.client.generate(prompt)
            else:
                return self.synthesize_fallback_answer(fund_name, factsheet_chunks, web_results, yahoo_data)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return self.synthesize_fallback_answer(fund_name, factsheet_chunks, web_results, yahoo_data)

    def llm_available(self):
        """Check if the LLM is available."""
        return self.client is not None 

# --- UI/Streamlit-compatible entry point ---
import functools

_chatbot_instance = None

def get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = EnhancedMutualFundChatbot()
    return _chatbot_instance

async def answer_query(query: str, force_web: bool = False, user_id: str = "default") -> dict:
    chatbot = get_chatbot()
    return await chatbot.process_query(query, force_web=force_web, user_id=user_id)

# Test entry point
if __name__ == "__main__":
    import asyncio
    chatbot = EnhancedMutualFundChatbot()
    query = "Who is the fund manager of HDFC Defence Fund?"
    result = asyncio.run(chatbot.process_query(query))
    print("\n===== Chatbot Result =====")
    print(result["formatted_answer"])
    print("\nRaw Answer:", result["full_answer"])
    print("\nQuality Metrics:", result["quality_metrics"])