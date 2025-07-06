#!/usr/bin/env python3
"""
Comprehensive Finance Chatbot
Integrates multiple data sources for complete financial intelligence:
1. PostgreSQL Database - Real-time fund data and analytics
2. Vector Store - Factsheets and processed documents
3. Web Search - Real-time market data and news
4. Selenium Scraping - Live data from financial websites
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Database and Vector Store
from src.services.data.db_access import get_db_instance, MutualFundDB
from ingestion.vector_store import VectorStore

# Web Search and Scraping
from duckduckgo_search import DDGS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# LLM
from groq import Groq
import os

logger = logging.getLogger(__name__)

class ComprehensiveFinanceChatbot:
    """Advanced chatbot that combines multiple data sources for comprehensive financial intelligence"""
    
    def __init__(self):
        # Initialize data sources
        self.db = get_db_instance()
        self.vector_store = VectorStore()
        self.llm = self._init_llm()
        
        # Intent detection patterns
        self.intent_patterns = {
            'fund_analytics': [
                r'\b(return|performance|cagr|sharpe|beta|alpha|volatility|drawdown|risk)\b',
                r'\b(\d+[- ]?year|yearly|annual)\b',
                r'\b(compare|comparison|vs|versus)\b',
                r'\b(top|best|worst|ranking)\b'
            ],
            'fund_details': [
                r'\b(nav|aum|expense|ratio|manager|benchmark|category|launch|min.*investment)\b',
                r'\b(holdings|portfolio|allocation|sector)\b',
                r'\b(rating|star|grade)\b'
            ],
            'market_data': [
                r'\b(current|live|today|latest|real.*time)\b',
                r'\b(market|price|trading|volume)\b',
                r'\b(gain|loss|change|movement)\b'
            ],
            'general_finance': [
                r'\b(what is|define|explain|how does|elss|equity|debt|hybrid)\b',
                r'\b(tax|sip|lump.*sum|investment|strategy)\b',
                r'\b(amfi|sebi|regulatory|compliance)\b'
            ]
        }
        
        # Fund name extraction patterns
        self.fund_houses = [
            'HDFC', 'Axis', 'Mirae', 'SBI', 'ICICI', 'Kotak', 'Aditya Birla',
            'Tata', 'Nippon', 'Franklin', 'DSP', 'Edelweiss', 'Invesco',
            'PGIM', 'Mahindra', 'Canara', 'Union', 'L&T', 'IDFC', 'Motilal',
            'UTI', 'Reliance', 'Sundaram', 'Principal', 'BNP Paribas'
        ]
    
    def _init_llm(self):
        """Initialize the language model"""
        try:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if groq_api_key:
                return Groq(api_key=groq_api_key)
            else:
                logger.warning("GROQ_API_KEY not found. LLM features will be limited.")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent
        
        return "general_finance"
    
    def extract_fund_names(self, query: str) -> List[str]:
        """Extract fund names from the query using fuzzy search and multiple strategies"""
        # Fuzzy search in the database
        fuzzy_matches = self.db.fuzzy_search_fund_by_name(query, limit=3, score_cutoff=15)  # Lower threshold for partial matches
        if fuzzy_matches:
            # If top match is strong, use it directly
            top_name, top_score = fuzzy_matches[0]
            if top_score >= 50:  # Lower threshold for strong matches
                return [top_name]
            else:
                # If ambiguous, store matches for user clarification
                self.last_fuzzy_matches = fuzzy_matches
                self.ambiguous_query = query
                return []  # Return empty to trigger clarification
        # Fallback to old extraction if no fuzzy match
        fund_names = []
        for house in self.fund_houses:
            if house.lower() in query.lower():
                patterns = [
                    rf'{house}[^.!?]*?(?:Fund|Scheme|Plan)',
                    rf'{house}[^.!?]*?(?:Growth|Direct|Regular)',
                    rf'{house}[^.!?]*?(?:Large Cap|Mid Cap|Small Cap|Flexi Cap)'
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    fund_names.extend(matches)
        fund_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Fund|Scheme|Plan))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Growth|Direct|Regular))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Large Cap|Mid Cap|Small Cap|Flexi Cap))'
        ]
        for pattern in fund_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            fund_names.extend(matches)
        fund_names = list(set(fund_names))
        fund_names = [name.strip() for name in fund_names if len(name.strip()) > 3]
        return fund_names
    
    def get_last_fuzzy_matches(self) -> List[str]:
        """Return the last set of fuzzy matches for user clarification."""
        return getattr(self, 'last_fuzzy_matches', [])
    
    def generate_clarification_prompt(self) -> str:
        """Generate a user-friendly clarification prompt for ambiguous fund matches"""
        if not hasattr(self, 'last_fuzzy_matches') or not self.last_fuzzy_matches:
            return ""
        
        matches = self.last_fuzzy_matches
        fund_options = []
        
        for i, (fund_name, score) in enumerate(matches, 1):
            fund_options.append(f"[{i}] {fund_name} (Match: {score:.1f}%)")
        
        prompt = f"Did you mean one of these funds?\n"
        prompt += "\n".join(fund_options)
        prompt += f"\n\nPlease specify the number (1-{len(matches)}) or provide a more specific fund name."
        
        return prompt
    
    def process_clarification_response(self, user_response: str) -> Tuple[List[str], str]:
        """
        Process user response to clarification prompt
        Returns: (fund_names, error_message)
        """
        if not hasattr(self, 'last_fuzzy_matches') or not self.last_fuzzy_matches:
            return [], "No pending clarification."
        
        matches = self.last_fuzzy_matches
        
        # Check if user provided a number
        try:
            selection = int(user_response.strip())
            if 1 <= selection <= len(matches):
                selected_fund_name = matches[selection - 1][0]
                # Clear the stored matches
                self.last_fuzzy_matches = []
                self.ambiguous_query = None
                return [selected_fund_name], ""
            else:
                return [], f"Please select a number between 1 and {len(matches)}."
        except ValueError:
            # User provided a fund name instead of a number
            # Try to find the best match from the stored matches
            user_fund = user_response.strip()
            best_match = None
            best_score = 0
            
            for fund_name, score in matches:
                if user_fund.lower() in fund_name.lower() or fund_name.lower() in user_fund.lower():
                    if score > best_score:
                        best_match = fund_name
                        best_score = score
            
            if best_match:
                # Clear the stored matches
                self.last_fuzzy_matches = []
                self.ambiguous_query = None
                return [best_match], ""
            else:
                return [], f"Could not match '{user_response}' to any of the suggested funds. Please try again."
    
    async def get_database_data(self, fund_names: List[str], query: str) -> Dict:
        """Get data from PostgreSQL database"""
        data = {}
        
        for fund_name in fund_names:
            try:
                fund_data = self.db.get_fund_details(fund_name=fund_name)
                if fund_data and fund_data.get('fund_info'):
                    data[fund_name] = fund_data
            except Exception as e:
                logger.error(f"Database error for {fund_name}: {e}")
        
        return data
    
    async def get_vector_store_data(self, query: str) -> str:
        """Get relevant data from vector store using embeddings"""
        try:
            # Generate embedding for the query
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query).tolist()
            
            # Query the vector store
            results = self.vector_store.query(query_embedding, k=3, score_threshold=0.3)
            
            if results:
                # Combine relevant documents
                documents = []
                for result in results:
                    if result['score'] > 0.5:  # Only include high-quality matches
                        documents.append(result['text'])
                
                if documents:
                    return "\n\n".join(documents[:2])  # Limit to 2 most relevant docs
                else:
                    return ""
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Vector store error: {e}")
            return ""
    
    async def get_web_search_data(self, query: str) -> List[Dict]:
        """Get real-time data from web search"""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=3)
                return list(results)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            # Return a fallback response when rate limited
            return [{
                'title': 'Mutual Fund Information',
                'body': 'For the most up-to-date information, please visit official mutual fund websites like AMFI, MoneyControl, or Value Research.',
                'link': 'https://www.amfiindia.com'
            }]
    
    async def get_selenium_data(self, fund_names: List[str]) -> Dict:
        """Get live data using Selenium scraping"""
        data = {}
        
        # Skip Selenium for now due to Chrome driver issues
        # In production, you would need to install chromedriver properly
        logger.info("Selenium scraping disabled due to Chrome driver setup requirements")
        return data
    
    def format_database_response(self, db_data: Dict, query: str) -> str:
        """Format database data into a comprehensive response"""
        if not db_data:
            return ""
        
        response_parts = []
        
        for fund_name, fund_data in db_data.items():
            fund_info = fund_data['fund_info']
            latest_nav = fund_data.get('latest_nav')
            ratings = fund_data.get('ratings', [])
            returns = fund_data.get('returns')
            analytics = fund_data.get('analytics')
            
            response_parts.append(f"**{fund_info['scheme_name']}**")
            response_parts.append(f"• **AMC:** {fund_info['amc']}")
            response_parts.append(f"• **Category:** {fund_info['sub_category']}")
            response_parts.append(f"• **Fund Type:** {fund_info['scheme_type']}")
            
            if fund_info.get('fund_manager'):
                response_parts.append(f"• **Fund Manager:** {fund_info['fund_manager']}")
            if fund_info.get('benchmark'):
                response_parts.append(f"• **Benchmark:** {fund_info['benchmark']}")
            if fund_info.get('aum'):
                response_parts.append(f"• **AUM:** ₹{fund_info['aum']:,.2f} Cr")
            if fund_info.get('expense_ratio'):
                response_parts.append(f"• **Expense Ratio:** {fund_info['expense_ratio']:.2f}%")
            if fund_info.get('minimum_lumpsum'):
                response_parts.append(f"• **Min Investment:** ₹{fund_info['minimum_lumpsum']:,.0f}")
            
            if latest_nav:
                response_parts.append(f"• **Latest NAV:** ₹{latest_nav['nav_value']:.4f} (as of {latest_nav['nav_date']})")
            
            if returns:
                response_parts.append("\n**Returns:**")
                if returns.get('return_1y'):
                    response_parts.append(f"• 1 Year: {returns['return_1y']:.2f}%")
                if returns.get('return_3y'):
                    response_parts.append(f"• 3 Years: {returns['return_3y']:.2f}%")
                if returns.get('return_5y'):
                    response_parts.append(f"• 5 Years: {returns['return_5y']:.2f}%")
                if returns.get('return_1m'):
                    response_parts.append(f"• 1 Month: {returns['return_1m']:.2f}%")
                if returns.get('return_3m'):
                    response_parts.append(f"• 3 Months: {returns['return_3m']:.2f}%")
                if returns.get('return_6m'):
                    response_parts.append(f"• 6 Months: {returns['return_6m']:.2f}%")
            
            if analytics:
                response_parts.append("\n**Risk Metrics:**")
                if analytics.get('sharpe_ratio'):
                    response_parts.append(f"• Sharpe Ratio: {analytics['sharpe_ratio']:.2f}")
                if analytics.get('volatility'):
                    response_parts.append(f"• Volatility: {analytics['volatility']:.2f}%")
                if analytics.get('beta'):
                    response_parts.append(f"• Beta: {analytics['beta']:.2f}")
                if analytics.get('alpha'):
                    response_parts.append(f"• Alpha: {analytics['alpha']:.2f}%")
                if analytics.get('max_drawdown'):
                    response_parts.append(f"• Max Drawdown: {analytics['max_drawdown']:.2f}%")
            
            if ratings:
                response_parts.append("\n**Ratings:**")
                for rating in ratings[:3]:
                    response_parts.append(f"• {rating['rating_agency']}: {rating['rating']}")
            
            response_parts.append(f"\n*Source: PostgreSQL Database - Updated daily*")
            response_parts.append("---")
        
        return "\n".join(response_parts)
    
    def format_web_search_response(self, web_data: List[Dict]) -> str:
        """Format web search results"""
        if not web_data:
            return ""
        
        response_parts = ["**Latest Web Information:**"]
        
        for i, result in enumerate(web_data[:3], 1):
            title = result.get('title', 'No title')
            snippet = result.get('body', 'No content')
            link = result.get('link', '')
            
            response_parts.append(f"{i}. **{title}**")
            response_parts.append(f"   {snippet[:200]}...")
            if link:
                response_parts.append(f"   [Read more]({link})")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def format_selenium_response(self, selenium_data: Dict) -> str:
        """Format Selenium scraping results"""
        if not selenium_data:
            return ""
        
        response_parts = ["**Live Market Data:**"]
        
        for fund_name, data in selenium_data.items():
            response_parts.append(f"**{fund_name}**")
            response_parts.append(f"• NAV: {data.get('nav', 'N/A')}")
            response_parts.append(f"• Source: {data.get('source', 'N/A')}")
            response_parts.append(f"• Updated: {data.get('timestamp', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    async def generate_llm_response(self, query: str, context: str) -> str:
        """Generate LLM response with context"""
        if not self.llm:
            return context
        
        try:
            prompt = f"""You are an expert financial advisor specializing in Indian mutual funds and finance. 
Answer the following question using ONLY the provided context. Be comprehensive, accurate, and helpful.

Context:
{context}

Question: {query}

Provide a detailed, well-structured answer with specific data points and insights:"""

            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return context
    
    async def answer_query(self, query: str) -> str:
        """Main method to answer any finance-related query"""
        try:
            # Check if this is a response to a clarification prompt
            if hasattr(self, 'last_fuzzy_matches') and self.last_fuzzy_matches:
                fund_names, error_msg = self.process_clarification_response(query)
                if error_msg:
                    return error_msg
                if not fund_names:
                    return "Please provide a valid selection or fund name."
                
                # Use the original query context but with the clarified fund names
                original_query = getattr(self, 'ambiguous_query', query)
                intent = self.detect_intent(original_query)
            else:
                # Step 1: Detect intent and extract fund names
                intent = self.detect_intent(query)
                fund_names = self.extract_fund_names(query)
                
                logger.info(f"Intent: {intent}, Fund names: {fund_names}")
                
                # Check if we have ambiguous fuzzy matches that need clarification
                if not fund_names and hasattr(self, 'last_fuzzy_matches') and self.last_fuzzy_matches:
                    clarification_prompt = self.generate_clarification_prompt()
                    if clarification_prompt:
                        return clarification_prompt
            
            # Use the appropriate query context
            query_context = getattr(self, 'ambiguous_query', query)
            
            # Step 2: Gather data from all sources
            db_data = await self.get_database_data(fund_names, query_context)
            vector_data = await self.get_vector_store_data(query_context)
            web_data = await self.get_web_search_data(query_context)
            selenium_data = await self.get_selenium_data(fund_names)
            
            # Step 3: Format responses from each source
            responses = []
            
            if db_data:
                db_response = self.format_database_response(db_data, query_context)
                if db_response:
                    responses.append(db_response)
            
            if vector_data:
                responses.append(f"**Factsheet Information:**\n{vector_data}")
            
            if web_data:
                web_response = self.format_web_search_response(web_data)
                if web_response:
                    responses.append(web_response)
            
            if selenium_data:
                selenium_response = self.format_selenium_response(selenium_data)
                if selenium_response:
                    responses.append(selenium_response)
            
            # Step 4: Combine all responses
            if responses:
                combined_response = "\n\n".join(responses)
                
                # Step 5: Use LLM to synthesize if available
                if self.llm:
                    final_response = await self.generate_llm_response(query_context, combined_response)
                else:
                    final_response = combined_response
                
                return final_response
            else:
                return "I couldn't find specific information about that. Please try rephrasing your question or check official sources."
        
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return f"I encountered an error while processing your request. Please try again or rephrase your question."

# Global instance
comprehensive_chatbot = None

def get_comprehensive_chatbot():
    """Get or create the comprehensive chatbot instance"""
    global comprehensive_chatbot
    if comprehensive_chatbot is None:
        comprehensive_chatbot = ComprehensiveFinanceChatbot()
    return comprehensive_chatbot

async def answer_query_async(query: str) -> str:
    """Async wrapper for answering queries"""
    chatbot = get_comprehensive_chatbot()
    return await chatbot.answer_query(query)

def answer_query(query: str) -> str:
    """Synchronous wrapper for answering queries"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(answer_query_async(query))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in synchronous wrapper: {e}")
        return "I encountered an error. Please try again."

if __name__ == "__main__":
    chatbot = get_comprehensive_chatbot()
    print("\n🤖 Comprehensive Finance Chatbot")
    print("Combines Database + Vector Store + Web Search + Selenium")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        answer = answer_query(user_query)
        print(f"Bot: {answer}\n") 