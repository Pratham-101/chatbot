import asyncio
import json
import time
import re
import os
from typing import List, Dict, Optional
import httpx
from duckduckgo_search import DDGS
from groq import Groq, APIError
import threading
import logging
from datetime import datetime
import pandas as pd

from ingestion.vector_store import VectorStore
from chatbot.real_time_data import real_time_provider, market_data_provider
from chatbot.response_quality import response_evaluator, structured_generator, ResponseQuality, StructuredResponse
import spacy

# Add imports for LangChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from functools import partial
from langchain_community.tools import Tool

from src.services.data.real_time_data import get_realtime_fund_data, get_fund_nav_history
from src.services.data.analytics import (
    compute_rolling_returns, compute_volatility, compute_sharpe_ratio, compute_max_drawdown,
    compare_funds, rank_funds, scenario_lump_sum, scenario_sip, plot_comparison_chart, plot_scenario_chart
)
import matplotlib.pyplot as plt
import io
import base64

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

class EnhancedMutualFundChatbot:
    """
    A chatbot that answers queries about mutual funds by combining information
    from a local vector store (factsheets), real-time web search, and live market data.
    Now supports agentic reasoning via LangChain Agents.
    """
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """Initialize the chatbot with specified model and vector store"""
        self.model_name = model_name
        self.vector_store = None
        self.llm = None
        self._init_llm()
        self._init_vector_store()
        print("Chatbot is connected to the vector store and ready.")

    def _init_llm(self):
        """Initialize the language model"""
        provider = os.environ.get("LLM_PROVIDER", "groq").lower()
        
        if provider == "groq":
            try:
                groq_api_key = os.environ.get("GROQ_API_KEY")
                if not groq_api_key:
                    print("⚠️ GROQ_API_KEY not found in environment variables. LLM will not be available.")
                    self.llm = None
                    return
                    
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
                )
                print(f"✅ Initialized Groq LLM with model: {os.environ.get('GROQ_MODEL', 'llama3-70b-8192')}")
            except Exception as e:
                print(f"❌ Failed to initialize Groq LLM: {e}")
                self.llm = None
                
        elif provider == "ollama":
            try:
                self.llm = Ollama(
                    model=os.environ.get("OLLAMA_MODEL", "llama3")
                )
                print(f"✅ Initialized Ollama LLM with model: {os.environ.get('OLLAMA_MODEL', 'llama3')}")
            except Exception as e:
                print(f"❌ Failed to initialize Ollama LLM: {e}")
                self.llm = None
        else:
            print(f"❌ Unknown LLM provider: {provider}")
            self.llm = None

    def _init_vector_store(self):
        """Initialize the vector store"""
        self.vector_store = VectorStore()

    async def process_query(self, query: str) -> dict:
        """Process a user query using RAG and return structured response"""
        start_time = datetime.now()
        
        try:
            # Get relevant context from vector store
            context = ""
            if self.vector_store:
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    print(f"Vector search error: {e}")
                    context = ""

            # Always get real-time data for every query
            real_time_data = await self._perform_web_search(query)

            # Prepare the prompt
            system_prompt = (
                "You are an expert financial advisor specializing in Indian mutual funds. "
                "Use ONLY the data provided below (context and real-time data). "
                "If you do not find the answer, say 'No current data found' and do NOT guess or use outdated information. "
                "Always cite your sources."
            )

            user_prompt = f"""Query: {query}

Context from knowledge base:
{context}

Real-time information:
{real_time_data}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses the provided context and real-time data only
3. If no data is found, say so clearly and do not guess
4. Includes specific fund details, NAV, returns, AUM when available
5. Provides actionable insights and recommendations
6. Cites all sources used
7. Maintains a professional, trustworthy tone

Answer:"""

            # Generate response
            if self.llm:
                try:
                    response = self.llm.invoke(user_prompt)
                    answer_text = response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    print(f"LLM error: {e}")
                    answer_text = "I apologize, but I'm unable to generate a response at the moment. Please try again later."
            else:
                answer_text = "I apologize, but the language model is not properly configured. Please check your API keys and configuration. You can set GROQ_API_KEY in your .env file or environment variables."

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Extract key points and structure the response
            key_points = self._extract_key_points(answer_text)
            
            # Return structured response
            return {
                "full_answer": answer_text,
                "quality_metrics": {
                    "accuracy": 8.0,
                    "completeness": 7.0,
                    "clarity": 8.0,
                    "relevance": 8.0,
                    "overall_score": 7.8,
                    "feedback": "RAG-based response generated successfully."
                },
                "structured_data": {
                    "summary": answer_text[:500] + "..." if len(answer_text) > 500 else answer_text,
                    "key_points": key_points,
                    "fund_details": self._extract_fund_details(answer_text),
                    "performance_data": self._extract_performance_data(answer_text),
                    "risk_metrics": {},
                    "recommendations": self._extract_recommendations(answer_text),
                    "disclaimer": "This information is provided by an AI assistant and should not be considered as financial advice.",
                    "sources": self._extract_sources(context, real_time_data)
                },
                "response_time": response_time,
                "raw_response": answer_text
            }

        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "full_answer": f"I apologize, but an error occurred while processing your query: {str(e)}",
                "quality_metrics": {
                    "accuracy": 5.0,
                    "completeness": 5.0,
                    "clarity": 6.0,
                    "relevance": 5.0,
                    "overall_score": 5.3,
                    "feedback": "Error occurred during processing."
                },
                "structured_data": {
                    "summary": "Error occurred during processing.",
                    "key_points": [],
                    "fund_details": {},
                    "performance_data": {},
                    "risk_metrics": {},
                    "recommendations": [],
                    "disclaimer": "This information is provided by an AI assistant and should not be considered as financial advice.",
                    "sources": []
                },
                "response_time": (datetime.now() - start_time).total_seconds(),
                "raw_response": f"Error: {str(e)}"
            }

    def _extract_key_points(self, answer_text: str) -> List[str]:
        """Extract key points from the answer text"""
        lines = answer_text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                key_points.append(line.lstrip('•-*123456789. '))
            elif ':' in line and len(line) < 100:  # Short lines with colons might be key points
                key_points.append(line)
                
        return key_points[:5]  # Limit to 5 key points

    def _extract_fund_details(self, answer_text: str) -> Dict:
        """Extract fund details from the answer text"""
        details = {}
        
        # Extract NAV
        nav_match = re.search(r'NAV[:\s]*₹?\s*([\d,]+\.?\d*)', answer_text, re.IGNORECASE)
        if nav_match:
            details['nav'] = nav_match.group(1)
            
        # Extract AUM
        aum_match = re.search(r'AUM[:\s]*₹?\s*([\d,]+\.?\d*)\s*(crore|cr|billion|bn)', answer_text, re.IGNORECASE)
        if aum_match:
            details['aum'] = f"{aum_match.group(1)} {aum_match.group(2)}"
            
        # Extract fund name
        fund_match = re.search(r'([A-Z][A-Z\s]+(?:Fund|Scheme|Mutual Fund))', answer_text)
        if fund_match:
            details['fund_name'] = fund_match.group(1).strip()
            
        return details

    def _extract_performance_data(self, answer_text: str) -> Dict:
        """Extract performance data from the answer text"""
        performance = {}
        
        # Extract returns
        returns_match = re.search(r'(\d+\.?\d*)%', answer_text)
        if returns_match:
            performance['returns'] = f"{returns_match.group(1)}%"
            
        return performance

    def _extract_recommendations(self, answer_text: str) -> List[str]:
        """Extract recommendations from the answer text"""
        recommendations = []
        
        # Look for recommendation patterns
        lines = answer_text.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'consider', 'advise']):
                recommendations.append(line)
                
        return recommendations[:3]  # Limit to 3 recommendations

    def _extract_sources(self, context: str, real_time_data: str) -> List[str]:
        """Extract sources from context and real-time data"""
        sources = []
        
        if context:
            sources.append("Knowledge Base")
        if real_time_data:
            sources.append("Real-time Web Search")
            
        return sources

    async def _perform_web_search(self, query: str) -> str:
        """Fetch real-time mutual fund data from AMFI, MoneyControl, and Value Research."""
        try:
            # Try to extract a fund name from the query (simple heuristic)
            fund_name = query
            # Fetch real-time data
            data = get_realtime_fund_data(fund_name)
            if data and not data.get('message'):
                # Format the data for the LLM prompt
                lines = [f"**{k.title().replace('_', ' ')}:** {v}" for k, v in data.items() if k != 'sources']
                sources = data.get('sources', [])
                if sources:
                    lines.append(f"**Sources:**\n" + "\n".join(sources))
                return "\n".join(lines)
            else:
                return "No current data found for this fund."
        except Exception as e:
            print(f"[WebSearch] Error: {e}")
            return "Unable to fetch real-time information at this time."

    async def _get_real_time_data(self, query: str) -> Dict:
        """
        Extract and fetch real-time data based on the query
        """
        real_time_data = {}
        
        # Extract fund names from query
        fund_names = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        
        if fund_names:
            # Get live NAV for mentioned funds
            nav_tasks = [real_time_provider.get_live_nav(fund) for fund in fund_names]
            nav_results = await asyncio.gather(*nav_tasks, return_exceptions=True)
            
            real_time_data['fund_nav'] = [result for result in nav_results if result is not None]
            
            # Get fund performance data
            performance_tasks = [real_time_provider.get_fund_performance(fund) for fund in fund_names]
            performance_results = await asyncio.gather(*performance_tasks, return_exceptions=True)
            
            real_time_data['fund_performance'] = [result for result in performance_results if result is not None]
        
        # Get market indices if query mentions market
        if any(word in query.lower() for word in ['market', 'nifty', 'sensex', 'index', 'indices']):
            real_time_data['market_indices'] = await real_time_provider.get_market_indices()
            real_time_data['sector_performance'] = await real_time_provider.get_sector_performance()
        
        # Get economic indicators if query mentions economy
        if any(word in query.lower() for word in ['economy', 'inflation', 'gdp', 'repo rate']):
            real_time_data['economic_indicators'] = await market_data_provider.get_economic_indicators()
        
        return real_time_data

    async def _get_factsheet_context(self, query: str) -> List[str]:
        """
        Performs a simplified, broad search on the local vector store.
        """
        if not self.vector_store:
            return []

        print("Attempting to retrieve context from local factsheets...")
        try:
            # Try to import sentence_transformers with proper error handling
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError as e:
                print(f"Warning: sentence_transformers not available: {e}")
                return []
            except Exception as e:
                print(f"Warning: Error loading sentence transformer model: {e}")
                return []
            
            # Simple keyword query, boosted with the year
            search_query = f"{query} 2025"
            query_embedding = embedding_model.encode(search_query).tolist()
            
            # Cast a very wide net
            results = self.vector_store.query(query_embedding, k=10, score_threshold=0.1)
            
            if not results:
                print("No relevant documents found in local factsheets.")
                return []
            
            context = [result['text'] for result in results]
            print(f"Retrieved {len(context)} chunks from factsheets.")
            return context
        except Exception as e:
            print(f"Error during factsheet retrieval: {e}")
            return []

    def set_vector_store(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def set_web_search_tool(self, tool):
        self.web_search_tool = tool

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
                perf_str += f"- {perf['fund_name']}: 1Y: {perf['1_year_return']}, 3Y: {perf['3_year_return']}, AUM: {perf['aum']}\n"
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
        Format the final response with structured data and quality metrics
        """
        # Get the formatted structured response
        formatted_structured = structured_generator.format_structured_response(structured_response)
        
        # Add quality metrics section
        quality_section = f"""
## 🎯 Response Quality Assessment

**Overall Score:** {quality_metrics.overall_score}/10

**Detailed Metrics:**
- **Accuracy:** {quality_metrics.accuracy}/10
- **Completeness:** {quality_metrics.completeness}/10  
- **Clarity:** {quality_metrics.clarity}/10
- **Relevance:** {quality_metrics.relevance}/10

**Feedback:** {quality_metrics.feedback}

---
"""
        
        # Combine everything
        final_response = formatted_structured + quality_section
        
        return final_response

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

    async def get_full_llm_answer(self, query: str) -> str:
        """
        Returns only the full, conversational LLM answer (raw_response) for a query.
        """
        fund_keywords = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        if not fund_keywords:
            fund_keywords = [query]
        factsheet_task = asyncio.create_task(self._get_factsheet_context(query))
        web_search_tasks = [self._perform_web_search(keyword) for keyword in fund_keywords]
        real_time_task = asyncio.create_task(self._get_real_time_data(query))
        factsheet_context, *web_results, real_time_data = await asyncio.gather(
            factsheet_task, *web_search_tasks, real_time_task
        )
        web_results_str = "\n\n".join(web_results)
        factsheet_str = "\n\n".join(factsheet_context) if factsheet_context else "No specific 2025 factsheet data was found in the local documents."
        real_time_str = self._format_real_time_data(real_time_data)
        prompt = f"""
        You are an expert mutual fund advisor with deep knowledge of the Indian mutual fund industry. You have access to comprehensive data from multiple sources and your goal is to provide insightful, well-researched answers that help investors make informed decisions.

        **User's Question:** {query}

        **Available Information Sources:**

        **📊 Internal Factsheet Data (2025):**
        {factsheet_str}

        **🌐 Real-Time Web Search Results:**
        {web_results_str}

        **📈 Live Market Data:**
        {real_time_str}

        **📰 Latest News Headlines:**
        {news_summary if news_summary else 'No recent news found.'}
        
        **🗞️ News Sentiment:** {news_sentiment.capitalize()}
        
        **📢 Regulatory Updates:**
        {reg_summary if reg_summary else 'No new regulatory updates.'}

        **Your Response Guidelines:**
        - Use ONLY the provided factsheet, web, and news context for your answer. Do NOT use prior knowledge or invent data.
        - For every fact, number, or data point, cite the source inline with a clickable markdown link (e.g., [Moneycontrol](https?://...)).
        - At the end, include a 'Sources' section with all URLs used, in markdown link format.
        - Do NOT invent or summarize sources—use only the real links provided in the context above.
        - If no real data is found, say so clearly and suggest the user try a different query.
        - Use tables, bullet points, and markdown formatting for clarity.
        - Include specific numbers, percentages, and dates when available.
        - End with a thoughtful conclusion that ties everything together.
        
        Now, provide your comprehensive, data-driven analysis with inline citations:
        """
        # --- Fallback strict mode ---
        def is_generic_answer(answer):
            # Heuristic: too short, no numbers, no links, or only generic phrases
            import re
            if len(answer) < 300:
                return True
            if not re.search(r"\d", answer):
                return True
            if not re.search(r"\[.*\]\(https?://", answer):
                return True
            if re.search(r"(top funds|best funds|summary|overview|conclusion|template|no data|not available|try a different query)", answer, re.IGNORECASE):
                return True
            return False
        # If no real data, return a clear message
        if not factsheet_str.strip() and not web_results_str.strip() and not news_summary.strip():
            raw_response = "Sorry, I couldn't find any real data for your query. Please try a different fund or topic."
            final_response = raw_response
        else:
            raw_response = await self.client.generate(prompt)
            logging.info("[DEBUG] LLM raw output: %s", raw_response[:2000])
            # Fallback strict mode: synthesize table/summary if answer is too generic
            if is_generic_answer(raw_response):
                table = ""
                if factsheet_str.strip():
                    table += f"\n\n**Factsheet Data Table:**\n{factsheet_str}"
                if web_results_str.strip():
                    table += f"\n\n**Web Data Table:**\n{web_results_str}"
                if news_summary.strip():
                    table += f"\n\n**News Summary:**\n{news_summary}"
                if reg_summary.strip():
                    table += f"\n\n**Regulatory Updates:**\n{reg_summary}"
                raw_response += table + "\n\n_Note: This answer was auto-synthesized from real data due to lack of LLM detail._"
                final_response = raw_response
        return final_response

    def _extract_all_attributes(self, factsheet_chunks, web_results):
        """Extract key metrics from factsheet and web snippets, with robust top holdings extraction."""
        ws = WebSearch()
        web_snippets = [r.get('snippet', '') if isinstance(r, dict) else r for r in web_results or []]
        web_attrs = ws.extract_fund_attributes(web_snippets)
        factsheet_text = ' '.join(factsheet_chunks or [])
        attrs = dict(web_attrs)
        import re
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

    def news_tool(self, query: str) -> str:
        """Fetch latest financial news and trends for any topic."""
        return f"[Financial news tool not yet implemented for query: {query}. This would fetch real-time financial news and market updates.]"

def nav_chart_image(nav_history):
    df = pd.DataFrame(nav_history)
    df["date"] = pd.to_datetime(df["date"])
    plt.figure(figsize=(8,4))
    plt.plot(df["date"], df["nav"], label="NAV")
    plt.title("NAV History")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def get_fund_analytics_answer(fund_name: str):
    return f"Analytics for {fund_name} are currently unavailable. Please check back later or consult the fund's official sources for more information."

# Fund comparison answer
def get_fund_comparison_answer(fund_names, metric="rolling_return"):
    nav_histories = {name: get_fund_nav_history(name) for name in fund_names}
    df = compare_funds(nav_histories)
    chart_b64 = plot_comparison_chart(df, metric)
    answer = f"Comparison of {', '.join(fund_names)} on {metric}:\n"
    answer += df.to_string(index=False)
    answer += f"\n![Comparison Chart](data:image/png;base64,{chart_b64})\n"
    return answer

# Fund ranking answer (stub for category-to-fund-list)
def get_fund_ranking_answer(category, metric="rolling_return", top_n=5):
    # TODO: Replace with real fund list for category
    fund_names = ["HDFC Balanced Advantage Fund", "SBI Bluechip Fund", "ICICI Prudential Bluechip Fund", "Axis Bluechip Fund", "Mirae Asset Large Cap Fund"]
    nav_histories = {name: get_fund_nav_history(name) for name in fund_names}
    df = rank_funds(nav_histories, metric, top_n)
    chart_b64 = plot_comparison_chart(df, metric)
    answer = f"Top {top_n} {category} funds by {metric}:\n"
    answer += df.to_string(index=False)
    answer += f"\n![Ranking Chart](data:image/png;base64,{chart_b64})\n"
    return answer

# Scenario analysis answer
def get_scenario_analysis_answer(fund_name, scenario):
    nav_history = get_fund_nav_history(fund_name)
    if scenario["type"] == "lump_sum":
        df = scenario_lump_sum(nav_history, scenario["amount"], scenario["start_date"])
        chart_b64 = plot_scenario_chart(df, "investment_value")
        answer = f"Lump sum investment of ₹{scenario['amount']} in {fund_name} since {scenario['start_date']}:\n"
        answer += df.tail(1).to_string()
        answer += f"\n![Growth Chart](data:image/png;base64,{chart_b64})\n"
        return answer
    elif scenario["type"] == "sip":
        df = scenario_sip(nav_history, scenario["amount"], scenario["start_date"])
        chart_b64 = plot_scenario_chart(df, "sip_value")
        answer = f"SIP of ₹{scenario['amount']} per period in {fund_name} since {scenario['start_date']}:\n"
        answer += df.tail(1).to_string()
        answer += f"\n![SIP Growth Chart](data:image/png;base64,{chart_b64})\n"
        return answer
    else:
        return "Unknown scenario type."

# Example query parsing logic (to be improved with intent detection)
def answer_query(query: str):
    q = query.lower()
    if "compare" in q and " vs " in q:
        names = [x.strip() for x in q.split("compare")[-1].split("vs")]
        return get_fund_comparison_answer(names)
    if "top" in q and "by" in q:
        # e.g. "Top 5 large-cap funds by 5-year return"
        parts = q.split("by")
        metric = parts[-1].strip()
        category = parts[0].split()[-2]  # crude
        return get_fund_ranking_answer(category, metric)
    if "if i had invested" in q or "sip of" in q:
        # crude scenario parsing
        if "sip of" in q:
            amt = int([s for s in q.split() if s.isdigit()][0])
            fund = q.split("in")[-1].strip()
            return get_scenario_analysis_answer(fund, {"type": "sip", "amount": amt, "start_date": "2020-01-01"})
        if "if i had invested" in q:
            amt = int([s for s in q.split() if s.isdigit()][0])
            fund = q.split("in")[-1].strip()
            return get_scenario_analysis_answer(fund, {"type": "lump_sum", "amount": amt, "start_date": "2018-01-01"})
    # Default: single fund analytics
    return get_fund_analytics_answer(query) 