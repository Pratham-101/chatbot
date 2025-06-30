import asyncio
import json
import time
import re
import os
from typing import List, Dict, Optional
import httpx
from duckduckgo_search import DDGS
from groq import Groq, APIError

from ingestion.vector_store import VectorStore
from chatbot.real_time_data import real_time_provider, market_data_provider
from chatbot.response_quality import response_evaluator, structured_generator, ResponseQuality, StructuredResponse
import spacy

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

class EnhancedMutualFundChatbot:
    """
    A chatbot that answers queries about mutual funds by combining information
    from a local vector store (factsheets), real-time web search, and live market data.
    """
    def __init__(self, model_name="llama3-8b-8192"):
        self.client = GroqClient(model=model_name)
        self.vector_store: Optional[VectorStore] = None
        # This will be populated by the web_search tool call
        self.web_search_tool = None 

    def set_vector_store(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def set_web_search_tool(self, tool):
        self.web_search_tool = tool

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

    async def _perform_web_search(self, query: str) -> str:
        """
        Performs a real-time web search using the duckduckgo_search library.
        """
        print(f"Performing real-time web search for: '{query}'")
        try:
            # We'll take the top 3 results to get a good summary
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(f"latest performance and details for {query} as of 2025", max_results=3)]
            
            if not results:
                return "No relevant information found on the web."

            # Format the results into a single string for the LLM context
            search_summary = "\n\n".join([f"Source: {res['href']}\nSnippet: {res['body']}" for res in results])
            print("Web search successful.")
            return search_summary
        except Exception as e:
            print(f"Error during web search: {e}")
            return "Failed to retrieve information from the web."

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

    async def process_query(self, query: str) -> dict:
        """
        Processes a query by combining factsheet data, web search, and real-time data.
        Returns a dict with both the full LLM answer and the formatted/structured response.
        """
        print(f"[Chatbot] Processing query: '{query}'")
        fund_keywords = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        if not fund_keywords:
            fund_keywords = [query]
        print(f"Extracted search keywords: {fund_keywords}")
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

        **Your Response Guidelines:**

        Write a comprehensive, engaging response that feels like a conversation with a knowledgeable friend who happens to be a mutual fund expert. Here's how to structure your answer:

        1. **Start with a compelling overview** - Give the reader a clear picture of what they're asking about
        2. **Present key metrics in an easy-to-understand format** - Use tables, bullet points, and clear formatting
        3. **Provide detailed analysis** - Explain the "why" behind the numbers, not just the "what"
        4. **Include market context** - Help the reader understand how this fits into the broader market
        5. **Offer actionable insights** - What should the reader consider or do next?
        6. **Be conversational but professional** - Use natural language, avoid jargon unless necessary

        **Key Requirements:**
        - Prioritize real-time data for current NAV, performance, and market conditions
        - Use factsheet data for fund details, objectives, and historical context
        - Incorporate web search results for latest news and market commentary
        - Always cite your sources clearly
        - If data sources conflict, explain the discrepancy and prioritize the most recent information
        - Write in a warm, engaging tone that builds trust
        - Use markdown formatting for clarity (bold headers, bullet points, tables)
        - Include specific numbers, percentages, and dates when available
        - End with a thoughtful conclusion that ties everything together

        **Remember:** You're not just providing data - you're helping someone understand their investment options and make better financial decisions. Be thorough, be clear, and be genuinely helpful.

        Now, provide your comprehensive analysis:
        """
        print("Generating synthesized response...")
        raw_response = await self.client.generate(prompt)
        print("Evaluating response quality...")
        context_for_evaluation = f"Factsheet: {factsheet_str[:500]}... Web: {web_results_str[:500]}... Real-time: {real_time_str[:500]}..."
        quality_metrics = await response_evaluator.evaluate_response(query, context_for_evaluation, raw_response)
        print("Generating structured response...")
        structured_response = await structured_generator.generate_structured_response(
            query, raw_response, real_time_data
        )
        print("Formatting final response...")
        final_response = self._format_final_response(
            structured_response, quality_metrics, raw_response
        )
        return {
            "full_answer": raw_response,
            "formatted_answer": final_response,
            "quality_metrics": quality_metrics,
            "structured_data": structured_response,
            "raw_response": raw_response
        }

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

        **Your Response Guidelines:**

        Write a comprehensive, engaging response that feels like a conversation with a knowledgeable friend who happens to be a mutual fund expert. Here's how to structure your answer:

        1. **Start with a compelling overview** - Give the reader a clear picture of what they're asking about
        2. **Present key metrics in an easy-to-understand format** - Use tables, bullet points, and clear formatting
        3. **Provide detailed analysis** - Explain the "why" behind the numbers, not just the "what"
        4. **Include market context** - Help the reader understand how this fits into the broader market
        5. **Offer actionable insights** - What should the reader consider or do next?
        6. **Be conversational but professional** - Use natural language, avoid jargon unless necessary

        **Key Requirements:**
        - Prioritize real-time data for current NAV, performance, and market conditions
        - Use factsheet data for fund details, objectives, and historical context
        - Incorporate web search results for latest news and market commentary
        - Always cite your sources clearly
        - If data sources conflict, explain the discrepancy and prioritize the most recent information
        - Write in a warm, engaging tone that builds trust
        - Use markdown formatting for clarity (bold headers, bullet points, tables)
        - Include specific numbers, percentages, and dates when available
        - End with a thoughtful conclusion that ties everything together

        **Remember:** You're not just providing data - you're helping someone understand their investment options and make better financial decisions. Be thorough, be clear, and be genuinely helpful.

        Now, provide your comprehensive analysis:
        """
        raw_response = await self.client.generate(prompt)
        return raw_response 