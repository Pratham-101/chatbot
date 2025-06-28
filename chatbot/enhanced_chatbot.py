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
from chatbot.knowledge_graph import MutualFundKnowledgeGraph
from chatbot.web_search import WebSearch

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
        self.web_search_tool = None 
        self.knowledge_graph = MutualFundKnowledgeGraph()

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
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
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
        print(f"[DEBUG] Entering _perform_web_search for: '{query}'")
        try:
            async def do_search():
                print(f"[DEBUG] Starting DuckDuckGo search for: '{query}'")
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(f"latest performance and details for {query} as of 2025", max_results=3)]
                print(f"[DEBUG] DuckDuckGo search complete for: '{query}'")
                return results
            try:
                results = await asyncio.wait_for(do_search(), timeout=10)
            except asyncio.TimeoutError:
                print(f"[DEBUG] DuckDuckGo search timed out for: '{query}'")
                return "Web search timed out."
            if not results:
                print(f"[DEBUG] No results from DuckDuckGo for: '{query}'")
                return "No relevant information found on the web."
            # Format the results into a single string for the LLM context
            search_summary = "\n\n".join([f"Source: {res['href']}\nSnippet: {res['body']}" for res in results])
            print(f"[DEBUG] Web search successful for: '{query}'")
            return search_summary
        except Exception as e:
            print(f"[DEBUG] Error during web search for '{query}': {e}")
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

    async def process_query(self, query: str) -> str:
        """
        Processes a query by combining factsheet data, web search, and real-time data.
        """
        print(f"[Chatbot] Processing query: '{query}'")
        
        # Step 1: Extract key topics/fund names from the query for targeted searches.
        fund_keywords = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        if not fund_keywords:
            fund_keywords = [query]
        
        print(f"Extracted search keywords: {fund_keywords}")

        # Step 1.5: Try knowledge graph first for direct fund details
        kg_fund_data = None
        for fund in fund_keywords:
            kg_fund_data = self.knowledge_graph.get_fund(fund)
            if kg_fund_data:
                print(f"Found fund in knowledge graph: {fund}")
                break
        if kg_fund_data:
            response = self._format_kg_response(kg_fund_data, query)
            return response

        # Step 2: Try web search attribute extraction and update knowledge graph
        web_search = WebSearch()
        for fund in fund_keywords:
            web_attrs = await web_search.search_and_extract_attributes(fund)
            if web_attrs and len(web_attrs) > 1:
                print(f"Extracted attributes from web for {fund}: {web_attrs}")
                self.knowledge_graph.update_fund(fund, web_attrs)
                response = self._format_kg_response(web_attrs, query)
                return response

        # Step 3: Get data from all sources concurrently
        factsheet_task = asyncio.create_task(self._get_factsheet_context(query))
        web_search_tasks = [self._perform_web_search(keyword) for keyword in fund_keywords]
        real_time_task = asyncio.create_task(self._get_real_time_data(query))
        
        # Gather all results
        factsheet_context, *web_results, real_time_data = await asyncio.gather(
            factsheet_task, *web_search_tasks, real_time_task
        )
        
        web_results_str = "\n\n".join(web_results)

        # Step 4: Build the prompt for the LLM
        factsheet_str = "\n\n".join(factsheet_context) if factsheet_context else "No specific 2025 factsheet data was found in the local documents."
        
        # Format real-time data
        real_time_str = self._format_real_time_data(real_time_data)
        
        prompt = f"""
        You are an expert mutual fund advisor. Your task is to provide a comprehensive answer to the user's query by synthesizing information from three sources: internal documents (2025 factsheets), real-time web search results, and live market data.

        User Query: "{query}"

        ====================
        Source 1: Internal Factsheet Data (Year 2025)
        ---
        {factsheet_str}
        ---
        ====================
        Source 2: Real-Time Web Search Results (Current Data)
        ---
        {web_results_str}
        ---
        ====================
        Source 3: Live Market Data (Real-Time)
        ---
        {real_time_str}
        ---
        ====================

        Instructions:
        1. Synthesize a single, coherent answer from ALL three sources above.
        2. Prioritize real-time data for current NAV, market indices, and live performance.
        3. Use factsheet data for fund details, objectives, and historical context.
        4. Use web search results for latest news, analysis, and market commentary.
        5. Clearly indicate the source and timestamp of real-time data.
        6. If the sources conflict, prioritize real-time data over historical data.
        7. Structure the response with clear headings, bullet points, and tables.
        8. Include relevant market context and economic indicators if applicable.

        Provide a comprehensive and helpful response now.
        """
        
        print("Generating synthesized response...")
        raw_response = await self.client.generate(prompt)
        
        # Step 5: Evaluate response quality
        print("Evaluating response quality...")
        context_for_evaluation = f"Factsheet: {factsheet_str[:500]}... Web: {web_results_str[:500]}... Real-time: {real_time_str[:500]}..."
        quality_metrics = await response_evaluator.evaluate_response(query, context_for_evaluation, raw_response)
        
        # Step 6: Generate structured response
        print("Generating structured response...")
        structured_response = await structured_generator.generate_structured_response(
            query, raw_response, real_time_data
        )
        
        # Step 7: Format final response with quality metrics
        final_response = self._format_final_response(
            structured_response, quality_metrics, raw_response
        )
        
        context_chunks = []
        if factsheet_context:
            context_chunks.append(str(factsheet_context))
        if web_results:
            for r in web_results:
                if isinstance(r, dict):
                    if r.get('snippet'):
                        context_chunks.append(r.get('snippet', ''))
                elif isinstance(r, str):
                    context_chunks.append(r)
        if real_time_data:
            context_chunks.append(str(real_time_data))
        if self.llm_available():
            # Use LLM to generate a full, conversational answer
            if asyncio.iscoroutinefunction(self.generate_llm_answer):
                return await self.generate_llm_answer(query, context_chunks, web_results, real_time_data)
            else:
                return self.generate_llm_answer(query, context_chunks, web_results, real_time_data)
        else:
            # Use fallback synthesis
            return self.synthesize_fallback_answer(fund_name, factsheet_context, web_results, real_time_data)

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
        """Extract key metrics from factsheet and web snippets."""
        ws = WebSearch()
        # From web
        web_snippets = [r.get('snippet', '') if isinstance(r, dict) else r for r in web_results or []]
        web_attrs = ws.extract_fund_attributes(web_snippets)
        # From factsheet (simple regex/heuristics)
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
            'top_holdings': r'Top Holdings?: ([A-Za-z0-9, &]+)',
        }
        for key, pat in patterns.items():
            match = re.search(pat, factsheet_text, re.IGNORECASE)
            if match:
                attrs[key] = match.group(1).strip()
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
            import re
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