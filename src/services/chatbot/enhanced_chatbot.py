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
import datetime
import logging

from ingestion.vector_store import VectorStore
from chatbot.real_time_data import real_time_provider, market_data_provider
from chatbot.response_quality import response_evaluator, structured_generator, ResponseQuality, StructuredResponse
import spacy
from chatbot.knowledge_graph import MutualFundKnowledgeGraph
from chatbot.web_search import WebSearch
from ingestion.structured_data_loader import StructuredDataLoader
from pipeline import MutualFundPipeline

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

    async def process_query(self, query: str, user_id: str = "default") -> dict:
        """
        Processes a query by combining factsheet data, web search, and real-time data.
        Loads and updates user session context in user_sessions.json.
        Returns a dict with both the full LLM answer and the formatted/structured response.
        Adds proactive market alerts if significant NAV/news changes are detected.
        Adds a dynamic follow-up suggestion based on the answer and context.
        """
        print(f"[Chatbot] Processing query: '{query}' for user {user_id}")
        session = load_user_session(user_id)
        last_fund = session.get("last_fund")
        fund_keywords = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        if not fund_keywords and last_fund:
            fund_keywords = [last_fund]
        elif not fund_keywords:
            fund_keywords = [query]
        print(f"Extracted search keywords: {fund_keywords}")
        # Update session with last fund
        if fund_keywords:
            session["last_fund"] = fund_keywords[0]
        # Store question history
        history = session.get("history", [])
        history.append({
            "question": query,
            "timestamp": datetime.datetime.now().isoformat()
        })
        session["history"] = history
        # Structured data lookup for each fund
        structured_facts = {}
        for fund in fund_keywords:
            fund_data = self.structured_data_loader.get_fund_data(fund)
            if fund_data:
                structured_facts[fund] = fund_data
        # Try to answer directly from structured data for key questions
        direct_answer = None
        for fund, records in structured_facts.items():
            # Fund manager: find all managers with the most recent 'since' date
            if "fund manager" in query.lower():
                all_managers = []
                for r in records:
                    if isinstance(r.get("fund_manager"), list):
                        all_managers.extend(r["fund_manager"])
                # Parse dates and find the most recent
                date_manager_map = {}
                for mgr in all_managers:
                    since_str = mgr.get("since")
                    try:
                        since_date = datetime.datetime.strptime(since_str, "%B %d, %Y").date()
                    except Exception:
                        try:
                            since_date = datetime.datetime.strptime(since_str, "%d-%m-%Y").date()
                        except Exception:
                            since_date = since_str  # fallback to string
                    date_manager_map.setdefault(since_date, []).append(mgr)
                # Find the most recent date (if any)
                if date_manager_map:
                    latest_date = max([d for d in date_manager_map if isinstance(d, datetime.date)], default=None)
                    if latest_date:
                        managers = date_manager_map[latest_date]
                        names = ", ".join([m["name"] for m in managers if m["name"]])
                        direct_answer = f"The current fund manager(s) for {fund} as of {latest_date.strftime('%B %d, %Y')} are: {names}."
                    else:
                        # fallback: just list all names
                        names = ", ".join([m["name"] for m in all_managers if m["name"]])
                        direct_answer = f"The fund manager(s) for {fund} are: {names}."
                break
            elif "aum" in query.lower():
                for r in records:
                    if r.get("aum"):
                        direct_answer = f"The AUM of {fund} is {r['aum']} (from factsheet)."
                        break
            elif "nav" in query.lower():
                for r in records:
                    if r.get("nav"):
                        direct_answer = f"The latest NAV of {fund} is {r['nav']} (from factsheet)."
                        break
            # Add more direct lookups as needed
            if direct_answer:
                break
        if direct_answer:
            return {
                "full_answer": direct_answer,
                "quality_metrics": {"accuracy": 10, "completeness": 10, "clarity": 10, "relevance": 10, "feedback": "Answered directly from structured factsheet data."},
                "structured_data": structured_facts
            }
        # --- Proactive Market Alert Logic ---
        market_alerts = []
        tracked_funds = session.get("tracked_funds", fund_keywords)
        last_navs = session.get("last_navs", {})
        # --- Use pipeline for retrieval instead of self._get_factsheet_context ---
        factsheet_context = await asyncio.to_thread(self.pipeline.retrieve, query, 5)
        web_search_tasks = [self._perform_web_search(keyword) for keyword in fund_keywords]
        real_time_task = asyncio.create_task(self._get_real_time_data(query))
        factsheet_context, *web_results, real_time_data = await asyncio.gather(
            asyncio.create_task(asyncio.to_thread(self.pipeline.retrieve, query, 5)),
            *web_search_tasks,
            real_time_task
        )
        # Check NAV changes
        if real_time_data.get('fund_nav'):
            for nav_info in real_time_data['fund_nav']:
                fund = nav_info['fund_name']
                nav = nav_info['nav']
                prev_nav = last_navs.get(fund)
                if prev_nav:
                    try:
                        nav_float = float(str(nav).replace(',', ''))
                        prev_nav_float = float(str(prev_nav).replace(',', ''))
                        if prev_nav_float > 0:
                            change_pct = 100 * (nav_float - prev_nav_float) / prev_nav_float
                            if abs(change_pct) >= 5:
                                alert = f"⚠️ NAV Alert: {fund} NAV changed by {change_pct:.2f}% since your last check. (Prev: ₹{prev_nav}, Now: ₹{nav})"
                                market_alerts.append(alert)
                    except Exception:
                        pass
                last_navs[fund] = nav
        session["last_navs"] = last_navs
        last_news = session.get("last_news", "")
        latest_news = ""
        if real_time_data.get('market_indices') and 'last_updated' in real_time_data['market_indices']:
            latest_news = real_time_data['market_indices']['last_updated']
        if real_time_data.get('fund_performance') and len(real_time_data['fund_performance']) > 0:
            latest_news = real_time_data['fund_performance'][0].get('last_updated', latest_news)
        if latest_news and latest_news != last_news:
            alert = f"📰 Market Update: New market data available as of {latest_news}."
            market_alerts.append(alert)
            session["last_news"] = latest_news
        save_user_session(user_id, session)
        # --- Advanced Real-Time News, Sentiment, and Regulatory Updates ---
        # 1. Fetch latest fund news and analyze sentiment
        fund_news = []
        news_sentiment = 'neutral'
        if fund_keywords:
            fund_news = await real_time_provider.get_fund_news(fund_keywords[0])
            news_headlines = [n['headline'] for n in fund_news]
            news_sentiment = real_time_provider.analyze_sentiment(news_headlines)
        # 2. Fetch latest regulatory updates
        regulatory_updates = await market_data_provider.get_regulatory_updates()
        last_reg_update_time = session.get('last_reg_update_time', '')
        new_reg_alerts = []
        latest_reg_time = last_reg_update_time
        for update in regulatory_updates:
            ts = update.get('timestamp') or update.get('published', '')
            if ts and ts > last_reg_update_time:
                new_reg_alerts.append(update)
                if not latest_reg_time or ts > latest_reg_time:
                    latest_reg_time = ts
        if latest_reg_time:
            session['last_reg_update_time'] = latest_reg_time
        # Add regulatory alerts to market_alerts
        for alert in new_reg_alerts:
            market_alerts.append(f"📢 Regulatory Update: {alert.get('title')} ({alert.get('link')})")
        # Save session after update (again, to persist reg update time)
        save_user_session(user_id, session)
        # 3. Summarize news for answer (use LLM if available, else simple join)
        news_summary = ''
        if fund_news:
            if self.llm_available():
                news_text = '\n'.join([f"- {n['headline']}: {n['summary']}" for n in fund_news])
                news_prompt = f"Summarize the following latest news for {fund_keywords[0]} in 2-3 sentences for an investor:\n{news_text}"
                news_summary = await self.client.generate(news_prompt)
            else:
                news_summary = '\n'.join([f"- {n['headline']}: {n['summary']}" for n in fund_news])
        # 4. Summarize regulatory updates (use LLM if available, else simple join)
        reg_summary = ''
        if new_reg_alerts:
            if self.llm_available():
                reg_text = '\n'.join([f"- {r['title']}: {r['summary']}" for r in new_reg_alerts])
                reg_prompt = f"Summarize the following new regulatory updates for mutual fund investors in 1-2 sentences:\n{reg_text}"
                reg_summary = await self.client.generate(reg_prompt)
            else:
                reg_summary = '\n'.join([f"- {r['title']}: {r['summary']}" for r in new_reg_alerts])
        # --- End Advanced Real-Time News, Sentiment, and Regulatory Updates ---
        # --- Advanced Top Funds Table Synthesis (no hardcoding) ---
        import re
        def extract_fund_rows(contexts):
            fund_rows = []
            seen = set()
            # Try to extract rows like: Fund Name, Return, AUM, Category, Link
            fund_pattern = re.compile(r"([A-Za-z0-9 &\-\.]+?)(?: Fund| Scheme| Plan)?[\s\-:|]+([\d.]+%)[\s\-:|]+([A-Za-z]+)?[\s\-:|]+([\d,]+ ?[Cc]r|[\d,]+ ?[Mm]n|[\d,]+ ?[Ll]akh)?", re.IGNORECASE)
            link_pattern = re.compile(r"https?://[\w./\-_%?=&]+")
            for chunk in contexts:
                # Extract links
                links = link_pattern.findall(chunk)
                # Extract fund rows
                for match in fund_pattern.finditer(chunk):
                    name, ret, cat, aum = match.groups()
                    key = (name.strip(), ret.strip(), aum.strip() if aum else '', cat.strip() if cat else '')
                    if key not in seen:
                        fund_rows.append({
                            'name': name.strip(),
                            'return': ret.strip(),
                            'category': cat.strip() if cat else '',
                            'aum': aum.strip() if aum else '',
                            'link': links[0] if links else ''
                        })
                        seen.add(key)
            return fund_rows
        # Gather all context for parsing
        all_context = []
        if factsheet_context: all_context.append(factsheet_context)
        if web_results: all_context.append(web_results)
        if news_summary: all_context.append(news_summary)
        # Extract fund rows
        fund_rows = extract_fund_rows(all_context)
        logging.info("[DEBUG] Parsed fund rows: %s", fund_rows)
        # Synthesize markdown table if enough rows
        def synthesize_fund_table(rows):
            if not rows:
                return ''
            table = "| Fund Name | Return | Category | AUM | Source Link |\n|---|---|---|---|---|\n"
            for r in rows:
                link = f"[{r['name']}]({r['link']})" if r['link'] else r['name']
                table += f"| {link} | {r['return']} | {r['category']} | {r['aum']} | {r['link']} |\n"
            return table
        fund_table = synthesize_fund_table(fund_rows)
        # --- End Advanced Top Funds Table Synthesis ---
        # --- BEGIN: General, Modular, ChatGPT-like Prompt Engineering ---
        factsheet_str = "\n\n".join(factsheet_context) if factsheet_context else "No specific 2025 factsheet data was found in the local documents."
        web_results_str = "\n\n".join([r if isinstance(r, str) else json.dumps(r) for r in web_results]) if web_results else "No relevant web results found."
        real_time_str = self._format_real_time_data(real_time_data)

        prompt = f'''
You are an expert financial advisor. Using ONLY the provided context, answer the user's question in a detailed, actionable, and user-friendly way.

User's Question: {query}

Context:
{factsheet_str}
{web_results_str}
{real_time_str}

Instructions:
- Use all available data to answer the question as completely as possible.
- If the question asks for a list, comparison, or ranking, present the data in a markdown table if possible.
- If the question is about performance, risk, or returns, provide numbers, trends, and cite sources inline (e.g., [Moneycontrol](...)).
- If the question is about recommendations, provide actionable advice and highlight risks or considerations.
- If data is missing, say so, but still provide as much as possible (e.g., "Based on the latest available data, here's what we know...").
- Use markdown formatting, bullet points, and clear language.
- Use chain-of-thought reasoning: break down your answer step by step, and synthesize across all sources.
- If the context contains partial or conflicting data, explain the limitations and provide the best possible synthesis.
- Always be transparent about the sources and limitations of the data.
'''
        # --- END: General, Modular, ChatGPT-like Prompt Engineering ---

        # --- LLM answer generation ---
        raw_response = await self.client.generate(prompt)
        logging.info("[DEBUG] LLM raw output: %s", raw_response[:2000])
        # Fallback strict mode: synthesize table/summary if answer is too generic or missing a table for 'top funds' queries
        is_top_funds_query = any(kw in query.lower() for kw in ["top funds", "best funds", "top 10", "top ten", "top performers"])
        if is_generic_answer(raw_response) or (is_top_funds_query and len(fund_rows) >= 2 and '| Fund Name |' not in raw_response):
            table = fund_table if fund_table else ''
            if factsheet_str.strip():
                table += f"\n\n**Factsheet Data Table:**\n{factsheet_str}"
            if web_results_str.strip():
                table += f"\n\n**Web Data Table:**\n{web_results_str}"
            if news_summary.strip():
                table += f"\n\n**News Summary:**\n{news_summary}"
            if reg_summary.strip():
                table += f"\n\n**Regulatory Updates:**\n{reg_summary}"
            raw_response += table + "\n\n_Note: This answer was auto-synthesized from real data due to lack of LLM detail or table._"
            final_response = raw_response
        else:
            final_response = raw_response
        # --- End LLM answer generation ---
        print("Evaluating response quality...")
        # --- Use pipeline for evaluation instead of old evaluator ---
        quality_metrics = self.pipeline.evaluate(raw_response, query)
        print("Generating structured response...")
        structured_response = await structured_generator.generate_structured_response(
            query, raw_response, real_time_data
        )
        print("Formatting final response...")
        final_response = self._format_final_response(
            structured_response, quality_metrics, raw_response
        )
        # --- Dynamic Follow-Up Suggestion ---
        follow_up_prompt = f"Given the user's question: '{query}' and the following answer: '{raw_response}', suggest a highly relevant, concise follow-up question or next step the user might want to ask. Respond with only the follow-up suggestion."
        follow_up_suggestion = await self.client.generate(follow_up_prompt)

        # --- Advanced Source Link Post-Processing ---
        # Collect all unique (title, href) pairs from web_results
        sources_links = []
        seen_links = set()
        for r in web_results:
            if isinstance(r, dict) and r.get('href'):
                title = r.get('title', 'Source')
                href = r['href']
                key = (title.strip(), href.strip())
                if href and key not in seen_links:
                    sources_links.append(f"- [{title}]({href})")
                    seen_links.add(key)
        # Always append sources section, even if LLM already outputs one
        if sources_links:
            sources_section = "\n\n**Sources:**\n" + "\n".join(sources_links)
            import re
            # Remove any existing 'Sources' section (case-insensitive, markdown or plain)
            raw_response = re.sub(r"\*\*Sources\*\*:(.|\n)*", '', raw_response, flags=re.IGNORECASE)
            raw_response = re.sub(r"Sources:(.|\n)*", '', raw_response, flags=re.IGNORECASE)
            final_response = re.sub(r"\*\*Sources\*\*:(.|\n)*", '', final_response, flags=re.IGNORECASE)
            final_response = re.sub(r"Sources:(.|\n)*", '', final_response, flags=re.IGNORECASE)
            raw_response = raw_response.strip() + sources_section
            final_response = final_response.strip() + sources_section
        # Also add to structured_data for UI
        if structured_response and hasattr(structured_response, 'sources'):
            structured_response.sources = sources_links
        elif isinstance(structured_response, dict):
            structured_response['sources'] = sources_links
        # Add news, sentiment, and regulatory info to returned dict for UI
        return {
            "full_answer": raw_response,
            "formatted_answer": final_response,
            "quality_metrics": quality_metrics,
            "structured_data": structured_response,
            "raw_response": raw_response,
            "market_alerts": market_alerts,
            "follow_up_suggestion": follow_up_suggestion.strip() if follow_up_suggestion else None,
            "news": fund_news,
            "news_sentiment": news_sentiment,
            "regulatory_updates": new_reg_alerts
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