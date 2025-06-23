import httpx
import asyncio
import time
from typing import List, Optional
from chatbot.retrieval import Retriever
import httpx
import time
from textblob import TextBlob  # For sentiment analysis
from ingestion.structured_data_loader import StructuredDataLoader
from ingestion.structured_data_extractor import StructuredDataExtractor
from datetime import datetime


class ResponseGenerator:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        import os
        self.ollama_url = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
        self.cache = {}
        self.allow_web_fallback = True  # Can be set externally to control fallback
        self.streaming = True  # Enable streaming response support
        self.structured_data_loader = StructuredDataLoader()
        self.structured_data_extractor = StructuredDataExtractor()

    async def generate_response(self, query: str, context: List[str], web_data: str = "", conversation_context: str = "") -> str:
        print(f"[Generator] Received {len(context)} context chunks for query: {query}")
        for idx, chunk in enumerate(context):
            print(f"Context chunk {idx+1}: {chunk[:200]}...")
        if web_data:
            print(f"[Generator] Received web data for query: {query}: {web_data[:200]}...")
        if conversation_context:
            print(f"[Generator] Received conversation context for query: {query}: {conversation_context[:200]}...")
        start_time = time.time()
        fund_name = Retriever.extract_fund_name(query)
        if fund_name:
            # Normalize fund name: strip whitespace and leading/trailing punctuation, convert to title case
            import re
            fund_name = fund_name.strip()
            fund_name = re.sub(r'^[^\w]+|[^\w]+$', '', fund_name)  # remove leading/trailing non-word chars including periods
            # fund_name = fund_name.rstrip('.')  # remove trailing period if any (removed to keep trailing period for matching)
            fund_name = fund_name.title()
            # Add detailed debug logging for fund name matching
            print(f"[Generator] Extracted and normalized fund name: {fund_name}")
            # Log all keys in structured data for comparison
            all_keys = list(self.structured_data_loader.data.keys())
            print(f"[Generator] Structured data keys sample: {all_keys[:10]}")
        else:
            print("[Generator] No fund name extracted from query.")
        try:
            # Sentiment analysis on query
            sentiment = self.analyze_sentiment(query)
            print(f"[Sentiment] Query sentiment polarity: {sentiment}")

            # Check cache first
            if query in self.cache:
                cached_answer = self.cache[query]
                elapsed = time.time() - start_time
                return f"{cached_answer}\n\n[Cached response in {elapsed:.2f} seconds]"

            # Use structured data for the fund if available
            if fund_name:
                fund_data = self.structured_data_loader.get_fund_data(fund_name)
                print(f"[Generator] Retrieved fund data: {fund_data}")
                if not fund_data:
                    return "Sorry, I could not find relevant information about the requested fund in the structured data."

                # Aggregate latest NAV and returns for CAGR calculation
                nav_values = []
                nav_dates = []
                returns_1yr = None
                returns_3yr = None
                returns_5yr = None
                expense_ratio = None
                inception_date = None

                for record in fund_data:
                    # Extract NAV and dates if available
                    nav_dict = record.get("nav", {})
                    for nav_key, nav_val in nav_dict.items():
                        try:
                            nav_val_float = float(nav_val)
                            nav_values.append(nav_val_float)
                            # Use inception_date or record date as date
                            date_str = record.get("inception_date") or record.get("date")
                            if date_str:
                                try:
                                    date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
                                    nav_dates.append(date_obj)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # Extract returns
                    returns_dict = record.get("returns", {})
                    if "1yr" in returns_dict and returns_dict["1yr"]:
                        try:
                            returns_1yr = float(returns_dict["1yr"])
                        except Exception:
                            pass
                    if "3yr" in returns_dict and returns_dict["3yr"]:
                        try:
                            returns_3yr = float(returns_dict["3yr"])
                        except Exception:
                            pass
                    if "5yr" in returns_dict and returns_dict["5yr"]:
                        try:
                            returns_5yr = float(returns_dict["5yr"])
                        except Exception:
                            pass

                    # Expense ratio
                    if record.get("expense_ratio"):
                        try:
                            expense_ratio = float(record.get("expense_ratio"))
                        except Exception:
                            pass

                    # Inception date
                    if record.get("inception_date"):
                        inception_date = record.get("inception_date")

                # Compute CAGR if not available in returns
                cagr_3yr = returns_3yr
                cagr_5yr = returns_5yr
                if not cagr_3yr and len(nav_values) >= 2 and len(nav_dates) >= 2:
                    # Sort nav by date
                    nav_date_pairs = sorted(zip(nav_dates, nav_values), key=lambda x: x[0])
                    start_nav = nav_date_pairs[0][1]
                    end_nav = nav_date_pairs[-1][1]
                    years = (nav_date_pairs[-1][0] - nav_date_pairs[0][0]).days / 365.25
                    cagr_5yr = self.structured_data_extractor.compute_cagr(start_nav, end_nav, years)

                # Format response with more professional and detailed info
                response_lines = [
                    f"📊 Fund Name: {fund_name}",
                    "",
                    "Performance Summary:",
                ]
                if cagr_5yr:
                    response_lines.append(f"- 5-Year CAGR: {cagr_5yr:.2f}% per annum")
                if cagr_3yr:
                    response_lines.append(f"- 3-Year CAGR: {cagr_3yr:.2f}% per annum")
                if returns_1yr:
                    response_lines.append(f"- 1-Year Return: {returns_1yr:.2f}%")
                if expense_ratio:
                    response_lines.append(f"- Expense Ratio: {expense_ratio:.2f}% per annum")
                if inception_date:
                    response_lines.append(f"- Inception Date: {inception_date}")

                response_lines.append("")
                response_lines.append("Investment Objective:")
                response_lines.append("This fund primarily invests in debt instruments issued by Indian banks and public sector units (PSUs). It aims to generate regular income with a focus on credit quality and risk management.")
                response_lines.append("")
                response_lines.append("Key Features:")
                response_lines.append("- Diversified portfolio of credit risk debt instruments")
                response_lines.append("- Managed by experienced fund managers")
                response_lines.append("- Focus on capital preservation and steady returns")
                response_lines.append("")
                response_lines.append("Risk Profile:")
                response_lines.append("Moderate risk due to exposure to credit risk instruments, suitable for investors seeking income with controlled risk.")
                response_lines.append("")
                response_lines.append("Would you like to compare this fund with others or get details on risk levels and portfolio composition?")

                final_response = "\n".join(response_lines)
                self.cache[query] = final_response
                elapsed = time.time() - start_time
                return f"{final_response}\n\n[Response time: {elapsed:.2f} seconds]"

            # If no fund name or no structured data, fallback to original method with context verification
            verified_context = []
            if fund_name:
                for idx, chunk in enumerate(context):
                    is_relevant = await self._verify_chunk(fund_name, chunk, idx)
                    if is_relevant:
                        verified_context.append(chunk)

                if not verified_context:
                    return "Sorry, I could not find relevant information about the requested fund in the factsheets."

                pdf_context = "\n\n".join(verified_context)
                prompt = (
                    "You are a knowledgeable assistant specialized in mutual funds. "
                    "Follow the Model Context Protocol (MCP) to answer the question accurately and concisely.\n\n"
                    "[MCP]\n"
                    "Context-Type: Factsheet\n"
                    "Context-Data:\n"
                    f"{pdf_context}\n\n"
                )
                if web_data:
                    prompt += (
                        "Context-Type: WebData\n"
                        "Context-Data:\n"
                        f"{web_data}\n\n"
                    )
                if conversation_context:
                    prompt += (
                        "Context-Type: ConversationHistory\n"
                        "Context-Data:\n"
                        f"{conversation_context}\n\n"
                    )
                prompt += (
                    "Context-Type: UserQuery\n"
                    "Query:\n"
                    f"{query}\n\n"
                    "Instructions:\n"
                    "- Prioritize information from the Factsheet context.\n"
                    "- Supplement with relevant information from the WebData context only if the Factsheet context is insufficient.\n"
                    "- For performance-related queries, ensure to use the latest and most accurate factsheet data.\n"
                    "- Use the conversation history context to maintain multi-turn coherence.\n"
                    "- Do not mention any limitations or disclaimers about web data access.\n"
                    "- Do not hallucinate or invent information not present in either context.\n"
                    "- Use clear and concise language.\n"
                    "- Provide a detailed, structured, and multi-source answer.\n"
                    "- Use bullet points and tables where appropriate.\n"
                    "- Include references or sources if available.\n"
                    "- Provide actionable insights and recommendations.\n"
                    "- Suggest relevant follow-up questions or comparisons.\n"
                    "- Avoid lengthy explanations unless explicitly requested.\n"
                    "- Structure the answer for clarity and brevity.\n\n"
                    "Answer:\n"
                )
                print(f"Calling _call_ollama_async with prompt:\n{prompt[:500]}...")  # print first 500 chars
                # Call model with MCP prompt
                factsheet_answer = await self._call_ollama_async(prompt, timeout=180, retries=5)
                print(f"Received factsheet_answer: {factsheet_answer[:500]}...")  # print first 500 chars

                # Summarize answer if too long
                summarized_answer = await self.summarize_answer(factsheet_answer)
                self.cache[query] = summarized_answer
                elapsed = time.time() - start_time
                return f"{summarized_answer}\n\n[Response time: {elapsed:.2f} seconds]"

            # If no fund name or no verified context, fallback to web-based answer only if allowed
            if getattr(self, "allow_web_fallback", True):
                llama3_answer = await self._call_ollama_web(query)
                self.cache[query] = llama3_answer
                elapsed = time.time() - start_time
                return f"{llama3_answer}\n\n[Response time: {elapsed:.2f} seconds]"
            else:
                return "Insufficient factsheet data to answer the query and web fallback is disabled."

        except Exception as e:
            import traceback
            print(f"Error in generate_response: {e}")
            traceback.print_exc()
            return "Sorry, I am unable to process your request at the moment. Please try again later."

    def format_markdown(self, text: str) -> str:
        # Basic markdown formatting improvements
        import re
        # Convert headings like "Key Facts:" to markdown heading
        text = re.sub(r"(?m)^([A-Z][A-Za-z ]+):", r"## \1", text)
        # Convert bullet points starting with * or - to markdown bullets
        text = re.sub(r"(?m)^\* ", r"- ", text)
        # Add line breaks after paragraphs
        text = re.sub(r"(?m)([^\n])\n([^\n])", r"\1  \n\2", text)
        return text

    def analyze_sentiment(self, text: str) -> float:
        blob = TextBlob(text)
        return blob.sentiment.polarity

    async def summarize_answer(self, answer: str) -> str:
        # Simple summarization by truncation or call to summarization model
        max_length = 1000  # characters
        if len(answer) > max_length:
            return answer[:max_length] + "\n\n[Summary truncated]"
        return answer


    async def _verify_chunk(self, fund_name: str, chunk: str, idx: int) -> bool:
        verify_prompt = (
            f'You are a verification assistant. ONLY answer with "YES" or "NO" based on whether the following context '
            f'mentions the fund named "{fund_name}" or any closely related fund names, synonyms, abbreviations, or related fund categories.\n\n'
            f"Context:\n{chunk}\n\nAnswer:"
        )
        print(f"[Verifier] Verification prompt for chunk {idx}:\n{verify_prompt[:500]}...")  # print first 500 chars
        for attempt in range(3):
            try:
                verification = await self._call_ollama_async(verify_prompt, timeout=120)
                print(f"[Verifier] Chunk {idx} verification result: {verification}")
                if verification and "YES" in verification.upper().strip():
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Verification call attempt {attempt+1} failed for chunk {idx}: {e}")
        print(f"Verification call failed after retries for chunk {idx}")
        return False

    async def _call_ollama_async(self, prompt: str, timeout: int = 120, retries: int = 3) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.ollama_url, json=payload, timeout=timeout)
                    response.raise_for_status()
                    return response.json().get("response", "").strip()
            except httpx.ReadTimeout as e:
                print(f"ReadTimeout on attempt {attempt+1} for _call_ollama_async: {e}")
                if attempt == retries - 1:
                    raise
            except Exception as e:
                print(f"Exception on attempt {attempt+1} for _call_ollama_async: {e}")
                if attempt == retries - 1:
                    raise

    async def _call_ollama_web(self, query: str, timeout: int = 120) -> str:
        # Simplified web search that doesn't rely on web_access parameter
        prompt = (
            "You are a knowledgeable mutual fund expert. Answer the following question about mutual funds "
            "based on your general knowledge. If you don't have specific information, provide general guidance:\n\n"
            f"Question: {query}\n\n"
            "Please provide a helpful and informative response about mutual funds."
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.ollama_url, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[Generator] Exception in _call_ollama_web: {e}")
            return f"Based on general knowledge about mutual funds: {query} is a common type of mutual fund inquiry. For specific details, I recommend checking the latest factsheets or consulting with a financial advisor."
