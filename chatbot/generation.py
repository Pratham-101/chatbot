import httpx
import asyncio
import time
from typing import List, Optional
from chatbot.retrieval import Retriever
import httpx
import time
from textblob import TextBlob  # For sentiment analysis


class ResponseGenerator:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.cache = {}
        self.allow_web_fallback = True  # Can be set externally to control fallback

    async def generate_response(self, query: str, context: List[str], web_data: str = "") -> str:
        print(f"[Generator] Received {len(context)} context chunks for query: {query}")
        for idx, chunk in enumerate(context):
            print(f"Context chunk {idx+1}: {chunk[:200]}...")
        if web_data:
            print(f"[Generator] Received web data for query: {query}: {web_data[:200]}...")
        start_time = time.time()
        fund_name = Retriever.extract_fund_name(query)
        verified_context = []

        try:
            # Sentiment analysis on query
            sentiment = self.analyze_sentiment(query)
            print(f"[Sentiment] Query sentiment polarity: {sentiment}")

            # Check cache first
            if query in self.cache:
                cached_answer = self.cache[query]
                elapsed = time.time() - start_time
                return f"{cached_answer}\n\n[Cached response in {elapsed:.2f} seconds]"

            # Verify context chunks to keep only relevant ones
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
                    "Follow the Model Context Protocol (MCP) to answer the question accurately and in detail.\n\n"
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
                prompt += (
                    "Context-Type: UserQuery\n"
                    "Query:\n"
                    f"{query}\n\n"
                    "Instructions:\n"
                    "- Prioritize information from the Factsheet context.\n"
                    "- Supplement with relevant information from the WebData context.\n"
                    "- Do not hallucinate or invent information not present in either context.\n"
                    "- If the Factsheet does not contain sufficient information, use WebData to provide a comprehensive answer.\n"
                    "- Use clear and concise language.\n"
                    "- Use bullet points to list key facts.\n"
                    "- Provide examples or comparisons if relevant.\n"
                    "- Summarize performance metrics clearly.\n"
                    "- Structure the answer with headings and bullet points for clarity.\n"
                    "- Provide detailed explanations and insights where applicable.\n"
                    "- Use numbered lists for step-by-step instructions or processes.\n"
                    "- Include relevant definitions or explanations of technical terms.\n"
                    "- Ensure the answer is comprehensive and covers multiple aspects of the query.\n\n"
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
        prompt = (
            "You are a helpful assistant with access to the web. Answer the following question accurately:\n\n"
            f"Question: {query}\n"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "web_access": True,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.ollama_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
