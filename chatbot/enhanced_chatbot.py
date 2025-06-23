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
    from a local vector store (factsheets) and a real-time web search.
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

    async def process_query(self, query: str) -> str:
        """
        Processes a query by first extracting key fund names, running a web search for each,
        then synthesizing a response.
        """
        print(f"[Chatbot] Processing query: '{query}'")
        
        # Step 1: Extract key topics/fund names from the query for targeted searches.
        # A simple keyword approach is more robust than complex NLP for this task.
        fund_keywords = re.findall(r'(HDFC.*?Fund|ICICI.*?Fund|SBI.*?Fund|Kotak.*?Fund|Nippon.*?Fund)', query, re.IGNORECASE)
        if not fund_keywords:
            # Fallback to the whole query if no specific funds are found
            fund_keywords = [query]
        
        print(f"Extracted search keywords: {fund_keywords}")

        # Step 2: Get context from both sources concurrently for each keyword
        factsheet_task = asyncio.create_task(self._get_factsheet_context(query)) # One broad search for factsheets
        
        # Multiple, targeted web searches
        web_search_tasks = [self._perform_web_search(keyword) for keyword in fund_keywords]
        
        # Gather all results
        factsheet_context, *web_results = await asyncio.gather(factsheet_task, *web_search_tasks)
        
        web_results_str = "\n\n".join(web_results)

        # Step 3: Build the prompt for the LLM
        factsheet_str = "\n\n".join(factsheet_context) if factsheet_context else "No specific 2025 factsheet data was found in the local documents."
        
        prompt = f"""
        You are an expert mutual fund advisor. Your task is to provide a comprehensive answer to the user's query by synthesizing information from two sources: internal documents (2025 factsheets) and a real-time web search.

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

        Instructions:
        1.  Synthesize a single, coherent answer from BOTH of the sources above.
        2.  If the query asks for a comparison, structure the response accordingly.
        3.  Prioritize the web search results for the most current numbers (e.g., performance, AUM).
        4.  Use the factsheet data to provide details on fund managers, objectives, etc., if available.
        5.  Clearly state the source of your information (e.g., "According to the April 2025 factsheet...", "The latest web search results show...").
        6.  If the sources conflict, point this out. If only one source provides information, rely on that one.
        7.  Do NOT use any of your own internal knowledge. Base your answer ONLY on the data provided above.
        8.  Structure the response clearly with headings, bullet points, and tables for readability.

        Provide a comprehensive and helpful response now.
        """

        # Step 4: Generate the final response
        print("Generating synthesized response...")
        final_response = await self.client.generate(prompt)
        
        if not final_response:
            return "I apologize, but I was unable to generate a response based on the available information."
            
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