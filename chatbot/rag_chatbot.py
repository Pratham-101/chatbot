import asyncio
from typing import List
from chatbot.retrieval import Retriever
from ingestion.vector_store import VectorStore
from chatbot.generation import ResponseGenerator
import time
import json
import os

class RAGChatbot:
    def __init__(self, model_name: str = "llama3", ollama_url: str = None):
        import os
        vector_store = VectorStore()
        self.retriever = Retriever(vector_store)
        self.response_generator = ResponseGenerator(model_name=model_name)
        self.model_name = model_name
        self.ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
        self.cache = {}
        self.conversation_history = []  # For multi-turn conversation
        self.current_fund_name = None  # Track current fund name context

    async def generate_answer(self, query: str, k: int = 15) -> str:
        start_time = time.time()
        print(f"[RAG] generate_answer started for query: {query}")

        # Append query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        print("[RAG] Appended user query to conversation history")

        # Try to extract fund name from current query
        fund_name_in_query = self.retriever.extract_fund_name(query)
        print(f"[RAG] Extracted fund name from query: {fund_name_in_query}")

        # Handle short affirmative follow-up queries like "yes" or "yeah"
        if query.strip().lower() in ["yes", "yeah", "yep", "sure", "please", "ok", "okay"]:
            fund_name_in_query = self.current_fund_name
            print(f"[RAG] Affirmative query detected, using current fund name: {fund_name_in_query}")

        if fund_name_in_query:
            self.current_fund_name = fund_name_in_query
            print(f"[RAG] Updated current fund name context: {self.current_fund_name}")

        # If fund name missing in query but found in context, enrich query
        enriched_query = query
        if self.current_fund_name and not fund_name_in_query:
            # Append last user query and last assistant answer for better context
            last_user_query = ""
            last_assistant_answer = ""
            for turn in reversed(self.conversation_history):
                if turn["role"] == "user":
                    last_user_query = turn["content"]
                    break
            for turn in reversed(self.conversation_history):
                if turn["role"] == "assistant":
                    last_assistant_answer = turn["content"]
                    break
            enriched_query = f"{query} about {self.current_fund_name}. Previous question: {last_user_query}. Previous answer: {last_assistant_answer}"
            print(f"[RAG] Enriched query: {enriched_query}")

        # Check if query is about fund manager and try direct extraction first
        if "fund manager" in enriched_query.lower() or "manager of" in enriched_query.lower():
            fund_manager_info = self.retriever.get_fund_manager(enriched_query)
            print(f"[RAG] Fund manager info: {fund_manager_info}")
            if fund_manager_info and "not found" not in fund_manager_info.lower():
                elapsed = time.time() - start_time
                self.conversation_history.append({"role": "assistant", "content": fund_manager_info})
                # Log conversation to file
                self._log_conversation(query, fund_manager_info)
                print(f"[RAG] Returning fund manager info with elapsed time {elapsed:.2f} seconds")
                return f"{fund_manager_info}\n\n[Response time: {elapsed:.2f} seconds]"

        # Otherwise proceed with normal retrieval and generation
        context_chunks = self.retriever.get_relevant_context(enriched_query, k=k)
        print(f"[RAG] Retrieved {len(context_chunks)} context chunks for query: {enriched_query}")

        # Limit context chunks to top 3 for generation
        limited_context_chunks = context_chunks[:3]
        print(f"[RAG] Limited context chunks to top 3")

        # Retrieve web data using the response generator's web call method
        web_data = ""
        try:
            web_data = await asyncio.wait_for(self.response_generator._call_ollama_web(enriched_query), timeout=15.0)
            print(f"[RAG] Retrieved web data for query: {enriched_query}")
        except asyncio.TimeoutError:
            print("[RAG] Timeout retrieving web data")
        except Exception as e:
            print(f"[RAG] Failed to retrieve web data: {e}")

        if not limited_context_chunks and not web_data:
            print("[RAG] No context chunks or web data found, returning fallback message")
            return "I couldn't find specific information about this in our factsheets. Would you like me to:\n1. Try a broader search\n2. Search the web for more details\n3. Help you find similar funds?"

        # Include conversation history in prompt construction (simple concatenation)
        conversation_context = "\n".join(
            [f"{turn['role'].capitalize()}: {turn['content']}" for turn in self.conversation_history[-6:]]
        )
        print(f"[RAG] Constructed conversation context")

        # Generate answer with conversation context and enriched query
        # Pass conversation context explicitly to generation prompt
        try:
            answer = await asyncio.wait_for(
                self.response_generator.generate_response(
                    enriched_query, limited_context_chunks, web_data=web_data, conversation_context=conversation_context
                ),
                timeout=30.0
            )
            print(f"[RAG] Generated answer")
        except asyncio.TimeoutError:
            print("[RAG] Timeout generating answer")
            answer = "I'm having trouble generating a complete response right now. You can:\n1. Try rephrasing your question\n2. Ask for specific fund details\n3. Request a simpler summary"
        except Exception as e:
            print(f"[RAG] Exception generating answer: {e}")
            answer = "Sorry, an error occurred while generating the answer. Please try again."

        # Append assistant's answer to conversation history
        self.conversation_history.append({"role": "assistant", "content": answer})
        print(f"[RAG] Appended assistant answer to conversation history")

        # Log conversation to file
        self._log_conversation(query, answer)
        print(f"[RAG] Logged conversation")

        # Generate follow-up suggestion based on conversation history
        follow_up_prompt = (
            "Based on the previous conversation, suggest a relevant follow-up question or comparison "
            "that the user might be interested in. Respond briefly."
        )
        follow_up_suggestion = await self.response_generator.generate_response(
            follow_up_prompt, [], web_data=""
        )
        print(f"[RAG] Generated follow-up suggestion")

        elapsed = time.time() - start_time
        print(f"[RAG] Total response time: {elapsed:.2f} seconds")
        return f"{answer}\n\n[Response time: {elapsed:.2f} seconds]\n\nFollow-up suggestion: {follow_up_suggestion}"

    def _log_conversation(self, user_query: str, bot_response: str):
        log_entry = {
            "user_query": user_query,
            "bot_response": bot_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        log_file = "conversation_log.json"
        try:
            if not os.path.exists(log_file):
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump([log_entry], f, indent=2)
            else:
                with open(log_file, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(log_entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error logging conversation: {e}")

# Example usage for testing
if __name__ == "__main__":
    import asyncio
    rag_bot = RAGChatbot()
    query = "Tell me about Mirae Asset Ultra Short Duration Fund"
    answer = asyncio.run(rag_bot.generate_answer(query))
    print(f"Answer:\\n{answer}")
