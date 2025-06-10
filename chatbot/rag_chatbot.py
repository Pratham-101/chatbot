import asyncio
from typing import List
from chatbot.retrieval import Retriever
from ingestion.vector_store import VectorStore
from chatbot.generation import ResponseGenerator
import time

class RAGChatbot:
    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://127.0.0.1:11434/api/generate"):
        vector_store = VectorStore()
        self.retriever = Retriever(vector_store)
        self.response_generator = ResponseGenerator(model_name=model_name)
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cache = {}

    async def generate_answer(self, query: str, k: int = 15) -> str:
        start_time = time.time()
        context_chunks = self.retriever.get_relevant_context(query, k=k)
        print(f"[RAG] Retrieved {len(context_chunks)} context chunks for query: {query}")

        # Retrieve web data using the response generator's web call method
        web_data = ""
        try:
            web_data = await self.response_generator._call_ollama_web(query)
            print(f"[RAG] Retrieved web data for query: {query}")
        except Exception as e:
            print(f"[RAG] Failed to retrieve web data: {e}")

        if not context_chunks and not web_data:
            return "Sorry, I could not find relevant information in the factsheets or on the web."

        answer = await self.response_generator.generate_response(query, context_chunks, web_data=web_data)
        elapsed = time.time() - start_time
        return f"{answer}\n\n[Response time: {elapsed:.2f} seconds]"

# Example usage for testing
if __name__ == "__main__":
    import asyncio
    rag_bot = RAGChatbot()
    query = "Tell me about Mirae Asset Ultra Short Duration Fund"
    answer = asyncio.run(rag_bot.generate_answer(query))
    print(f"Answer:\n{answer}")
