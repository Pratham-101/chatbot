import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from chatbot.enhanced_chatbot import EnhancedMutualFundChatbot

async def main():
    # Initialize vector store
    vector_store = VectorStore()
    print(f"Vector store contains {vector_store.count_documents()} documents.")

    # Initialize chatbot
    chatbot = EnhancedMutualFundChatbot(model_name="llama3")
    chatbot.set_vector_store(vector_store)

    # Ask a test query
    query = "Tell me about HDFC Large and Mid Cap Fund performance and compare it with ICICI Prudential funds."
    print(f"\nUser: {query}")
    answer = await chatbot.generate_answer(query)
    print(f"\nChatbot: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 