import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from chatbot.enhanced_chatbot import EnhancedMutualFundChatbot

async def debug_final_prompt():
    print("--- Debugging Final Prompt ---")
    
    # Initialize services
    vector_store = VectorStore()
    chatbot = EnhancedMutualFundChatbot(model_name="llama3")
    chatbot.set_vector_store(vector_store)

    # The user's query
    query = "Tell me about HDFC Large and Mid Cap Fund performance and compare it with ICICI Prudential funds."
    print(f"\n[USER QUERY]: {query}")

    # 1. Get the factsheet context that the chatbot retrieves
    print("\n\n--- 1. RETRIEVING FACTSHEET CONTEXT ---")
    factsheet_context = await chatbot._get_factsheet_context(query)
    
    if factsheet_context:
        print(f"Found {len(factsheet_context)} context chunks.")
        for i, chunk in enumerate(factsheet_context):
            print(f"\n[CHUNK {i+1}]:\n{chunk}")
    else:
        print("!!! No factsheet context was retrieved for this query. This is likely the problem. !!!")

    # 2. Get the web data
    print("\n\n--- 2. RETRIEVING WEB CONTEXT ---")
    web_data = await chatbot._get_web_data(query)
    print(f"\n[WEB DATA]:\n{web_data[:1000]}...") # Print first 1000 chars

    # 3. Construct the final prompt
    print("\n\n--- 3. CONSTRUCTING FINAL PROMPT ---")
    context_str = "\\n\\n".join(factsheet_context) if factsheet_context else "No factsheet data available."
    prompt = f"""
    You are an expert mutual fund advisor. Your knowledge cutoff is 2023. The user has provided you with new documents.
    You MUST answer the user's query based ONLY on the information provided in the "Factsheet Information" and "Current Market Information" sections below.
    Do NOT use any of your internal knowledge. State the date of the data you are using, which can be found in the context. Assume the current year is 2025.

    User Query: "{query}"

    ====================
    Factsheet Information (from 2025 documents):
    ---
    {context_str}
    ---
    ====================
    Current Market Information:
    ---
    {web_data if web_data else "No additional market data available."}
    ---
    ====================

    Instructions:
    1.  Synthesize a comprehensive answer using ONLY the provided information above.
    2.  If the factsheet information contains the answer, state that it comes from the factsheet (e.g., "According to the April 2025 factsheet...").
    3.  If the web information is more current or relevant, use that and state its source.
    4.  If the information is not in the provided context, you MUST state "The provided documents do not contain information about..." and do not provide an answer.
    5.  Structure your response clearly with headings and bullet points.

    Provide a comprehensive, helpful response based *only* on the provided documents:
    """
    print(prompt)


if __name__ == "__main__":
    asyncio.run(debug_final_prompt()) 