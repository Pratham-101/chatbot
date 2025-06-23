import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.rag_chatbot import RAGChatbot

async def test_chatbot():
    print("Testing Mutual Fund Chatbot...")
    print("=" * 50)
    
    # Initialize the chatbot
    chatbot = RAGChatbot(model_name="llama3")
    
    # Test questions
    test_questions = [
        "Tell me about HDFC Large and Mid Cap Fund",
        "What is the performance of ICICI Prudential funds?",
        "Compare HDFC and Kotak mutual funds"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 30)
        
        try:
            answer = await chatbot.generate_answer(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(test_chatbot()) 