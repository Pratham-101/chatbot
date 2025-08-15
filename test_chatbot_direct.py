import asyncio
import sys
import os
sys.path.append('src')

from services.chatbot.enhanced_chatbot import EnhancedMutualFundChatbot

async def test_chatbot_direct():
    print("=== Testing Chatbot API Integration ===")
    
    # Initialize chatbot
    chatbot = EnhancedMutualFundChatbot()
    
    # Test query
    query = "Tell me about Axis India Manufacturing Fund"
    print(f"\nQuery: {query}")
    
    # Test the API data retrieval directly
    print("\n=== Testing API Data Retrieval ===")
    fund_data = await chatbot._get_fund_data_robust(query)
    
    print(f"Fund data keys: {fund_data.keys()}")
    print(f"API data available: {bool(fund_data.get('api_data'))}")
    
    if fund_data.get('api_data'):
        api_data = fund_data['api_data']
        print(f"\n=== REAL API DATA ===")
        print(f"Fund Name: {api_data.get('scheme_name')}")
        print(f"NAV: {api_data.get('nav')}")
        print(f"Manager: {api_data.get('fund_manager')}")
        print(f"Expense Ratio: {api_data.get('expense_ratio')}")
        print(f"Risk Category: {api_data.get('sebi_risk_category')}")
        print(f"1Y Return: {api_data.get('return_1y')}")
        print(f"3Y Return: {api_data.get('return_3y')}")
        print(f"5Y Return: {api_data.get('return_5y')}")
    else:
        print("No API data found!")
    
    # Test the full process_query method
    print("\n=== Testing Full Process Query ===")
    result = await chatbot.process_query(query)
    
    print(f"\n=== FINAL RESPONSE ===")
    print(result.get('answer', 'No answer'))
    print(f"\nQuality Score: {result.get('quality_score')}")
    print(f"Sources: {result.get('sources')}")

if __name__ == "__main__":
    asyncio.run(test_chatbot_direct()) 