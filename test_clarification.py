#!/usr/bin/env python3
"""
Test script for the clarification functionality
"""

import asyncio
from src.services.chatbot.comprehensive_finance_chatbot import get_comprehensive_chatbot

async def test_clarification():
    """Test the clarification functionality"""
    chatbot = get_comprehensive_chatbot()
    
    print("🤖 Testing Clarification Functionality")
    print("=" * 50)
    
    # First query - should trigger clarification
    print("\n1. Query: 'What is the performance of HDFC?'")
    result1 = await chatbot.answer_query("What is the performance of HDFC?")
    print("Response:")
    print(result1)
    
    # Second query - response to clarification
    print("\n2. User selects option 2")
    result2 = await chatbot.answer_query("2")
    print("Response:")
    print(result2)
    
    # Test with a different ambiguous query
    print("\n3. Query: 'Tell me about Axis fund'")
    result3 = await chatbot.answer_query("Tell me about Axis fund")
    print("Response:")
    print(result3)
    
    # Test with a specific fund name that should work directly
    print("\n4. Query: 'What is the performance of HDFC Flexi Cap Fund?'")
    result4 = await chatbot.answer_query("What is the performance of HDFC Flexi Cap Fund?")
    print("Response:")
    print(result4)

if __name__ == "__main__":
    asyncio.run(test_clarification()) 