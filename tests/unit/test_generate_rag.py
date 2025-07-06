import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from chatbot.rag_chatbot import RAGChatbot

@pytest.mark.asyncio
@patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
async def test_generate_answer(mock_generate):
    mock_generate.return_value = "Mocked response"
    rag_bot = RAGChatbot()
    query = "Tell me about HDFC Mutual Fund"
    print(f"Testing generate_answer with query: {query}")
    answer = await rag_bot.generate_answer(query)
    print(f"Answer:\\n{answer}")

if __name__ == "__main__":
    asyncio.run(test_generate_answer())
