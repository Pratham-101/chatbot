import unittest
import asyncio
from unittest.mock import patch, AsyncMock
from chatbot.rag_chatbot import RAGChatbot

class TestRAGChatbotIntegration(unittest.TestCase):
    def setUp(self):
        self.rag_bot = RAGChatbot()

    @patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
    def test_generate_answer_known_fund(self, mock_generate):
        mock_generate.return_value = "Mocked response"
        query = "Tell me about Mirae Asset Ultra Short Duration Fund"
        answer = asyncio.run(self.rag_bot.generate_answer(query))
        self.assertIsInstance(answer, str)
        self.assertNotIn("no mention", answer.lower())  # Expecting some relevant answer

    @patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
    def test_generate_answer_unknown_fund(self, mock_generate):
        mock_generate.return_value = "Mocked response"
        query = "Tell me about Unknown Fund XYZ"
        answer = asyncio.run(self.rag_bot.generate_answer(query))
        self.assertIsInstance(answer, str)
        self.assertTrue("could not find" in answer.lower() or "no mention" in answer.lower())

if __name__ == "__main__":
    unittest.main()
