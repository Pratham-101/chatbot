import unittest
import asyncio
from chatbot.rag_chatbot import RAGChatbot

class TestRAGChatbotIntegration(unittest.TestCase):
    def setUp(self):
        self.rag_bot = RAGChatbot()

    def test_generate_answer_known_fund(self):
        query = "Tell me about Mirae Asset Ultra Short Duration Fund"
        answer = asyncio.run(self.rag_bot.generate_answer(query))
        self.assertIsInstance(answer, str)
        self.assertNotIn("no mention", answer.lower())  # Expecting some relevant answer

    def test_generate_answer_unknown_fund(self):
        query = "Tell me about Unknown Fund XYZ"
        answer = asyncio.run(self.rag_bot.generate_answer(query))
        self.assertIsInstance(answer, str)
        self.assertTrue("could not find" in answer.lower() or "no mention" in answer.lower())

if __name__ == "__main__":
    unittest.main()
