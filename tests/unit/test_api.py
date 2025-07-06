import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from api.app import app
from httpx import AsyncClient
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.mark.asyncio
@patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
async def test_ask_endpoint_valid_question(mock_generate):
    mock_generate.return_value = "Mocked response"
    async with AsyncClient() as ac:
        response = await ac.post("http://localhost:8000/ask", json={"question": "Tell me about Mirae Asset Ultra Short Duration Fund"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert data["answer"].strip() != ""

# Other tests can be converted similarly if needed

    @patch("chatbot.rag_chatbot.RAGChatbot.generate_answer")
    def test_ask_endpoint_empty_question(self, mock_generate):
        mock_generate.return_value = "Mocked response"
        response = client.post("/ask", json={"question": ""})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Please provide a question.")

    @patch("chatbot.rag_chatbot.RAGChatbot.generate_answer")
    def test_ask_endpoint_missing_question(self, mock_generate):
        mock_generate.return_value = "Mocked response"
        response = client.post("/ask", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Please provide a question.")

    def test_ask_endpoint_invalid_method(self):
        response = client.get("/ask")
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.asyncio
@patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
async def test_ask_endpoint_timeout(mock_generate):
    # Simulate timeout by making generate_answer sleep longer than API timeout
    async def slow_generate_answer(query):
        await asyncio.sleep(35)
        return "Slow response"
    mock_generate.side_effect = slow_generate_answer

    response = client.post("/ask", json={"question": "Test timeout"})
    assert response.status_code == 504
    data = response.json()
    assert "timed out" in data["answer"].lower()

@pytest.mark.asyncio
@patch("chatbot.rag_chatbot.RAGChatbot.generate_answer", new_callable=AsyncMock)
async def test_ask_endpoint_server_error(mock_generate):
    # Simulate exception in generate_answer
    async def error_generate_answer(query):
        raise Exception("Test error")
    mock_generate.side_effect = error_generate_answer

    response = client.post("/ask", json={"question": "Test error"})
    assert response.status_code == 500
    data = response.json()
    assert "error occurred" in data["answer"].lower()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"

if __name__ == "__main__":
    unittest.main()
