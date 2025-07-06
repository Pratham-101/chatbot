import pytest
import requests
import time

UI_URL = "http://localhost:30081"
API_ASK_URL = "http://localhost:30080/ask"
API_HEALTH_URL = "http://localhost:30080/health"

def test_health_endpoint():
    response = requests.get(API_HEALTH_URL)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

def test_ask_endpoint_valid_question():
    question = "Tell me about Mirae Asset Ultra Short Duration Fund"
    response = requests.post(API_ASK_URL, json={"question": question}, timeout=35)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"].strip()) > 0

def test_ui_chat_interaction():
    # Basic test to check UI is reachable
    response = requests.get(UI_URL)
    assert response.status_code == 200
    assert "Mutual Fund Factsheet Chatbot" in response.text

    # For full UI interaction, use Playwright or Selenium in separate test_ui.py

def test_performance_ask_endpoint():
    start_time = time.time()
    question = "Performance test question"
    response = requests.post(API_ASK_URL, json={"question": question}, timeout=35)
    duration = time.time() - start_time
    assert response.status_code == 200
    assert duration < 30  # Should respond within timeout

# Additional tests for error handling, edge cases, and fund manager extraction can be added here
