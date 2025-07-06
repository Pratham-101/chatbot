import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture(scope="session")
def playwright_context():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        context.close()
        browser.close()

def test_streamlit_chatbot_ui(playwright_context):
    page = playwright_context.new_page()
    # Adjust URL if backend UI is served on different port or path
    page.goto("http://localhost:8501")

    # Wait for Streamlit title to appear
    page.wait_for_selector("text=Mutual Fund Factsheet Chatbot")

    # Type a question in the chat input
    chat_input = page.locator("textarea[aria-label='Ask a question about mutual funds']")
    chat_input.fill("Tell me about Mirae Asset Ultra Short Duration Fund")
    chat_input.press("Enter")

    # Wait for assistant response to appear
    page.wait_for_selector("text=assistant", timeout=15000)  # wait up to 15 seconds

    # Check that assistant response contains expected text snippet
    assistant_messages = page.locator("div.stChatMessage > div[data-testid='stMarkdownContainer']")
    assert assistant_messages.count() > 0
    last_message = assistant_messages.nth(-1).inner_text()
    assert "Fund" in last_message or "Sorry" not in last_message

    # Additional UI interaction tests can be added here
