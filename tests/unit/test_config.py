import pytest
from src.core.config import settings

def test_settings():
    assert settings.app_name == "Mutual Fund Chatbot"
    assert settings.api_port == 8000
