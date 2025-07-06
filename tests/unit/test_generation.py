import pytest
from unittest.mock import AsyncMock, patch
from chatbot.generation import ResponseGenerator

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_with_verified_chunks_updated(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            if "Mirae" in prompt:
                return "YES"
            else:
                return "NO"
        else:
            # Adjusted mock response to match actual response pattern
            return "📊 Fund Name: Mirae Asset Ultra Short Duration Fund\n\nPerformance Summary:\n- Expense Ratio: 0.00% per annum\n- Inception Date: 2020-10-01\n\n[Response time: 10.00 seconds]"
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Mirae Asset Ultra Short Duration Fund"
    context = [
        "Some unrelated context",
        "Details about Mirae Asset Ultra Short Duration Fund performance",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Fund Name" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_fallback_full_context_new(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            return "NO"
        else:
            return "The Unknown Fund!\n\nAfter conducting some research I found that there are several entities and organizations referred to as \"Unknown Fund\" or similar. If you have a specific Unknown Fund in mind, please provide me with additional context or details and I'll do my best to help you out!\n\n[Response time: 36.39 seconds]"
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Unknown Fund"
    context = [
        "Some unrelated context",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Unknown Fund" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_fallback_full_context_updated(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            return "NO"
        else:
            # Adjusted mock response to match actual fallback response pattern
            return "The Unknown Fund!\n\nAfter conducting some research I found that there are several entities and organizations referred to as \"Unknown Fund\" or similar. If you have a specific Unknown Fund in mind, please provide me with additional context or details and I'll do my best to help you out!\n\n[Response time: 36.39 seconds]"
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Unknown Fund"
    context = [
        "Some unrelated context",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Unknown Fund" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_no_fund_name_updated(mock_call):
    mock_call.return_value = "I'd be happy to help answer your general question! Please go ahead and ask away and I'll do my best to provide an accurate response using my internet connection.\n\n[Response time: 7.08 seconds]"

    generator = ResponseGenerator()
    query = "General question without fund name"
    context = [
        "Some context"
    ]

    response = await generator.generate_response(query, context)
    assert "happy to help" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_additional_fund_query_updated(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            if "HDFC Balanced Advantage Fund" in prompt:
                return "YES"
            else:
                return "NO"
        else:
            return "Detailed answer about HDFC Balanced Advantage Fund."
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about HDFC Balanced Advantage Fund"
    context = [
        "Some unrelated context",
        "Details about HDFC Balanced Advantage Fund performance",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Detailed answer" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_with_verified_chunks(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            if "Mirae" in prompt:
                return "YES"
            else:
                return "NO"
        else:
            return "This is the final answer based on verified context."
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Mirae Asset Ultra Short Duration Fund"
    context = [
        "Some unrelated context",
        "Details about Mirae Asset Ultra Short Duration Fund performance",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "final answer" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_fallback_full_context(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            return "NO"
        else:
            # Adjusted mock response to match actual fallback response pattern
            return "The \"Unknown Fund\" is a fascinating topic!"
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Unknown Fund"
    context = [
        "Some unrelated context",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "fascinating topic" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_no_fund_name(mock_call):
    mock_call.return_value = "Answer without fund name context."

    generator = ResponseGenerator()
    query = "General question without fund name"
    context = [
        "Some context"
    ]

    response = await generator.generate_response(query, context)
    assert "Answer without fund name" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_additional_fund_query(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            if "HDFC Balanced Advantage Fund" in prompt:
                return "YES"
            else:
                return "NO"
        else:
            return "Detailed answer about HDFC Balanced Advantage Fund."
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about HDFC Balanced Advantage Fund"
    context = [
        "Some unrelated context",
        "Details about HDFC Balanced Advantage Fund performance",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Detailed answer" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_fallback_full_context(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            return "NO"
        else:
            return "Fallback final answer using full context."
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about Unknown Fund"
    context = [
        "Some unrelated context",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Fallback final answer" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_no_fund_name(mock_call):
    mock_call.return_value = "Answer without fund name context."

    generator = ResponseGenerator()
    query = "General question without fund name"
    context = [
        "Some context"
    ]

    response = await generator.generate_response(query, context)
    assert "Answer without fund name" in response

@pytest.mark.asyncio
@patch.object(ResponseGenerator, '_call_ollama_async', new_callable=AsyncMock)
async def test_generate_response_additional_fund_query(mock_call):
    async def side_effect(prompt, timeout=120, retries=3):
        if "verification assistant" in prompt:
            if "HDFC Balanced Advantage Fund" in prompt:
                return "YES"
            else:
                return "NO"
        else:
            return "Detailed answer about HDFC Balanced Advantage Fund."
    mock_call.side_effect = side_effect

    generator = ResponseGenerator()
    query = "Tell me about HDFC Balanced Advantage Fund"
    context = [
        "Some unrelated context",
        "Details about HDFC Balanced Advantage Fund performance",
        "Other fund details"
    ]

    response = await generator.generate_response(query, context)
    assert "Detailed answer" in response
