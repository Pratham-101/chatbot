"""Chat API routes."""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..models.schemas import QueryRequest, QueryResponse
from ...core.exceptions import ValidationError
from ...core.logging import get_logger, log_performance
from ...services.chatbot.enhanced_chatbot import EnhancedChatbot

router = APIRouter()
logger = get_logger("api.chat")

# Global chatbot instance (in production, this should be properly managed)
chatbot: EnhancedChatbot | None = None


async def get_chatbot() -> EnhancedChatbot:
    """Get or create chatbot instance."""
    global chatbot
    if chatbot is None:
        logger.info("Initializing chatbot instance")
        chatbot = EnhancedChatbot()
    return chatbot


@router.post("/query", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    req: Request,
    chatbot_instance: EnhancedChatbot = Depends(get_chatbot),
) -> QueryResponse:
    """
    Process a user query and return a comprehensive response.
    
    This endpoint:
    - Validates the input query
    - Retrieves relevant context from factsheets
    - Performs web search for real-time information
    - Generates a comprehensive answer using LLM
    - Evaluates response quality
    - Returns structured data and metrics
    """
    start_time = time.time()
    request_id = req.headers.get("X-Request-ID", "unknown")
    
    try:
        logger.info(
            "Processing query",
            request_id=request_id,
            query_length=len(request.text),
            user_id=request.user_id,
            session_id=request.session_id,
        )
        
        # Process the query
        response = await chatbot_instance.process_query(request.text)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log performance
        log_performance(
            "query_processing",
            response_time,
            request_id=request_id,
            query_length=len(request.text),
        )
        
        # Create response
        query_response = QueryResponse(
            answer=response.get("answer", "No answer generated"),
            response_time=response_time,
            quality_metrics=response.get("quality_metrics", {}),
            structured_data=response.get("structured_data", {}),
            raw_response=response.get("raw_response", ""),
            request_id=request_id,
        )
        
        logger.info(
            "Query processed successfully",
            request_id=request_id,
            response_time=response_time,
            quality_score=query_response.quality_metrics.overall_score,
        )
        
        return query_response
        
    except ValidationError as e:
        logger.warning(
            "Validation error",
            request_id=request_id,
            error=str(e),
            field=e.field,
        )
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(
            "Error processing query",
            request_id=request_id,
            error=str(e),
            response_time=response_time,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}")
async def get_session_history(
    session_id: str,
    req: Request,
    chatbot_instance: EnhancedChatbot = Depends(get_chatbot),
) -> dict[str, Any]:
    """Get chat session history."""
    # This would typically query a database
    # For now, return a placeholder
    return {
        "session_id": session_id,
        "messages": [],
        "created_at": None,
        "last_activity": None,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    req: Request,
    chatbot_instance: EnhancedChatbot = Depends(get_chatbot),
) -> dict[str, str]:
    """Delete a chat session."""
    # This would typically delete from a database
    logger.info("Deleting session", session_id=session_id)
    return {"message": f"Session {session_id} deleted"}


@router.post("/feedback")
async def submit_feedback(
    request_id: str,
    rating: int,
    feedback_text: str | None = None,
    req: Request = None,
) -> dict[str, str]:
    """Submit feedback for a response."""
    logger.info(
        "Feedback submitted",
        request_id=request_id,
        rating=rating,
        feedback_text=feedback_text,
    )
    
    # This would typically store feedback in a database
    return {"message": "Feedback submitted successfully"} 