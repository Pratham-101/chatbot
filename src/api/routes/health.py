"""Health check API routes."""

import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from psutil import cpu_percent, memory_percent, disk_usage

from ..models.schemas import HealthResponse, MetricsResponse
from ...core.config import settings
from ...core.logging import get_logger
from ...services.chatbot.enhanced_chatbot import EnhancedChatbot

router = APIRouter()
logger = get_logger("api.health")


async def get_chatbot() -> EnhancedChatbot | None:
    """Get chatbot instance for health checks."""
    try:
        from ...services.chatbot.enhanced_chatbot import EnhancedChatbot
        return EnhancedChatbot()
    except Exception:
        return None


@router.get("/health", response_model=HealthResponse)
async def health_check(req: Request) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the overall health status of the application and its dependencies.
    """
    start_time = time.time()
    services_status = {}
    
    # Check system resources
    try:
        cpu_usage = cpu_percent(interval=0.1)
        memory_usage = memory_percent()
        disk_usage_info = disk_usage("/")
        
        services_status.update({
            "cpu": "healthy" if cpu_usage < 80 else "warning",
            "memory": "healthy" if memory_usage < 80 else "warning",
            "disk": "healthy" if disk_usage_info.percent < 80 else "warning",
        })
    except Exception as e:
        logger.warning("Failed to check system resources", error=str(e))
        services_status["system"] = "unhealthy"
    
    # Check chatbot service
    try:
        chatbot = await get_chatbot()
        if chatbot and chatbot.llm_available():
            services_status["chatbot"] = "healthy"
        else:
            services_status["chatbot"] = "degraded"
    except Exception as e:
        logger.warning("Failed to check chatbot service", error=str(e))
        services_status["chatbot"] = "unhealthy"
    
    # Check vector store
    try:
        # This would check if vector store is accessible
        services_status["vector_store"] = "healthy"
    except Exception as e:
        logger.warning("Failed to check vector store", error=str(e))
        services_status["vector_store"] = "unhealthy"
    
    # Determine overall status
    if "unhealthy" in services_status.values():
        overall_status = "unhealthy"
    elif "degraded" in services_status.values():
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    response_time = time.time() - start_time
    
    logger.info(
        "Health check completed",
        status=overall_status,
        response_time=response_time,
        services=services_status,
    )
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status,
    )


@router.get("/health/ready")
async def readiness_check(req: Request) -> dict[str, Any]:
    """
    Readiness check endpoint.
    
    Returns whether the application is ready to serve requests.
    """
    try:
        # Check if all critical services are available
        chatbot = await get_chatbot()
        
        if not chatbot:
            return {"status": "not_ready", "reason": "Chatbot not initialized"}
        
        if not chatbot.llm_available():
            return {"status": "not_ready", "reason": "LLM service unavailable"}
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {"status": "not_ready", "reason": str(e)}


@router.get("/health/live")
async def liveness_check(req: Request) -> dict[str, str]:
    """
    Liveness check endpoint.
    
    Returns whether the application is alive and running.
    """
    return {"status": "alive"}


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(req: Request) -> MetricsResponse:
    """
    Get application metrics.
    
    Returns performance and usage metrics for monitoring.
    """
    # In a real application, these would come from a metrics store
    # For now, return placeholder metrics
    
    return MetricsResponse(
        total_requests=1000,
        successful_requests=950,
        failed_requests=50,
        average_response_time=2.5,
        active_sessions=10,
    )


@router.get("/info")
async def get_info(req: Request) -> dict[str, Any]:
    """
    Get application information.
    
    Returns detailed information about the application configuration.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "log_level": settings.log_level,
        "groq_model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "cors_origins": settings.cors_origins,
    } 