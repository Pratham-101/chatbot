"""Main FastAPI application entry point."""

from fastapi import FastAPI
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Create the application instance
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="An intelligent chatbot for mutual fund information and analysis",
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    ) 