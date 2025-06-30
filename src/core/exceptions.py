"""Custom exceptions for the mutual fund chatbot."""

from typing import Any, Dict, Optional


class ChatbotException(Exception):
    """Base exception for all chatbot-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: Dict[str, Any] | None = None,
        status_code: int = 500
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code


class ConfigurationError(ChatbotException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "CONFIG_ERROR", details, 500)


class LLMError(ChatbotException):
    """Raised when there's an error with the LLM service."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "LLM_ERROR", details, 503)


class VectorStoreError(ChatbotException):
    """Raised when there's an error with the vector store."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "VECTOR_STORE_ERROR", details, 503)


class WebSearchError(ChatbotException):
    """Raised when there's an error with web search."""
    
    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "WEB_SEARCH_ERROR", details, 503)


class ValidationError(ChatbotException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str | None = None, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "VALIDATION_ERROR", details, 400)
        self.field = field


class RateLimitError(ChatbotException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None) -> None:
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, "RATE_LIMIT_ERROR", details, 429)


class AuthenticationError(ChatbotException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, "AUTHENTICATION_ERROR", {}, 401)


class AuthorizationError(ChatbotException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed") -> None:
        super().__init__(message, "AUTHORIZATION_ERROR", {}, 403)


class ResourceNotFoundError(ChatbotException):
    """Raised when a resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str | None = None) -> None:
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        details = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(message, "RESOURCE_NOT_FOUND", details, 404)


class ServiceUnavailableError(ChatbotException):
    """Raised when a service is unavailable."""
    
    def __init__(self, service_name: str, details: Dict[str, Any] | None = None) -> None:
        message = f"Service {service_name} is unavailable"
        super().__init__(message, "SERVICE_UNAVAILABLE", details, 503)


class DataProcessingError(ChatbotException):
    """Raised when there's an error processing data."""
    
    def __init__(self, message: str, data_type: str | None = None, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message, "DATA_PROCESSING_ERROR", details, 422)
        self.data_type = data_type


class TimeoutError(ChatbotException):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout_seconds: float | None = None) -> None:
        message = f"Operation '{operation}' timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds} seconds"
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        super().__init__(message, "TIMEOUT_ERROR", details, 408)


def handle_exception(exc: Exception) -> ChatbotException:
    """Convert generic exceptions to appropriate ChatbotException."""
    
    if isinstance(exc, ChatbotException):
        return exc
    
    # Handle common exceptions
    if isinstance(exc, ValueError):
        return ValidationError(str(exc))
    elif isinstance(exc, TimeoutError):
        return TimeoutError("Unknown operation")
    elif isinstance(exc, ConnectionError):
        return ServiceUnavailableError("External service", {"error": str(exc)})
    elif isinstance(exc, FileNotFoundError):
        return ResourceNotFoundError("File", str(exc))
    else:
        return ChatbotException(
            f"Unexpected error: {str(exc)}",
            "UNEXPECTED_ERROR",
            {"original_exception": type(exc).__name__}
        ) 