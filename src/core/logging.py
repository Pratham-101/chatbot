"""Structured logging configuration for the mutual fund chatbot."""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logging capabilities to classes."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_request_info(request_id: str, method: str, path: str, **kwargs: Any) -> None:
    """Log request information."""
    logger = get_logger("api.request")
    logger.info(
        "Request received",
        request_id=request_id,
        method=method,
        path=path,
        **kwargs
    )


def log_response_info(request_id: str, status_code: int, response_time: float, **kwargs: Any) -> None:
    """Log response information."""
    logger = get_logger("api.response")
    logger.info(
        "Response sent",
        request_id=request_id,
        status_code=status_code,
        response_time=response_time,
        **kwargs
    )


def log_error(error: Exception, context: Dict[str, Any] | None = None) -> None:
    """Log error with context."""
    logger = get_logger("error")
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        exc_info=True
    )


def log_performance(operation: str, duration: float, **kwargs: Any) -> None:
    """Log performance metrics."""
    logger = get_logger("performance")
    logger.info(
        "Performance metric",
        operation=operation,
        duration=duration,
        **kwargs
    )


def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security-related events."""
    logger = get_logger("security")
    logger.warning(
        "Security event",
        event_type=event_type,
        details=details
    )


# Initialize logging on module import
configure_logging() 