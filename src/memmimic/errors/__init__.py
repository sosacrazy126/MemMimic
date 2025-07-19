"""
MemMimic Error Handling Framework

A comprehensive error handling system providing structured exceptions, 
error context management, recovery mechanisms, and monitoring capabilities.

Usage Examples:
    Basic exception handling:
    >>> from memmimic.errors import MemoryRetrievalError, ErrorContext
    >>> try:
    ...     # risky operation
    ...     pass
    ... except Exception as e:
    ...     raise MemoryRetrievalError("Failed to retrieve memory", context={"memory_id": "123"}) from e
    
    Using error decorators:
    >>> from memmimic.errors import handle_errors, retry
    >>> @handle_errors(catch=[DatabaseError], log_level="ERROR")
    >>> @retry(max_attempts=3)
    >>> def database_operation():
    ...     # operation that might fail
    ...     pass
    
    Error context management:
    >>> from memmimic.errors import create_error_context
    >>> context = create_error_context(
    ...     operation="memory_retrieval",
    ...     component="active_manager",
    ...     metadata={"user_id": "123"}
    ... )
"""

# Core exception hierarchy
from .exceptions import (
    # Base exceptions
    MemMimicError,
    SystemError,
    MemMimicMemoryError,
    CXDError, 
    MCPError,
    APIError,
    ExternalServiceError,
    
    # System errors
    ConfigurationError,
    InitializationError,
    ResourceError,
    
    # Memory errors
    MemoryStorageError,
    MemoryRetrievalError,
    MemoryCorruptionError,
    MemoryValidationError,
    
    # CXD errors
    ClassificationError,
    TrainingError,
    ModelError,
    CXDIntegrationError,
    
    # MCP errors
    ProtocolError,
    HandlerError,
    CommunicationError,
    MCPValidationError,
    
    # API errors
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    
    # External service errors
    DatabaseError,
    NetworkError,
    TimeoutError,
    ExternalAPIError,
)

# Error context management
from .context import (
    ErrorContext,
    ErrorSeverity,
    create_error_context,
    get_current_context,
    with_error_context,
)

# Error handling decorators and utilities
from .handlers import (
    handle_errors,
    retry,
    circuit_breaker,
    fallback,
    log_errors,
    ErrorHandler,
    RetryPolicy,
    ExponentialBackoff,
    LinearBackoff,
    CircuitBreaker,
)

# Recovery mechanisms - TODO: implement in next phase
# from .recovery import (
#     RecoveryStrategy,
#     RetryStrategy,
#     FallbackStrategy,
#     CircuitBreakerStrategy,
#     RecoveryManager,
# )

# Error monitoring and analysis - TODO: implement in next phase
# from .monitoring import (
#     ErrorCollector,
#     ErrorAnalyzer,
#     ErrorMetrics,
#     get_error_stats,
#     register_error_handler,
# )

# Logging configuration
from .logging_config import (
    configure_error_logging,
    get_error_logger,
    StructuredFormatter,
    CorrelationIDFilter,
    ErrorContextFilter,
    SeverityLevelFilter,
    ErrorLoggingHandler,
    create_logger_with_context,
    log_error_with_context,
    enhance_error_with_logging,
)

# Version info
__version__ = "1.0.0"
__author__ = "MemMimic Error Framework Team"

# Export main factory functions
__all__ = [
    # Exception hierarchy
    "MemMimicError",
    "SystemError", "ConfigurationError", "InitializationError", "ResourceError",
    "MemMimicMemoryError", "MemoryStorageError", "MemoryRetrievalError", "MemoryCorruptionError", "MemoryValidationError",
    "CXDError", "ClassificationError", "TrainingError", "ModelError", "CXDIntegrationError", 
    "MCPError", "ProtocolError", "HandlerError", "CommunicationError", "MCPValidationError",
    "APIError", "ValidationError", "AuthenticationError", "AuthorizationError", "RateLimitError",
    "ExternalServiceError", "DatabaseError", "NetworkError", "TimeoutError", "ExternalAPIError",
    
    # Error context
    "ErrorContext", "ErrorSeverity", "create_error_context", "get_current_context", "with_error_context",
    
    # Error handling
    "handle_errors", "retry", "circuit_breaker", "fallback", "log_errors",
    "ErrorHandler", "RetryPolicy", "ExponentialBackoff", "LinearBackoff", "CircuitBreaker",
    
    # Recovery mechanisms - TODO: implement in next phase
    # "RecoveryStrategy", "RetryStrategy", "FallbackStrategy", "CircuitBreakerStrategy", "RecoveryManager",
    
    # Monitoring - TODO: implement in next phase
    # "ErrorCollector", "ErrorAnalyzer", "ErrorMetrics", "get_error_stats", "register_error_handler",
    
    # Logging
    "configure_error_logging", "get_error_logger", "StructuredFormatter", "CorrelationIDFilter",
    "ErrorContextFilter", "SeverityLevelFilter", "ErrorLoggingHandler",
    "create_logger_with_context", "log_error_with_context", "enhance_error_with_logging",
]


def configure_error_framework(
    log_level: str = "INFO",
    enable_monitoring: bool = True,
    enable_correlation_ids: bool = True,
    default_retry_attempts: int = 3,
    circuit_breaker_threshold: int = 10
) -> None:
    """
    Configure the error handling framework with default settings.
    
    Args:
        log_level: Default logging level for error messages
        enable_monitoring: Whether to enable error monitoring and collection
        enable_correlation_ids: Whether to track correlation IDs across operations
        default_retry_attempts: Default number of retry attempts
        circuit_breaker_threshold: Default circuit breaker failure threshold
        
    Example:
        >>> from memmimic.errors import configure_error_framework
        >>> configure_error_framework(
        ...     log_level="ERROR",
        ...     enable_monitoring=True,
        ...     default_retry_attempts=5
        ... )
    """
    # Configure logging
    configure_error_logging(
        level=log_level,
        include_correlation_ids=enable_correlation_ids,
        include_performance=True,
        include_context=True
    )
    
    # Set default retry policy
    RetryPolicy.set_default(
        max_attempts=default_retry_attempts,
        backoff=ExponentialBackoff()
    )
    
    # Configure circuit breaker defaults
    CircuitBreaker.set_default_threshold(circuit_breaker_threshold)
    
    # TODO: Initialize monitoring when monitoring module is implemented
    # if enable_monitoring:
    #     ErrorCollector.initialize()
    #     register_error_handler(ErrorCollector.collect_error)


def get_framework_status() -> dict:
    """
    Get current status and configuration of the error handling framework.
    
    Returns:
        Dictionary containing framework status information
        
    Example:
        >>> from memmimic.errors import get_framework_status
        >>> status = get_framework_status()
        >>> print(f"Active circuit breakers: {status['active_circuit_breakers']}")
    """
    return {
        "version": __version__,
        "logging_configured": hasattr(configure_error_logging, '_configured'),
        "monitoring_enabled": False,  # TODO: implement when monitoring is available
        "total_errors_collected": 0,  # TODO: implement when monitoring is available
        "active_circuit_breakers": CircuitBreaker.get_active_count(),
        "default_retry_attempts": RetryPolicy.get_default_attempts(),
    }