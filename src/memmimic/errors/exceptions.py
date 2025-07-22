"""
Centralized exception hierarchy for MemMimic.

Provides a comprehensive, structured approach to error handling with rich context,
consistent formatting, and domain-specific exception types.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and alerting."""
    CRITICAL = "critical"    # System-breaking errors requiring immediate attention
    HIGH = "high"           # Important errors affecting functionality
    MEDIUM = "medium"       # Moderate errors with workarounds available
    LOW = "low"            # Minor errors with minimal impact
    INFO = "info"          # Informational errors for debugging


class MemMimicError(Exception):
    """
    Base exception for all MemMimic-specific errors.
    
    Provides structured error information including context, correlation IDs,
    and severity levels for consistent error handling across the platform.
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional error context and metadata
        error_id: Unique identifier for this error instance
        timestamp: When the error occurred
        severity: Error severity level
        correlation_id: ID for tracking related operations
        component: Component where error occurred
        operation: Operation that was being performed
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.severity = severity
        self.correlation_id = correlation_id
        self.component = component
        self.operation = operation
        
        # Store original exception if this was raised from another exception
        # Note: This will be None during __init__ and should be checked later
        self.original_exception = None
    
    @property
    def _get_original_exception(self):
        """Get the original exception from the chain."""
        return self.__cause__ or self.__context__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "correlation_id": self.correlation_id,
            "context": self.context,
            "original_exception": {
                "type": type(self._get_original_exception).__name__,
                "message": str(self._get_original_exception)
            } if self._get_original_exception else None
        }
    
    def add_context(self, key: str, value: Any) -> 'MemMimicError':
        """Add additional context to the error."""
        self.context[key] = value
        return self
    
    def set_correlation_id(self, correlation_id: str) -> 'MemMimicError':
        """Set correlation ID for tracking related operations."""
        self.correlation_id = correlation_id
        return self
    
    def set_component(self, component: str) -> 'MemMimicError':
        """Set the component where error occurred."""
        self.component = component
        return self
    
    def set_operation(self, operation: str) -> 'MemMimicError':
        """Set the operation that was being performed."""
        self.operation = operation
        return self
    
    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.component:
            parts.append(f"Component: {self.component}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        if self.correlation_id:
            parts.append(f"Correlation ID: {self.correlation_id}")
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}', error_id='{self.error_id}')"


# =============================================================================
# System Errors - Infrastructure and configuration issues
# =============================================================================

class SystemError(MemMimicError):
    """Base class for system-level errors."""
    
    def __init__(self, message: str, **kwargs):
        # Extract and set default severity if not provided
        if 'severity' not in kwargs:
            kwargs['severity'] = ErrorSeverity.HIGH
        super().__init__(message, **kwargs)


class ConfigurationError(SystemError):
    """Raised when there are configuration-related issues."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if config_key:
            context['config_key'] = config_key
        
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            context=context,
            **kwargs
        )


class InitializationError(SystemError):
    """Raised when component initialization fails."""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(
            message,
            error_code="INITIALIZATION_ERROR",
            component=component,
            **kwargs
        )


class ResourceError(SystemError):
    """Raised when system resources are unavailable or exhausted."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            context=context,
            **kwargs
        )


# =============================================================================
# Memory Errors - Memory storage, retrieval, and management issues
# =============================================================================

class MemMimicMemoryError(MemMimicError):
    """Base class for memory-related errors."""
    
    def __init__(self, message: str, memory_id: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if memory_id:
            context['memory_id'] = memory_id
        
        super().__init__(
            message,
            component="memory_system",
            context=context,
            **kwargs
        )


class MemoryStorageError(MemMimicMemoryError):
    """Raised when memory storage operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="MEMORY_STORAGE_ERROR",
            operation="memory_storage",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class MemoryRetrievalError(MemMimicMemoryError):
    """Raised when memory retrieval operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="MEMORY_RETRIEVAL_ERROR",
            operation="memory_retrieval",
            **kwargs
        )


class MemoryCorruptionError(MemMimicMemoryError):
    """Raised when memory data is corrupted or invalid."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="MEMORY_CORRUPTION_ERROR",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class MemoryValidationError(MemMimicMemoryError):
    """Raised when memory data fails validation."""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            error_code="MEMORY_VALIDATION_ERROR",
            context=context,
            **kwargs
        )


# =============================================================================
# CXD Errors - Classification and model-related issues
# =============================================================================

class CXDError(MemMimicError):
    """Base class for CXD (Control/Context/Data) classification errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            component="cxd_system",
            **kwargs
        )


class ClassificationError(CXDError):
    """Raised when CXD classification operations fail."""
    
    def __init__(self, message: str, content: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if content:
            # Store first 200 chars for debugging without logging full content
            context['content_preview'] = content[:200] + "..." if len(content) > 200 else content
            context['content_length'] = len(content)
        
        super().__init__(
            message,
            error_code="CLASSIFICATION_ERROR",
            operation="cxd_classification",
            context=context,
            **kwargs
        )


class TrainingError(CXDError):
    """Raised when CXD model training fails."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if model_type:
            context['model_type'] = model_type
        
        super().__init__(
            message,
            error_code="TRAINING_ERROR",
            operation="model_training",
            context=context,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ModelError(CXDError):
    """Raised when CXD model operations fail."""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if model_id:
            context['model_id'] = model_id
        
        super().__init__(
            message,
            error_code="MODEL_ERROR",
            context=context,
            **kwargs
        )


class CXDIntegrationError(CXDError):
    """Raised when CXD integration with other systems fails."""
    
    def __init__(self, message: str, integration_point: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if integration_point:
            context['integration_point'] = integration_point
        
        super().__init__(
            message,
            error_code="CXD_INTEGRATION_ERROR",
            context=context,
            **kwargs
        )


# =============================================================================
# MCP Errors - Model Context Protocol issues
# =============================================================================

class MCPError(MemMimicError):
    """Base class for MCP (Model Context Protocol) errors."""
    
    def __init__(self, message: str, request_id: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if request_id:
            context['request_id'] = request_id
        
        super().__init__(
            message,
            component="mcp_system",
            context=context,
            **kwargs
        )


class ProtocolError(MCPError):
    """Raised when MCP protocol violations occur."""
    
    def __init__(self, message: str, protocol_version: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if protocol_version:
            context['protocol_version'] = protocol_version
        
        super().__init__(
            message,
            error_code="PROTOCOL_ERROR",
            context=context,
            **kwargs
        )


class HandlerError(MCPError):
    """Raised when MCP handler operations fail."""
    
    def __init__(self, message: str, handler_type: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if handler_type:
            context['handler_type'] = handler_type
        
        super().__init__(
            message,
            error_code="HANDLER_ERROR",
            context=context,
            **kwargs
        )


class CommunicationError(MCPError):
    """Raised when MCP communication fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="COMMUNICATION_ERROR",
            **kwargs
        )


class MCPValidationError(MCPError):
    """Raised when MCP request/response validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            error_code="MCP_VALIDATION_ERROR",
            context=context,
            **kwargs
        )


# =============================================================================
# API Errors - Web API and authentication issues
# =============================================================================

class APIError(MemMimicError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if status_code:
            context['status_code'] = status_code
        
        super().__init__(
            message,
            component="api_system",
            context=context,
            **kwargs
        )


class ValidationError(APIError):
    """Raised when API input validation fails."""
    
    def __init__(self, message: str, field_errors: Optional[Dict[str, str]] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if field_errors:
            context['field_errors'] = field_errors
        
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            context=context,
            **kwargs
        )


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if required_permission:
            context['required_permission'] = required_permission
        
        super().__init__(
            message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            context=context,
            **kwargs
        )


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, limit: Optional[int] = None, reset_time: Optional[int] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if limit:
            context['rate_limit'] = limit
        if reset_time:
            context['reset_time'] = reset_time
        
        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            context=context,
            **kwargs
        )


# =============================================================================
# External Service Errors - Database, network, and third-party service issues
# =============================================================================

class ExternalServiceError(MemMimicError):
    """Base class for external service errors."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if service_name:
            context['service_name'] = service_name
        
        super().__init__(
            message,
            component="external_services",
            context=context,
            **kwargs
        )


class DatabaseError(ExternalServiceError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if query:
            # Store first 500 chars of query for debugging
            context['query_preview'] = query[:500] + "..." if len(query) > 500 else query
        
        # Avoid service_name conflict
        if 'service_name' not in kwargs:
            kwargs['service_name'] = "database"
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(
            message,
            error_code="DATABASE_ERROR",
            context=context,
            **kwargs
        )


class NetworkError(ExternalServiceError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if url:
            context['url'] = url
        if status_code:
            context['status_code'] = status_code
        
        super().__init__(
            message,
            error_code="NETWORK_ERROR",
            context=context,
            **kwargs
        )


class TimeoutError(ExternalServiceError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            context=context,
            **kwargs
        )


class ExternalAPIError(ExternalServiceError):
    """Raised when external API calls fail."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, response_code: Optional[int] = None, **kwargs):
        # Extract context and remove from kwargs to avoid conflicts
        context = kwargs.pop('context', {})
        if api_name:
            context['api_name'] = api_name
        if response_code:
            context['response_code'] = response_code
        
        super().__init__(
            message,
            error_code="EXTERNAL_API_ERROR",
            context=context,
            **kwargs
        )


# =============================================================================
# Utility functions for exception creation and handling
# =============================================================================

def create_error(
    error_type: str,
    message: str,
    **kwargs
) -> MemMimicError:
    """
    Factory function to create errors by type name.
    
    Args:
        error_type: Name of the error class to create
        message: Error message
        **kwargs: Additional error context
        
    Returns:
        Instance of the specified error type
        
    Raises:
        ValueError: If error_type is not recognized
    """
    error_classes = {
        'system': SystemError,
        'configuration': ConfigurationError,
        'initialization': InitializationError,
        'resource': ResourceError,
        'memory': MemMimicMemoryError,
        'memory_storage': MemoryStorageError,
        'memory_retrieval': MemoryRetrievalError,
        'memory_corruption': MemoryCorruptionError,
        'memory_validation': MemoryValidationError,
        'cxd': CXDError,
        'classification': ClassificationError,
        'training': TrainingError,
        'model': ModelError,
        'cxd_integration': CXDIntegrationError,
        'mcp': MCPError,
        'protocol': ProtocolError,
        'handler': HandlerError,
        'communication': CommunicationError,
        'mcp_validation': MCPValidationError,
        'api': APIError,
        'validation': ValidationError,
        'authentication': AuthenticationError,
        'authorization': AuthorizationError,
        'rate_limit': RateLimitError,
        'external_service': ExternalServiceError,
        'database': DatabaseError,
        'network': NetworkError,
        'timeout': TimeoutError,
        'external_api': ExternalAPIError,
    }
    
    error_class = error_classes.get(error_type.lower())
    if not error_class:
        raise ValueError(f"Unknown error type: {error_type}")
    
    return error_class(message, **kwargs)


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if an error is retriable based on its type and context.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is likely retriable, False otherwise
    """
    # Network and timeout errors are generally retriable
    if isinstance(error, (NetworkError, TimeoutError)):
        return True
    
    # Database errors might be retriable depending on the specific error
    if isinstance(error, DatabaseError):
        # Check if it's a connection or timeout related database error
        error_message = str(error).lower()
        retriable_patterns = ['connection', 'timeout', 'temporary', 'deadlock']
        return any(pattern in error_message for pattern in retriable_patterns)
    
    # External API errors might be retriable for certain status codes
    if isinstance(error, ExternalAPIError):
        context = getattr(error, 'context', {})
        status_code = context.get('response_code') or context.get('status_code')
        if status_code:
            # 5xx errors and 429 (rate limit) are generally retriable
            return status_code >= 500 or status_code == 429
    
    # Configuration and validation errors are generally not retriable
    if isinstance(error, (ConfigurationError, ValidationError, AuthenticationError)):
        return False
    
    # Default to not retriable for safety
    return False


def get_error_severity(error: Exception) -> ErrorSeverity:
    """
    Get the severity level of an error.
    
    Args:
        error: Exception to evaluate
        
    Returns:
        ErrorSeverity level
    """
    if isinstance(error, MemMimicError):
        return error.severity
    
    # Determine severity based on error type for non-MemMimic errors
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return ErrorSeverity.CRITICAL
    elif isinstance(error, (MemoryError, OSError)):
        return ErrorSeverity.HIGH
    elif isinstance(error, (ValueError, TypeError)):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW