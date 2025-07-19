"""
Structured logging configuration for the MemMimic error handling framework.

Provides enhanced logging with correlation IDs, structured formatting,
context preservation, and integration with the error handling system.
"""

import logging
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum

from .exceptions import ErrorSeverity, MemMimicError
from .context import get_current_context, get_context_correlation_id


class LogLevel(Enum):
    """Enhanced log levels with error handling integration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Structured log context information."""
    timestamp: str
    level: str
    logger: str
    message: str
    correlation_id: Optional[str] = None
    error_id: Optional[str] = None
    error_code: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging with error context integration.
    
    Produces machine-readable logs with consistent structure and rich context
    from the error handling framework.
    """
    
    def __init__(
        self,
        include_correlation_ids: bool = True,
        include_performance: bool = True,
        include_context: bool = True,
        json_indent: Optional[int] = None,
        exclude_fields: Optional[List[str]] = None
    ):
        super().__init__()
        self.include_correlation_ids = include_correlation_ids
        self.include_performance = include_performance
        self.include_context = include_context
        self.json_indent = json_indent
        self.exclude_fields = exclude_fields or []
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Get current error context
        error_context = get_current_context()
        correlation_id = get_context_correlation_id() if self.include_correlation_ids else None
        
        # Extract error information if present
        error_info = self._extract_error_info(record)
        
        # Build log context
        log_context = LogContext(
            timestamp=datetime.now().isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            correlation_id=correlation_id or getattr(record, 'correlation_id', None),
            error_id=error_info.get('error_id'),
            error_code=error_info.get('error_code'),
            component=error_info.get('component') or getattr(record, 'component', None) or (error_context.component if error_context else None),
            operation=error_info.get('operation') or getattr(record, 'operation', None) or (error_context.operation if error_context else None),
            user_id=getattr(record, 'user_id', None) or (error_context.user_id if error_context else None),
            session_id=getattr(record, 'session_id', None),
            thread_id=str(threading.current_thread().ident),
            process_id=record.process,
            context=self._build_context(record, error_context),
            metadata=self._build_metadata(record),
            performance=self._build_performance(record, error_context),
            stack_trace=self.formatException(record.exc_info) if record.exc_info else None
        )
        
        # Convert to dict and filter excluded fields
        log_dict = log_context.to_dict()
        for field in self.exclude_fields:
            log_dict.pop(field, None)
        
        # Format as JSON
        try:
            return json.dumps(log_dict, indent=self.json_indent, default=str)
        except (TypeError, ValueError) as e:
            # Fallback to simple message if JSON serialization fails
            return f"[JSON_ERROR] {log_context.message} (serialization failed: {e})"
    
    def _extract_error_info(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract error information from log record."""
        error_info = {}
        
        # Check if there's an exception
        if record.exc_info and record.exc_info[1]:
            exception = record.exc_info[1]
            if isinstance(exception, MemMimicError):
                error_info['error_id'] = exception.error_id
                error_info['error_code'] = exception.error_code
                error_info['severity'] = exception.severity.value
                error_info['component'] = exception.component
                error_info['operation'] = exception.operation
        
        # Check record extras
        for attr in ['error_id', 'error_code', 'error_type', 'severity', 'component', 'operation']:
            if hasattr(record, attr):
                error_info[attr] = getattr(record, attr)
        
        return error_info
    
    def _build_context(self, record: logging.LogRecord, error_context) -> Optional[Dict[str, Any]]:
        """Build context information from record and error context."""
        if not self.include_context:
            return None
        
        context = {}
        
        # Add error context if available
        if error_context:
            context.update(error_context.metadata)
        
        # Add context from MemMimicError exception if present
        if record.exc_info and record.exc_info[1]:
            exception = record.exc_info[1]
            if isinstance(exception, MemMimicError) and exception.context:
                context.update(exception.context)
        
        # Add record-specific context
        for attr in ['function', 'module', 'filename', 'lineno']:
            if hasattr(record, attr):
                context[attr] = getattr(record, attr)
        
        # Add custom context from record
        if hasattr(record, 'context') and isinstance(record.context, dict):
            context.update(record.context)
        
        return context if context else None
    
    def _build_metadata(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Build metadata from record extras."""
        metadata = {}
        
        # Standard metadata fields
        metadata_fields = [
            'request_id', 'trace_id', 'span_id', 'version', 'environment',
            'service_name', 'hostname', 'deployment_id'
        ]
        
        for field in metadata_fields:
            if hasattr(record, field):
                metadata[field] = getattr(record, field)
        
        # Add any other extras that don't conflict with standard fields
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'getMessage', 'extra'
        }
        
        for key, value in record.__dict__.items():
            if (key not in standard_fields and 
                key not in metadata_fields and 
                not key.startswith('_') and
                key not in ['context', 'component', 'operation', 'user_id', 'correlation_id']):
                metadata[key] = value
        
        return metadata if metadata else None
    
    def _build_performance(self, record: logging.LogRecord, error_context) -> Optional[Dict[str, Any]]:
        """Build performance information."""
        if not self.include_performance:
            return None
        
        performance = {}
        
        # Add error context performance data
        if error_context and error_context.duration_ms is not None:
            performance['duration_ms'] = error_context.duration_ms
            performance['memory_usage_mb'] = error_context.memory_usage_mb
            performance['cpu_usage_percent'] = error_context.cpu_usage_percent
        
        # Add record-specific performance data
        perf_fields = ['duration_ms', 'response_time', 'memory_usage', 'cpu_usage']
        for field in perf_fields:
            if hasattr(record, field):
                performance[field] = getattr(record, field)
        
        return performance if performance else None


class CorrelationIDFilter(logging.Filter):
    """
    Filter that adds correlation IDs to log records.
    
    Automatically adds correlation IDs from the error context or generates
    new ones for request tracking and debugging.
    """
    
    def __init__(self, auto_generate: bool = True):
        super().__init__()
        self.auto_generate = auto_generate
        self._local = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record."""
        # Try to get correlation ID from error context
        correlation_id = get_context_correlation_id()
        
        # Try to get from record extras
        if not correlation_id:
            correlation_id = getattr(record, 'correlation_id', None)
        
        # Try to get from thread local storage
        if not correlation_id:
            correlation_id = getattr(self._local, 'correlation_id', None)
        
        # Generate new one if needed and allowed
        if not correlation_id and self.auto_generate:
            correlation_id = str(uuid.uuid4())
            self._local.correlation_id = correlation_id
        
        # Add to record
        if correlation_id:
            record.correlation_id = correlation_id
        
        return True
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        self._local.correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear correlation ID for current thread."""
        if hasattr(self._local, 'correlation_id'):
            delattr(self._local, 'correlation_id')
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return getattr(self._local, 'correlation_id', None)


class ErrorContextFilter(logging.Filter):
    """
    Filter that adds error context information to log records.
    
    Automatically enriches log records with context from the error handling
    framework for better debugging and monitoring.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add error context to record."""
        error_context = get_current_context()
        
        if error_context:
            # Add context fields if not already present
            if not hasattr(record, 'component') and error_context.component:
                record.component = error_context.component
            
            if not hasattr(record, 'operation') and error_context.operation:
                record.operation = error_context.operation
            
            if not hasattr(record, 'user_id') and error_context.user_id:
                record.user_id = error_context.user_id
            
            if not hasattr(record, 'correlation_id') and error_context.correlation_id:
                record.correlation_id = error_context.correlation_id
            
            # Add performance data
            if error_context.duration_ms is not None:
                record.duration_ms = error_context.duration_ms
            
            # Add context as metadata
            if error_context.metadata and not hasattr(record, 'context'):
                record.context = error_context.metadata
        
        return True


class SeverityLevelFilter(logging.Filter):
    """
    Filter that maps MemMimic error severities to log levels.
    
    Ensures consistent severity mapping between the error handling
    framework and the logging system.
    """
    
    SEVERITY_TO_LEVEL = {
        ErrorSeverity.CRITICAL: logging.CRITICAL,
        ErrorSeverity.HIGH: logging.ERROR,
        ErrorSeverity.MEDIUM: logging.WARNING,
        ErrorSeverity.LOW: logging.INFO,
        ErrorSeverity.INFO: logging.DEBUG,
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Map error severity to log level."""
        # Check if there's an exception with severity
        if record.exc_info and record.exc_info[1]:
            exception = record.exc_info[1]
            if isinstance(exception, MemMimicError):
                # Map severity to log level
                mapped_level = self.SEVERITY_TO_LEVEL.get(exception.severity, record.levelno)
                record.levelno = mapped_level
                record.levelname = logging.getLevelName(mapped_level)
                record.severity = exception.severity.value
        
        # Check record extras for severity
        elif hasattr(record, 'severity'):
            try:
                severity = ErrorSeverity(record.severity)
                mapped_level = self.SEVERITY_TO_LEVEL.get(severity, record.levelno)
                record.levelno = mapped_level
                record.levelname = logging.getLevelName(mapped_level)
            except (ValueError, TypeError):
                pass  # Invalid severity, keep original level
        
        return True


class ErrorLoggingHandler(logging.Handler):
    """
    Custom handler for error-specific logging with enhanced formatting.
    
    Provides specialized handling for MemMimic errors with automatic
    context enrichment and structured output.
    """
    
    def __init__(
        self,
        level: Union[int, str] = logging.ERROR,
        include_stack_trace: bool = True,
        include_context: bool = True,
        max_message_length: int = 1000
    ):
        super().__init__(level)
        self.include_stack_trace = include_stack_trace
        self.include_context = include_context
        self.max_message_length = max_message_length
        
        # Set up structured formatter
        self.setFormatter(StructuredFormatter(
            include_correlation_ids=True,
            include_performance=True,
            include_context=include_context
        ))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with error-specific enhancements."""
        try:
            # Enhance record for MemMimic errors
            if record.exc_info and record.exc_info[1]:
                exception = record.exc_info[1]
                if isinstance(exception, MemMimicError):
                    self._enhance_error_record(record, exception)
            
            # Truncate long messages
            if len(record.msg) > self.max_message_length:
                record.msg = record.msg[:self.max_message_length] + "... [TRUNCATED]"
            
            # Let parent handler emit
            super().emit(record)
            
        except Exception:
            self.handleError(record)
    
    def _enhance_error_record(self, record: logging.LogRecord, error: MemMimicError) -> None:
        """Enhance log record with MemMimic error information."""
        record.error_id = error.error_id
        record.error_code = error.error_code
        record.severity = error.severity.value
        record.component = error.component
        record.operation = error.operation
        record.correlation_id = error.correlation_id
        
        # Add error context as record context
        if error.context and self.include_context:
            record.context = error.context


# =============================================================================
# Configuration and setup functions
# =============================================================================

def configure_error_logging(
    level: Union[int, str] = logging.INFO,
    format_type: str = "structured",  # "structured" or "simple"
    include_correlation_ids: bool = True,
    include_performance: bool = True,
    include_context: bool = True,
    output_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_indent: Optional[int] = None
) -> None:
    """
    Configure error logging for the MemMimic framework.
    
    Args:
        level: Minimum log level to capture
        format_type: "structured" for JSON or "simple" for text
        include_correlation_ids: Whether to include correlation IDs
        include_performance: Whether to include performance metrics
        include_context: Whether to include error context
        output_file: Optional file path for log output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        json_indent: JSON indentation for pretty printing (None for compact)
        
    Example:
        >>> configure_error_logging(
        ...     level=logging.WARNING,
        ...     format_type="structured",
        ...     output_file="/var/log/memmimic/errors.log"
        ... )
    """
    # Get root logger for MemMimic
    logger = logging.getLogger('memmimic')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if format_type == "structured":
        formatter = StructuredFormatter(
            include_correlation_ids=include_correlation_ids,
            include_performance=include_performance,
            include_context=include_context,
            json_indent=json_indent
        )
    else:
        # Simple text formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add filters
    if include_correlation_ids:
        console_handler.addFilter(CorrelationIDFilter())
    console_handler.addFilter(ErrorContextFilter())
    console_handler.addFilter(SeverityLevelFilter())
    
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if output_file:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            output_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        
        # Add same filters
        if include_correlation_ids:
            file_handler.addFilter(CorrelationIDFilter())
        file_handler.addFilter(ErrorContextFilter())
        file_handler.addFilter(SeverityLevelFilter())
        
        logger.addHandler(file_handler)
    
    # Mark as configured
    configure_error_logging._configured = True


def get_error_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for error handling.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_error_logger(__name__)
        >>> logger.error("Something went wrong", extra={"user_id": "123"})
    """
    return logging.getLogger(f'memmimic.{name}')


def setup_error_handler_logging() -> None:
    """
    Set up logging specifically for the error handlers module.
    
    Configures structured logging with correlation IDs and context
    for the error handling framework components.
    """
    # Configure if not already done
    if not hasattr(configure_error_logging, '_configured'):
        configure_error_logging(
            level=logging.INFO,
            format_type="structured",
            include_correlation_ids=True,
            include_performance=True,
            include_context=True
        )


def create_logger_with_context(
    name: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> logging.LoggerAdapter:
    """
    Create a logger adapter with pre-configured context.
    
    Args:
        name: Logger name
        component: Component name for context
        operation: Operation name for context
        correlation_id: Correlation ID for request tracking
        
    Returns:
        Logger adapter with context
        
    Example:
        >>> logger = create_logger_with_context(
        ...     __name__,
        ...     component="memory_manager",
        ...     operation="store_memory"
        ... )
        >>> logger.info("Memory stored successfully")
    """
    base_logger = get_error_logger(name)
    
    extra = {}
    if component:
        extra['component'] = component
    if operation:
        extra['operation'] = operation
    if correlation_id:
        extra['correlation_id'] = correlation_id
    
    return logging.LoggerAdapter(base_logger, extra)


def log_error_with_context(
    logger: logging.Logger,
    level: Union[int, str],
    message: str,
    error: Optional[Exception] = None,
    **context
) -> None:
    """
    Log an error with rich context information.
    
    Args:
        logger: Logger instance to use
        level: Log level
        message: Log message
        error: Optional exception to include
        **context: Additional context to include
        
    Example:
        >>> log_error_with_context(
        ...     logger,
        ...     logging.ERROR,
        ...     "Database operation failed",
        ...     error=db_error,
        ...     user_id="123",
        ...     operation="store_memory"
        ... )
    """
    # Build extra context
    extra = dict(context)
    
    # Add error information if provided
    if error:
        if isinstance(error, MemMimicError):
            extra.update({
                'error_id': error.error_id,
                'error_code': error.error_code,
                'severity': error.severity.value
            })
        extra['error_type'] = type(error).__name__
    
    # Log with context
    if error:
        logger.log(level, message, exc_info=True, extra=extra)
    else:
        logger.log(level, message, extra=extra)


# =============================================================================
# Integration utilities
# =============================================================================

def enhance_error_with_logging(error: MemMimicError, logger: Optional[logging.Logger] = None) -> MemMimicError:
    """
    Enhance an error with automatic logging.
    
    Args:
        error: MemMimic error to enhance
        logger: Optional logger (uses default if not provided)
        
    Returns:
        The same error instance (for chaining)
        
    Example:
        >>> error = MemoryStorageError("Failed to store")
        >>> enhance_error_with_logging(error, logger)
    """
    if not logger:
        logger = get_error_logger('errors')
    
    # Log the error with full context
    log_error_with_context(
        logger,
        logging.ERROR,
        f"[{error.error_code}] {error.message}",
        error=error,
        component=error.component,
        operation=error.operation,
        correlation_id=error.correlation_id
    )
    
    return error


# Initialize logging on import
setup_error_handler_logging()