"""
Error context management for the MemMimic error handling framework.

Provides structured error context tracking, correlation ID management,
and contextual information preservation across operations.
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager

from .exceptions import ErrorSeverity


@dataclass
class ErrorContext:
    """
    Structured error context information.
    
    Contains all relevant information about the execution context when an error occurs,
    enabling rich debugging and monitoring capabilities.
    """
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    # Performance context
    start_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Call stack context
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    
    # Business context
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            "user_id": self.user_id,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "severity": self.severity.value,
            "performance": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "duration_ms": self.duration_ms,
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent,
            },
            "call_stack": {
                "function_name": self.function_name,
                "module_name": self.module_name,
                "file_path": self.file_path,
                "line_number": self.line_number,
            },
            "business_context": {
                "entity_type": self.entity_type,
                "entity_id": self.entity_id,
                "action": self.action,
            }
        }
    
    def add_metadata(self, key: str, value: Any) -> 'ErrorContext':
        """Add metadata to the context."""
        self.metadata[key] = value
        return self
    
    def set_performance_data(
        self,
        duration_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None
    ) -> 'ErrorContext':
        """Set performance context data."""
        if duration_ms is not None:
            self.duration_ms = duration_ms
        if memory_usage_mb is not None:
            self.memory_usage_mb = memory_usage_mb
        if cpu_usage_percent is not None:
            self.cpu_usage_percent = cpu_usage_percent
        return self
    
    def set_call_stack_info(
        self,
        function_name: Optional[str] = None,
        module_name: Optional[str] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> 'ErrorContext':
        """Set call stack context information."""
        if function_name is not None:
            self.function_name = function_name
        if module_name is not None:
            self.module_name = module_name
        if file_path is not None:
            self.file_path = file_path
        if line_number is not None:
            self.line_number = line_number
        return self
    
    def set_business_context(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None
    ) -> 'ErrorContext':
        """Set business context information."""
        if entity_type is not None:
            self.entity_type = entity_type
        if entity_id is not None:
            self.entity_id = entity_id
        if action is not None:
            self.action = action
        return self
    
    def clone(self, **overrides) -> 'ErrorContext':
        """Create a copy of the context with optional overrides."""
        # Create a new context with current values
        new_context = ErrorContext(
            error_id=str(uuid.uuid4()),  # Always new error ID
            timestamp=datetime.now(),     # Always new timestamp
            operation=self.operation,
            component=self.component,
            user_id=self.user_id,
            correlation_id=self.correlation_id,  # Keep same correlation ID
            session_id=self.session_id,
            request_id=self.request_id,
            metadata=self.metadata.copy(),
            severity=self.severity,
            start_time=self.start_time,
            duration_ms=self.duration_ms,
            memory_usage_mb=self.memory_usage_mb,
            cpu_usage_percent=self.cpu_usage_percent,
            function_name=self.function_name,
            module_name=self.module_name,
            file_path=self.file_path,
            line_number=self.line_number,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            action=self.action,
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)
        
        return new_context


class ErrorContextManager:
    """
    Thread-safe error context manager.
    
    Maintains error context state per thread and provides utilities for
    context creation, retrieval, and management.
    """
    
    def __init__(self):
        self._local = threading.local()
        self._global_context: Dict[str, Any] = {}
    
    def set_global_context(self, **context: Any) -> None:
        """Set global context that applies to all threads."""
        self._global_context.update(context)
    
    def get_global_context(self) -> Dict[str, Any]:
        """Get the global context."""
        return self._global_context.copy()
    
    def set_current_context(self, context: ErrorContext) -> None:
        """Set the current error context for this thread."""
        self._local.context = context
    
    def get_current_context(self) -> Optional[ErrorContext]:
        """Get the current error context for this thread."""
        return getattr(self._local, 'context', None)
    
    def clear_current_context(self) -> None:
        """Clear the current error context for this thread."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def create_context(
        self,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **kwargs
    ) -> ErrorContext:
        """
        Create a new error context with global context merged in.
        
        Args:
            operation: Operation being performed
            component: Component where operation is happening
            **kwargs: Additional context parameters
            
        Returns:
            New ErrorContext instance
        """
        # Start with global context
        context_data = self._global_context.copy()
        
        # Add specific parameters
        if operation:
            context_data['operation'] = operation
        if component:
            context_data['component'] = component
        
        # Add any additional parameters
        context_data.update(kwargs)
        
        # Extract metadata for the metadata field
        metadata = context_data.pop('metadata', {})
        
        # Create context
        context = ErrorContext(**context_data)
        context.metadata.update(metadata)
        
        return context
    
    def with_context(
        self,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **kwargs
    ) -> ContextManager[ErrorContext]:
        """
        Context manager for error context.
        
        Args:
            operation: Operation being performed
            component: Component where operation is happening
            **kwargs: Additional context parameters
            
        Returns:
            Context manager that sets and clears error context
        """
        return _ErrorContextContextManager(self, operation, component, **kwargs)


class _ErrorContextContextManager:
    """Context manager implementation for error context."""
    
    def __init__(
        self,
        manager: ErrorContextManager,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **kwargs
    ):
        self.manager = manager
        self.operation = operation
        self.component = component
        self.kwargs = kwargs
        self.context: Optional[ErrorContext] = None
        self.previous_context: Optional[ErrorContext] = None
    
    def __enter__(self) -> ErrorContext:
        # Save previous context
        self.previous_context = self.manager.get_current_context()
        
        # Create new context
        self.context = self.manager.create_context(
            operation=self.operation,
            component=self.component,
            **self.kwargs
        )
        
        # Set performance start time
        self.context.start_time = datetime.now()
        
        # Set as current context
        self.manager.set_current_context(self.context)
        
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration if start time was set
        if self.context and self.context.start_time:
            duration = (datetime.now() - self.context.start_time).total_seconds() * 1000
            self.context.duration_ms = duration
        
        # Restore previous context
        if self.previous_context:
            self.manager.set_current_context(self.previous_context)
        else:
            self.manager.clear_current_context()


# Global context manager instance
_context_manager = ErrorContextManager()


# =============================================================================
# Public API functions
# =============================================================================

def create_error_context(
    operation: Optional[str] = None,
    component: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **metadata
) -> ErrorContext:
    """
    Create a new error context with the specified parameters.
    
    Args:
        operation: Operation being performed
        component: Component where operation is happening
        correlation_id: Correlation ID for tracking related operations
        user_id: User associated with the operation
        severity: Severity level of potential errors
        **metadata: Additional metadata to include
        
    Returns:
        New ErrorContext instance
        
    Example:
        >>> context = create_error_context(
        ...     operation="memory_retrieval",
        ...     component="active_manager",
        ...     user_id="user_123",
        ...     memory_id="mem_456"
        ... )
    """
    context = _context_manager.create_context(
        operation=operation,
        component=component,
        correlation_id=correlation_id,
        user_id=user_id,
        severity=severity
    )
    
    # Add metadata
    context.metadata.update(metadata)
    
    return context


def get_current_context() -> Optional[ErrorContext]:
    """
    Get the current error context for this thread.
    
    Returns:
        Current ErrorContext if set, None otherwise
        
    Example:
        >>> current = get_current_context()
        >>> if current:
        ...     print(f"Current operation: {current.operation}")
    """
    return _context_manager.get_current_context()


def set_current_context(context: ErrorContext) -> None:
    """
    Set the current error context for this thread.
    
    Args:
        context: ErrorContext to set as current
        
    Example:
        >>> context = create_error_context(operation="test")
        >>> set_current_context(context)
    """
    _context_manager.set_current_context(context)


def clear_current_context() -> None:
    """
    Clear the current error context for this thread.
    
    Example:
        >>> clear_current_context()
    """
    _context_manager.clear_current_context()


def set_global_context(**context: Any) -> None:
    """
    Set global context that applies to all threads.
    
    Args:
        **context: Context parameters to set globally
        
    Example:
        >>> set_global_context(
        ...     service_name="memmimic",
        ...     version="2.0.0",
        ...     environment="production"
        ... )
    """
    _context_manager.set_global_context(**context)


def get_global_context() -> Dict[str, Any]:
    """
    Get the global context that applies to all threads.
    
    Returns:
        Dictionary of global context parameters
        
    Example:
        >>> global_ctx = get_global_context()
        >>> print(f"Service: {global_ctx.get('service_name')}")
    """
    return _context_manager.get_global_context()


@contextmanager
def with_error_context(
    operation: Optional[str] = None,
    component: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **metadata
) -> ContextManager[ErrorContext]:
    """
    Context manager for error context that automatically sets and clears context.
    
    Args:
        operation: Operation being performed
        component: Component where operation is happening
        correlation_id: Correlation ID for tracking related operations
        user_id: User associated with the operation
        severity: Severity level of potential errors
        **metadata: Additional metadata to include
        
    Yields:
        ErrorContext instance that is automatically managed
        
    Example:
        >>> with with_error_context(
        ...     operation="memory_storage",
        ...     component="active_manager",
        ...     memory_id="mem_123"
        ... ) as ctx:
        ...     # Perform operation
        ...     risky_operation()
        ...     # Context is automatically cleared
    """
    context = create_error_context(
        operation=operation,
        component=component,
        correlation_id=correlation_id,
        user_id=user_id,
        severity=severity,
        **metadata
    )
    
    # Set performance start time
    context.start_time = datetime.now()
    
    # Store previous context
    previous_context = get_current_context()
    
    try:
        # Set as current context
        set_current_context(context)
        yield context
    finally:
        # Calculate duration
        if context.start_time:
            duration = (datetime.now() - context.start_time).total_seconds() * 1000
            context.duration_ms = duration
        
        # Restore previous context
        if previous_context:
            set_current_context(previous_context)
        else:
            clear_current_context()


def add_context_metadata(key: str, value: Any) -> None:
    """
    Add metadata to the current error context.
    
    Args:
        key: Metadata key
        value: Metadata value
        
    Example:
        >>> with with_error_context(operation="test"):
        ...     add_context_metadata("step", "validation")
        ...     add_context_metadata("input_size", 1024)
    """
    context = get_current_context()
    if context:
        context.add_metadata(key, value)


def get_context_correlation_id() -> Optional[str]:
    """
    Get the correlation ID from the current context.
    
    Returns:
        Correlation ID if context is set, None otherwise
        
    Example:
        >>> correlation_id = get_context_correlation_id()
        >>> if correlation_id:
        ...     logger.info(f"Processing request {correlation_id}")
    """
    context = get_current_context()
    return context.correlation_id if context else None


def extract_call_stack_info() -> Dict[str, Any]:
    """
    Extract call stack information for error context.
    
    Returns:
        Dictionary containing call stack information
    """
    import inspect
    
    # Get the calling frame (skip this function and the caller)
    frame = inspect.currentframe()
    try:
        # Go up the stack to find the actual caller
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        
        if caller_frame:
            return {
                'function_name': caller_frame.f_code.co_name,
                'file_path': caller_frame.f_code.co_filename,
                'line_number': caller_frame.f_lineno,
                'module_name': caller_frame.f_globals.get('__name__', 'unknown')
            }
    finally:
        # Clean up frame references to prevent memory leaks
        del frame
    
    return {}


def auto_set_call_stack_context() -> None:
    """
    Automatically set call stack information in the current context.
    
    Example:
        >>> with with_error_context(operation="test"):
        ...     auto_set_call_stack_context()
        ...     # Current context now has call stack info
    """
    context = get_current_context()
    if context:
        stack_info = extract_call_stack_info()
        context.set_call_stack_info(**stack_info)