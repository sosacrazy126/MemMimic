"""
MemMimic Memory System Error Handling
Custom exception classes and error handling utilities for the memory system.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Type, Union


# Base Exception Classes
class MemoryError(Exception):
    """Base exception for all memory system errors"""
    pass


class MemoryStorageError(MemoryError):
    """Exception raised for storage-related errors"""
    pass


class MemoryRetrievalError(MemoryError):
    """Exception raised for retrieval-related errors"""
    pass


class DatabaseError(MemoryError):
    """Exception raised for database-related errors"""
    pass


class GovernanceConfigError(MemoryError):
    """Exception raised for governance configuration errors"""
    pass


# Error Context Management
@contextmanager
def with_error_context(
    operation: str,
    component: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Generator[None, None, None]:
    """
    Context manager for adding error context to exceptions
    
    Args:
        operation: The operation being performed
        component: The component where the operation is happening
        metadata: Additional metadata for error context
    """
    try:
        yield
    except Exception as e:
        # Enhance the exception with context information
        context_info = f"Operation: {operation}, Component: {component}"
        if metadata:
            context_info += f", Metadata: {metadata}"
        
        # Re-raise with enhanced message
        if hasattr(e, 'args') and e.args:
            enhanced_message = f"{e.args[0]} [{context_info}]"
            e.args = (enhanced_message,) + e.args[1:]
        else:
            e.args = (f"Error in {context_info}",)
        
        raise


# Error Handling Decorators
def handle_errors(
    catch: list = None,
    reraise: bool = False,
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Decorator for handling common errors in memory operations
    
    Args:
        catch: List of exception types to catch
        reraise: Whether to reraise caught exceptions
        default_return: Default value to return if exception is caught and not reraised
        log_errors: Whether to log caught exceptions
    """
    if catch is None:
        catch = [Exception]
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except tuple(catch) as e:
                if log_errors:
                    logger = get_error_logger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                else:
                    return default_return
        return wrapper
    return decorator


# Logging Utilities
_error_loggers = {}

def get_error_logger(component: str) -> logging.Logger:
    """
    Get or create an error logger for a specific component
    
    Args:
        component: The component name for the logger
        
    Returns:
        Logger instance for the component
    """
    if component not in _error_loggers:
        logger = logging.getLogger(f"memmimic.memory.{component}")
        logger.setLevel(logging.INFO)
        
        # Add console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        _error_loggers[component] = logger
    
    return _error_loggers[component]


# Error Classification and Severity
class ErrorSeverity:
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def classify_error(error: Exception) -> str:
    """
    Classify error severity based on exception type and context
    
    Args:
        error: The exception to classify
        
    Returns:
        Severity level string
    """
    if isinstance(error, DatabaseError):
        return ErrorSeverity.HIGH
    elif isinstance(error, MemoryStorageError):
        return ErrorSeverity.MEDIUM
    elif isinstance(error, MemoryRetrievalError):
        return ErrorSeverity.LOW
    elif isinstance(error, GovernanceConfigError):
        return ErrorSeverity.HIGH
    else:
        return ErrorSeverity.MEDIUM


# Recovery Strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can recover from the given error"""
        return False
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class DatabaseRetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for database connection issues"""
    
    def __init__(self, max_retries: int = 3, delay_seconds: float = 1.0):
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
    
    def can_recover(self, error: Exception) -> bool:
        return isinstance(error, DatabaseError)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        import asyncio
        
        for attempt in range(self.max_retries):
            try:
                # Wait before retry
                if attempt > 0:
                    await asyncio.sleep(self.delay_seconds * attempt)
                
                # Attempt recovery by re-establishing database connection
                operation = context.get('operation')
                if operation:
                    return await operation()
                
            except Exception as retry_error:
                if attempt == self.max_retries - 1:
                    raise retry_error
        
        raise error