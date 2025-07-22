"""
Error handling decorators and utilities for MemMimic.

Provides standardized error handling patterns through decorators and utility classes
for retry logic, circuit breakers, fallback mechanisms, and structured error logging.
"""

import asyncio
import functools
import logging
import time
import random
from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union, Tuple,
    TypeVar, ParamSpec, Awaitable
)
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .exceptions import (
    MemMimicError, ErrorSeverity, 
    NetworkError, TimeoutError, DatabaseError, ExternalAPIError
)
from .context import (
    create_error_context, get_current_context, with_error_context,
    add_context_metadata
)

# Type variables for generic decorators
P = ParamSpec('P')
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)


class BackoffStrategy(ABC):
    """Abstract base class for retry backoff strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff strategy with jitter."""
    
    def __init__(
        self, 
        initial_delay: float = 0.1,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add ±10% jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.0, delay)


class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy."""
    
    def __init__(self, delay: float = 1.0, max_delay: float = 30.0):
        self.delay = delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        return min(self.delay * attempt, self.max_delay)


class ConstantBackoff(BackoffStrategy):
    """Constant delay backoff strategy."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
    
    def get_delay(self, attempt: int) -> float:
        """Return constant delay."""
        return self.delay


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff: BackoffStrategy = field(default_factory=ExponentialBackoff)
    retriable_exceptions: Tuple[Type[Exception], ...] = (
        NetworkError, TimeoutError, DatabaseError, ExternalAPIError
    )
    stop_on_exceptions: Tuple[Type[Exception], ...] = ()
    
    # Global default policy
    _default: Optional['RetryPolicy'] = None
    
    @classmethod
    def set_default(cls, **kwargs) -> None:
        """Set global default retry policy."""
        cls._default = cls(**kwargs)
    
    @classmethod
    def get_default(cls) -> 'RetryPolicy':
        """Get global default retry policy."""
        return cls._default or cls()
    
    @classmethod
    def get_default_attempts(cls) -> int:
        """Get default max attempts for framework status."""
        return cls.get_default().max_attempts


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    # Global registry of circuit breakers
    _instances: Dict[str, 'CircuitBreaker'] = {}
    _default_threshold: int = 10
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 10,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        
        # Register instance
        CircuitBreaker._instances[name] = self
    
    @classmethod
    def get_or_create(
        cls,
        name: str,
        failure_threshold: int = 10,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3
    ) -> 'CircuitBreaker':
        """Get existing circuit breaker or create new one."""
        if name in cls._instances:
            return cls._instances[name]
        return cls(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold
        )
    
    @classmethod
    def set_default_threshold(cls, threshold: int) -> None:
        """Set default failure threshold."""
        cls._default_threshold = threshold
    
    @classmethod
    def get_active_count(cls) -> int:
        """Get count of active circuit breakers for framework status."""
        return len(cls._instances)
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self, exception: Exception) -> None:
        """Record failed operation."""
        if not isinstance(exception, self.expected_exception):
            return
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.failure_threshold):
            self.state = CircuitState.OPEN


class CircuitBreakerError(MemMimicError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str, **kwargs):
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open",
            error_code="CIRCUIT_BREAKER_OPEN",
            component="error_handling",
            operation="circuit_breaker_check",
            severity=ErrorSeverity.HIGH,
            context={"circuit_name": circuit_name},
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling coordinator."""
    
    def __init__(self):
        self.error_collectors: List[Callable[[Exception, Dict[str, Any]], None]] = []
        self.global_fallbacks: Dict[Type[Exception], Callable] = {}
    
    def register_collector(self, collector: Callable[[Exception, Dict[str, Any]], None]) -> None:
        """Register error collector for monitoring."""
        self.error_collectors.append(collector)
    
    def register_fallback(self, exception_type: Type[Exception], fallback: Callable) -> None:
        """Register global fallback for exception type."""
        self.global_fallbacks[exception_type] = fallback
    
    def handle_error(
        self, 
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None,
        fallback: Optional[Callable] = None
    ) -> Any:
        """Handle error with collectors and fallbacks."""
        error_context = context or {}
        
        # Collect error for monitoring
        for collector in self.error_collectors:
            try:
                collector(exception, error_context)
            except Exception as e:
                logger.warning(f"Error collector failed: {e}")
        
        # Try specific fallback first, then global fallback
        if fallback:
            return fallback()
        
        for exc_type, global_fallback in self.global_fallbacks.items():
            if isinstance(exception, exc_type):
                return global_fallback()
        
        # Re-raise if no fallback
        raise exception


# Global error handler instance
_error_handler = ErrorHandler()


def handle_errors(
    catch: Optional[List[Type[Exception]]] = None,
    ignore: Optional[List[Type[Exception]]] = None,
    log_level: Union[int, str] = logging.ERROR,
    include_stack_trace: bool = True,
    reraise: bool = True,
    fallback: Optional[Callable] = None,
    context: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[P, T]], Callable[P, Optional[T]]]:
    """
    Decorator for standardized error handling with logging and context preservation.
    
    Args:
        catch: List of exception types to catch (default: all exceptions)
        ignore: List of exception types to ignore (don't log or handle)
        log_level: Logging level for caught exceptions
        include_stack_trace: Whether to include stack trace in logs
        reraise: Whether to re-raise exceptions after handling
        fallback: Fallback function to call on errors
        context: Additional context to include with errors
        
    Returns:
        Decorated function with error handling
        
    Example:
        @handle_errors(
            catch=[DatabaseError, NetworkError],
            log_level=logging.WARNING,
            fallback=lambda: None
        )
        def risky_operation():
            # operation that might fail
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            operation_context = context or {}
            operation_context.update({
                "function": func.__name__,
                "module": func.__module__,
            })
            
            try:
                with with_error_context(
                    operation=func.__name__,
                    component=func.__module__,
                    **operation_context
                ):
                    return func(*args, **kwargs)
                    
            except Exception as e:
                # Check if we should ignore this exception
                if ignore and isinstance(e, tuple(ignore)):
                    raise
                
                # Check if we should catch this exception
                if catch and not isinstance(e, tuple(catch)):
                    raise
                
                # Convert to MemMimicError if it's not already
                if not isinstance(e, MemMimicError):
                    current_context = get_current_context()
                    context_dict = current_context.to_dict() if current_context else {}
                    context_dict.update(operation_context)
                    
                    handled_error = MemMimicError(
                        message=f"Error in {func.__name__}: {str(e)}",
                        error_code=f"{type(e).__name__}_IN_{func.__name__.upper()}",
                        context=context_dict,
                        component=func.__module__,
                        operation=func.__name__
                    )
                else:
                    handled_error = e
                
                # Log the error
                log_message = f"Error in {func.__name__}: {handled_error.message}"
                extra = {
                    "error_id": handled_error.error_id,
                    "error_code": handled_error.error_code,
                    "context": handled_error.context
                }
                
                if include_stack_trace:
                    logger.log(log_level, log_message, exc_info=True, extra=extra)
                else:
                    logger.log(log_level, log_message, extra=extra)
                
                # Handle with global error handler
                try:
                    result = _error_handler.handle_error(e, operation_context, fallback)  # Use original exception for type matching
                    if not reraise:
                        return result
                except Exception:
                    pass  # Continue to reraise original
                
                # Re-raise or return fallback result
                if reraise:
                    raise handled_error
                elif fallback:
                    return fallback()
                
                return None
                
        return wrapper
    return decorator


def retry(
    policy: Optional[RetryPolicy] = None,
    max_attempts: Optional[int] = None,
    backoff: Optional[BackoffStrategy] = None,
    retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for automatic retry with configurable policies.
    
    Args:
        policy: Complete retry policy (overrides other parameters)
        max_attempts: Maximum number of attempts
        backoff: Backoff strategy for delays
        retriable_exceptions: Tuple of exceptions that should trigger retries
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry(
            max_attempts=5,
            backoff=ExponentialBackoff(initial_delay=0.5),
            retriable_exceptions=(NetworkError, TimeoutError)
        )
        def network_operation():
            # operation that might need retries
            pass
    """
    # Use provided policy or create from parameters
    if policy is None:
        policy = RetryPolicy(
            max_attempts=max_attempts or RetryPolicy.get_default().max_attempts,
            backoff=backoff or RetryPolicy.get_default().backoff,
            retriable_exceptions=retriable_exceptions or RetryPolicy.get_default().retriable_exceptions
        )
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(policy.max_attempts):
                try:
                    # Add retry context
                    add_context_metadata("retry_attempt", attempt + 1)
                    add_context_metadata("max_attempts", policy.max_attempts)
                    
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                        )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should stop on this exception
                    if policy.stop_on_exceptions and isinstance(e, policy.stop_on_exceptions):
                        logger.debug(f"Stop exception {type(e).__name__} encountered, not retrying")
                        raise
                    
                    # Check if this exception is retriable
                    if not isinstance(e, policy.retriable_exceptions):
                        logger.debug(f"Non-retriable exception {type(e).__name__}, not retrying")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == policy.max_attempts - 1:
                        logger.warning(
                            f"Function {func.__name__} failed after {policy.max_attempts} attempts"
                        )
                        break
                    
                    # Calculate delay and wait
                    delay = policy.backoff.get_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
            
            # All attempts failed
            if last_exception:
                raise last_exception
            
            # This shouldn't happen, but just in case
            raise RuntimeError(f"Function {func.__name__} failed without exception")
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(policy.max_attempts):
                try:
                    # Add retry context
                    add_context_metadata("retry_attempt", attempt + 1)
                    add_context_metadata("max_attempts", policy.max_attempts)
                    
                    result = func(*args, **kwargs)
                    
                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                        )
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should stop on this exception
                    if policy.stop_on_exceptions and isinstance(e, policy.stop_on_exceptions):
                        logger.debug(f"Stop exception {type(e).__name__} encountered, not retrying")
                        raise
                    
                    # Check if this exception is retriable
                    if not isinstance(e, policy.retriable_exceptions):
                        logger.debug(f"Non-retriable exception {type(e).__name__}, not retrying")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == policy.max_attempts - 1:
                        logger.warning(
                            f"Function {func.__name__} failed after {policy.max_attempts} attempts"
                        )
                        break
                    
                    # Calculate delay and wait
                    delay = policy.backoff.get_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    
                    if delay > 0:
                        time.sleep(delay)
            
            # All attempts failed
            if last_exception:
                raise last_exception
            
            # This shouldn't happen, but just in case
            raise RuntimeError(f"Function {func.__name__} failed without exception")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def circuit_breaker(
    name: str,
    failure_threshold: int = 10,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for circuit breaker pattern.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Type of exception that triggers circuit breaker
        
    Returns:
        Decorated function with circuit breaker logic
        
    Example:
        @circuit_breaker(
            name="external_api",
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=NetworkError
        )
        def call_external_api():
            # external API call
            pass
    """
    breaker = CircuitBreaker.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Check if request should be allowed
            if not breaker.should_allow_request():
                raise CircuitBreakerError(name)
            
            try:
                # Add circuit breaker context
                add_context_metadata("circuit_breaker", name)
                add_context_metadata("circuit_state", breaker.state.value)
                
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
                
            except Exception as e:
                breaker.record_failure(e)
                raise
                
        return wrapper
    return decorator


def fallback(
    fallback_func: Callable[..., T],
    catch: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for providing fallback behavior on errors.
    
    Args:
        fallback_func: Function to call as fallback
        catch: List of exception types that trigger fallback
        
    Returns:
        Decorated function with fallback logic
        
    Example:
        @fallback(
            fallback_func=lambda: "default_value",
            catch=[NetworkError, TimeoutError]
        )
        def get_remote_data():
            # operation that might fail
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should use fallback for this exception
                if catch and not isinstance(e, tuple(catch)):
                    raise
                
                logger.info(
                    f"Using fallback for {func.__name__} due to {type(e).__name__}: {str(e)}"
                )
                
                # Add fallback context
                add_context_metadata("fallback_triggered", True)
                add_context_metadata("original_error", str(e))
                
                # Call fallback function with same arguments
                try:
                    return fallback_func(*args, **kwargs)
                except TypeError:
                    # Fallback function might not accept same arguments
                    return fallback_func()
                    
        return wrapper
    return decorator


def log_errors(
    log_level: Union[int, str] = logging.ERROR,
    include_args: bool = False,
    include_stack_trace: bool = True
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for automatic error logging.
    
    Args:
        log_level: Logging level for errors
        include_args: Whether to include function arguments in logs
        include_stack_trace: Whether to include stack trace
        
    Returns:
        Decorated function with error logging
        
    Example:
        @log_errors(log_level=logging.WARNING, include_args=True)
        def important_operation(data):
            # operation that should be logged on errors
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Prepare log message
                log_message = f"Error in {func.__name__}: {str(e)}"
                
                # Prepare extra context
                extra = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "error_type": type(e).__name__
                }
                
                if include_args:
                    extra["args"] = str(args) if args else None
                    extra["kwargs"] = str(kwargs) if kwargs else None
                
                # Log with or without stack trace
                if include_stack_trace:
                    logger.log(log_level, log_message, exc_info=True, extra=extra)
                else:
                    logger.log(log_level, log_message, extra=extra)
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def combine_decorators(*decorators) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Utility function to combine multiple decorators in the correct order.
    
    Applies decorators from outside to inside (last to first in argument list).
    This is useful for ensuring proper ordering of error handling decorators.
    
    Args:
        *decorators: Decorators to combine
        
    Returns:
        Combined decorator function
        
    Example:
        @combine_decorators(
            handle_errors(catch=[DatabaseError]),
            retry(max_attempts=3),
            circuit_breaker("database", failure_threshold=5),
            log_errors()
        )
        def database_operation():
            # operation with full error handling stack
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Apply decorators in reverse order (outside to inside)
        result = func
        for dec in reversed(decorators):
            result = dec(result)
        return result
    return decorator


# =============================================================================
# Utility functions for error handler management
# =============================================================================

def register_error_collector(collector: Callable[[Exception, Dict[str, Any]], None]) -> None:
    """Register global error collector for monitoring."""
    _error_handler.register_collector(collector)


def register_global_fallback(exception_type: Type[Exception], fallback: Callable) -> None:
    """Register global fallback for exception type."""
    _error_handler.register_fallback(exception_type, fallback)


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker instance by name."""
    return CircuitBreaker._instances.get(name)


def reset_circuit_breaker(name: str) -> bool:
    """Reset circuit breaker to closed state."""
    breaker = get_circuit_breaker(name)
    if breaker:
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.success_count = 0
        breaker.last_failure_time = None
        return True
    return False


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return CircuitBreaker._instances.copy()