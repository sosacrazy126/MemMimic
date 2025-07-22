"""
Security Validation Decorators

Function decorators for automatic input validation, output sanitization,
rate limiting, and security auditing.
"""

import functools
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, deque
import logging

from .validation import get_input_validator, ValidationError, SecurityValidationError
from .sanitization import get_security_sanitizer
from .audit import get_security_audit_logger, SecurityEvent

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, limit: int, window: int, retry_after: int):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(message)


class SecurityDecoratorError(Exception):
    """Exception raised by security decorators."""
    pass


# Rate limiting storage
_rate_limit_storage = defaultdict(lambda: deque())
_rate_limit_lock = threading.Lock()


def validate_input(
    validation_type: str = "auto",
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    log_violations: bool = True
):
    """
    Decorator for automatic input validation.
    
    Args:
        validation_type: Type of validation ("memory", "tale", "query", "json", "auto")
        schema: Custom validation schema
        strict: Whether to raise exceptions on validation failures
        log_violations: Whether to log security violations
    
    Example:
        @validate_input(validation_type="memory", strict=True)
        def store_memory(content: str, memory_type: str = "interaction"):
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = get_input_validator()
            audit_logger = get_security_audit_logger()
            
            try:
                # Extract arguments for validation based on function signature
                validated_args, validated_kwargs = _validate_function_inputs(
                    func, args, kwargs, validation_type, validator, strict
                )
                
                # Log security validation if requested
                if log_violations:
                    audit_logger.log_security_event(SecurityEvent(
                        event_type="input_validation",
                        component="decorator",
                        function_name=func.__name__,
                        metadata={"validation_type": validation_type, "strict": strict}
                    ))
                
                # Call original function with validated inputs
                return func(*validated_args, **validated_kwargs)
                
            except (ValidationError, SecurityValidationError) as e:
                if log_violations:
                    audit_logger.log_security_event(SecurityEvent(
                        event_type="validation_failure",
                        component="decorator",
                        function_name=func.__name__,
                        severity="HIGH" if isinstance(e, SecurityValidationError) else "MEDIUM",
                        details=str(e),
                        metadata={
                            "field": getattr(e, 'field', None),
                            "validation_type": validation_type
                        }
                    ))
                
                if strict:
                    raise SecurityDecoratorError(f"Input validation failed in {func.__name__}: {str(e)}")
                else:
                    logger.warning(f"Input validation warning in {func.__name__}: {str(e)}")
                    return func(*args, **kwargs)  # Continue with original inputs
                    
            except Exception as e:
                logger.error(f"Input validation error in {func.__name__}: {str(e)}")
                if strict:
                    raise SecurityDecoratorError(f"Input validation error in {func.__name__}: {str(e)}")
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_output(
    sanitization_type: str = "memory",
    preserve_structure: bool = True,
    log_sanitization: bool = True
):
    """
    Decorator for automatic output sanitization.
    
    Args:
        sanitization_type: Type of sanitization ("memory", "json", "html")
        preserve_structure: Whether to preserve data structure
        log_sanitization: Whether to log sanitization actions
    
    Example:
        @sanitize_output(sanitization_type="memory")
        def get_memory_content(memory_id: str):
            # Function returns content that will be sanitized
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sanitizer = get_security_sanitizer()
            audit_logger = get_security_audit_logger()
            
            try:
                # Call original function
                result = func(*args, **kwargs)
                
                # Sanitize output based on type
                sanitized_result = _sanitize_function_output(
                    result, sanitization_type, sanitizer, preserve_structure
                )
                
                # Log sanitization if requested
                if log_sanitization:
                    audit_logger.log_security_event(SecurityEvent(
                        event_type="output_sanitization",
                        component="decorator",
                        function_name=func.__name__,
                        metadata={
                            "sanitization_type": sanitization_type,
                            "preserve_structure": preserve_structure
                        }
                    ))
                
                return sanitized_result
                
            except Exception as e:
                logger.error(f"Output sanitization error in {func.__name__}: {str(e)}")
                return result  # Return original result on sanitization error
        
        return wrapper
    return decorator


def rate_limit(
    max_calls: int = 100,
    window_seconds: int = 60,
    per_user: bool = False,
    key_func: Optional[Callable] = None
):
    """
    Decorator for rate limiting function calls.
    
    Args:
        max_calls: Maximum number of calls allowed in the window
        window_seconds: Time window in seconds
        per_user: Whether to apply rate limiting per user (requires user identification)
        key_func: Custom function to generate rate limiting key
    
    Example:
        @rate_limit(max_calls=10, window_seconds=60, per_user=True)
        def expensive_operation(user_id: str, data: str):
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limiting key
            if key_func:
                rate_key = key_func(*args, **kwargs)
            elif per_user:
                # Try to extract user identifier from arguments
                rate_key = _extract_user_key(func, args, kwargs)
            else:
                # Global rate limiting for this function
                rate_key = f"global:{func.__name__}"
            
            current_time = time.time()
            
            with _rate_limit_lock:
                # Get request history for this key
                requests = _rate_limit_storage[rate_key]
                
                # Remove old requests outside the window
                while requests and requests[0] < current_time - window_seconds:
                    requests.popleft()
                
                # Check if rate limit exceeded
                if len(requests) >= max_calls:
                    oldest_request = requests[0]
                    retry_after = int(oldest_request + window_seconds - current_time)
                    
                    # Log rate limit violation
                    audit_logger = get_security_audit_logger()
                    audit_logger.log_security_event(SecurityEvent(
                        event_type="rate_limit_exceeded",
                        component="decorator",
                        function_name=func.__name__,
                        severity="MEDIUM",
                        details=f"Rate limit exceeded for key: {rate_key}",
                        metadata={
                            "max_calls": max_calls,
                            "window_seconds": window_seconds,
                            "current_calls": len(requests),
                            "retry_after": retry_after
                        }
                    ))
                    
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {func.__name__}. "
                        f"Max {max_calls} calls per {window_seconds} seconds.",
                        max_calls, window_seconds, retry_after
                    )
                
                # Add current request
                requests.append(current_time)
            
            # Call original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def audit_security(
    event_type: str = "function_call",
    log_inputs: bool = False,
    log_outputs: bool = False,
    sensitive_params: List[str] = None
):
    """
    Decorator for security audit logging.
    
    Args:
        event_type: Type of security event
        log_inputs: Whether to log function inputs
        log_outputs: Whether to log function outputs
        sensitive_params: List of parameter names to mask in logs
    
    Example:
        @audit_security(event_type="memory_access", log_inputs=True, 
                        sensitive_params=["password", "api_key"])
        def access_memory(user_id: str, password: str):
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_security_audit_logger()
            start_time = time.time()
            
            try:
                # Prepare audit metadata
                metadata = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "start_time": start_time
                }
                
                # Log inputs if requested (with sensitive parameter masking)
                if log_inputs:
                    safe_inputs = _mask_sensitive_params(
                        func, args, kwargs, sensitive_params or []
                    )
                    metadata["inputs"] = safe_inputs
                
                # Call original function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                metadata["execution_time_ms"] = execution_time * 1000
                
                # Log outputs if requested
                if log_outputs:
                    metadata["output_type"] = type(result).__name__
                    if isinstance(result, (str, int, float, bool)) and len(str(result)) < 200:
                        metadata["output_sample"] = str(result)[:200]
                
                # Log successful execution
                audit_logger.log_security_event(SecurityEvent(
                    event_type=event_type,
                    component="decorator",
                    function_name=func.__name__,
                    details=f"Function {func.__name__} executed successfully",
                    metadata=metadata
                ))
                
                return result
                
            except Exception as e:
                # Log exception
                execution_time = time.time() - start_time
                audit_logger.log_security_event(SecurityEvent(
                    event_type=f"{event_type}_error",
                    component="decorator",
                    function_name=func.__name__,
                    severity="HIGH",
                    details=f"Function {func.__name__} failed: {str(e)}",
                    metadata={
                        "function": func.__name__,
                        "module": func.__module__,
                        "execution_time_ms": execution_time * 1000,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                ))
                raise
        
        return wrapper
    return decorator


# Specialized decorators for common use cases


def validate_memory_content(strict: bool = True):
    """Specialized decorator for memory content validation."""
    return validate_input(validation_type="memory", strict=strict)


def validate_tale_input(strict: bool = True):
    """Specialized decorator for tale input validation."""
    return validate_input(validation_type="tale", strict=strict)


def validate_query_input(strict: bool = True):
    """Specialized decorator for query input validation."""
    return validate_input(validation_type="query", strict=strict)


# Helper functions

def _validate_function_inputs(
    func: Callable, args: tuple, kwargs: dict, 
    validation_type: str, validator, strict: bool
) -> tuple:
    """Validate function inputs based on validation type."""
    import inspect
    
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    validated_args = list(args)
    validated_kwargs = dict(kwargs)
    
    # Apply validation based on type
    if validation_type in ["memory", "auto"]:
        # Look for content parameter
        if 'content' in bound_args.arguments:
            content = bound_args.arguments['content']
            memory_type = bound_args.arguments.get('memory_type', 'interaction')
            result = validator.validate_memory_content(content, memory_type)
            
            if not result.is_valid and strict:
                raise ValidationError("Memory content validation failed", 
                                    field="content", validation_type="memory")
            
            # Update with sanitized content
            if 'content' in kwargs:
                validated_kwargs['content'] = result.sanitized_value
            else:
                # Find content in args
                param_names = list(sig.parameters.keys())
                if 'content' in param_names:
                    content_idx = param_names.index('content')
                    if content_idx < len(validated_args):
                        validated_args[content_idx] = result.sanitized_value
    
    elif validation_type == "tale":
        # Validate tale-specific parameters
        tale_params = {}
        for param in ['name', 'content', 'category', 'tags']:
            if param in bound_args.arguments:
                tale_params[param] = bound_args.arguments[param]
        
        if tale_params:
            result = validator.validate_tale_input(**tale_params)
            if not result.is_valid and strict:
                raise ValidationError("Tale input validation failed", validation_type="tale")
            
            # Update with sanitized values
            for param, value in result.sanitized_value.items():
                if param in kwargs:
                    validated_kwargs[param] = value
                else:
                    param_names = list(sig.parameters.keys())
                    if param in param_names:
                        param_idx = param_names.index(param)
                        if param_idx < len(validated_args):
                            validated_args[param_idx] = value
    
    elif validation_type == "query":
        # Validate query parameters
        if 'query' in bound_args.arguments:
            query = bound_args.arguments['query']
            limit = bound_args.arguments.get('limit')
            filters = bound_args.arguments.get('filters')
            
            result = validator.validate_query_input(query, limit, filters)
            if not result.is_valid and strict:
                raise ValidationError("Query input validation failed", 
                                    field="query", validation_type="query")
            
            # Update with sanitized values
            for param, value in result.sanitized_value.items():
                if param in kwargs:
                    validated_kwargs[param] = value
    
    return tuple(validated_args), validated_kwargs


def _sanitize_function_output(result: Any, sanitization_type: str, 
                            sanitizer, preserve_structure: bool) -> Any:
    """Sanitize function output based on type."""
    if result is None:
        return result
    
    if sanitization_type == "memory":
        if isinstance(result, str):
            sanitization_result = sanitizer.sanitize_memory_content(result)
            return sanitization_result.sanitized_value
        elif isinstance(result, dict) and preserve_structure:
            # Sanitize dictionary values that are strings
            sanitized = {}
            for key, value in result.items():
                if isinstance(value, str):
                    sanitization_result = sanitizer.sanitize_memory_content(value)
                    sanitized[key] = sanitization_result.sanitized_value
                else:
                    sanitized[key] = value
            return sanitized
    
    elif sanitization_type == "json":
        sanitization_result = sanitizer.sanitize_json_input(result)
        return sanitization_result.sanitized_value
    
    return result


def _extract_user_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Extract user identifier for rate limiting."""
    import inspect
    
    # Common parameter names for user identification
    user_params = ['user_id', 'username', 'user', 'client_id', 'session_id']
    
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    # Look for user identifier in arguments
    for param in user_params:
        if param in bound_args.arguments:
            user_value = bound_args.arguments[param]
            if user_value:
                return f"user:{user_value}:{func.__name__}"
    
    # Fallback to global rate limiting
    return f"global:{func.__name__}"


def _mask_sensitive_params(func: Callable, args: tuple, kwargs: dict, 
                         sensitive_params: List[str]) -> Dict[str, Any]:
    """Mask sensitive parameters in function arguments for logging."""
    import inspect
    
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    safe_inputs = {}
    for param_name, param_value in bound_args.arguments.items():
        if param_name.lower() in [p.lower() for p in sensitive_params]:
            # Mask sensitive parameters
            safe_inputs[param_name] = "[MASKED]"
        elif isinstance(param_value, str) and len(param_value) > 100:
            # Truncate long strings
            safe_inputs[param_name] = param_value[:100] + "...[TRUNCATED]"
        else:
            safe_inputs[param_name] = param_value
    
    return safe_inputs