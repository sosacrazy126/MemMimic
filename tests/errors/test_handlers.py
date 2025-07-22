"""
Tests for the MemMimic error handling decorators and utilities.

Tests comprehensive error handling patterns including retry logic,
circuit breakers, fallbacks, and structured error logging.
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Any

# Set up path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memmimic.errors.handlers import (
    # Backoff strategies
    ExponentialBackoff, LinearBackoff, ConstantBackoff,
    
    # Retry policy and circuit breaker
    RetryPolicy, CircuitBreaker, CircuitState, CircuitBreakerError,
    
    # Error handler
    ErrorHandler,
    
    # Decorators
    handle_errors, retry, circuit_breaker, fallback, log_errors, combine_decorators,
    
    # Utilities
    register_error_collector, register_global_fallback,
    get_circuit_breaker, reset_circuit_breaker, get_all_circuit_breakers
)

from memmimic.errors.exceptions import (
    MemMimicError, NetworkError, TimeoutError, DatabaseError, 
    ValidationError, ErrorSeverity
)


class TestBackoffStrategies:
    """Test backoff strategy implementations."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        backoff = ExponentialBackoff(
            initial_delay=0.1,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
        # Test exponential growth
        assert backoff.get_delay(0) == 0.1
        assert backoff.get_delay(1) == 0.2
        assert backoff.get_delay(2) == 0.4
        assert backoff.get_delay(3) == 0.8
        
        # Test max delay cap
        assert backoff.get_delay(10) == 10.0
    
    def test_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            jitter=True
        )
        
        # With jitter, delays should vary slightly
        delays = [backoff.get_delay(1) for _ in range(10)]
        
        # All delays should be around 2.0 but with some variation
        for delay in delays:
            assert 1.8 <= delay <= 2.2  # Within ±10% jitter range
        
        # Should have some variation
        assert len(set(f"{d:.3f}" for d in delays)) > 1
    
    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        backoff = LinearBackoff(delay=0.5, max_delay=5.0)
        
        assert backoff.get_delay(0) == 0.0  # 0.5 * 0
        assert backoff.get_delay(1) == 0.5  # 0.5 * 1
        assert backoff.get_delay(2) == 1.0  # 0.5 * 2
        assert backoff.get_delay(3) == 1.5  # 0.5 * 3
        
        # Test max delay cap
        assert backoff.get_delay(20) == 5.0
    
    def test_constant_backoff(self):
        """Test constant backoff strategy."""
        backoff = ConstantBackoff(delay=2.0)
        
        assert backoff.get_delay(0) == 2.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(5) == 2.0
        assert backoff.get_delay(100) == 2.0


class TestRetryPolicy:
    """Test retry policy configuration."""
    
    def test_default_policy(self):
        """Test default retry policy."""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 3
        assert isinstance(policy.backoff, ExponentialBackoff)
        assert NetworkError in policy.retriable_exceptions
        assert TimeoutError in policy.retriable_exceptions
        assert DatabaseError in policy.retriable_exceptions
    
    def test_custom_policy(self):
        """Test custom retry policy configuration."""
        custom_backoff = LinearBackoff(delay=1.0)
        policy = RetryPolicy(
            max_attempts=5,
            backoff=custom_backoff,
            retriable_exceptions=(NetworkError, TimeoutError),
            stop_on_exceptions=(ValidationError,)
        )
        
        assert policy.max_attempts == 5
        assert policy.backoff is custom_backoff
        assert policy.retriable_exceptions == (NetworkError, TimeoutError)
        assert policy.stop_on_exceptions == (ValidationError,)
    
    def test_global_default_policy(self):
        """Test global default policy management."""
        original_default = RetryPolicy._default
        
        try:
            # Set custom default
            RetryPolicy.set_default(max_attempts=7, backoff=ConstantBackoff(1.5))
            
            default_policy = RetryPolicy.get_default()
            assert default_policy.max_attempts == 7
            assert isinstance(default_policy.backoff, ConstantBackoff)
            assert default_policy.backoff.delay == 1.5
            
            # Test default attempts
            assert RetryPolicy.get_default_attempts() == 7
            
        finally:
            # Restore original default
            RetryPolicy._default = original_default


class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    def setup_method(self):
        """Clear circuit breaker registry before each test."""
        CircuitBreaker._instances.clear()
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and registration."""
        breaker = CircuitBreaker(
            name="test_breaker",
            failure_threshold=5,
            recovery_timeout=30.0
        )
        
        assert breaker.name == "test_breaker"
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 30.0
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        
        # Check registration
        assert CircuitBreaker._instances["test_breaker"] is breaker
        assert CircuitBreaker.get_active_count() == 1
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        # Initially closed and allowing requests
        assert breaker.should_allow_request() is True
        assert breaker.state == CircuitState.CLOSED
        
        # Record failures
        for i in range(2):
            breaker.record_failure(Exception("test error"))
            assert breaker.state == CircuitState.CLOSED
            assert breaker.should_allow_request() is True
        
        # Third failure should open circuit
        breaker.record_failure(Exception("test error"))
        assert breaker.state == CircuitState.OPEN
        assert breaker.should_allow_request() is False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Force open state
        breaker.record_failure(Exception("error 1"))
        breaker.record_failure(Exception("error 2"))
        assert breaker.state == CircuitState.OPEN
        
        # Should not allow requests immediately
        assert breaker.should_allow_request() is False
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should transition to half-open and allow one request
        assert breaker.should_allow_request() is True
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Success should close the circuit
        breaker.record_success()
        breaker.record_success()
        breaker.record_success()  # Need 3 successes by default
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Force to half-open state
        breaker.record_failure(Exception("error 1"))
        breaker.record_failure(Exception("error 2"))
        time.sleep(0.15)
        breaker.should_allow_request()  # Transitions to half-open
        
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Failure in half-open should go back to open
        breaker.record_failure(Exception("error in half-open"))
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_breaker_utilities(self):
        """Test circuit breaker utility functions."""
        breaker1 = CircuitBreaker("breaker1", failure_threshold=3)
        breaker2 = CircuitBreaker("breaker2", failure_threshold=5)
        
        # Test get_circuit_breaker
        assert get_circuit_breaker("breaker1") is breaker1
        assert get_circuit_breaker("nonexistent") is None
        
        # Test get_all_circuit_breakers
        all_breakers = get_all_circuit_breakers()
        assert len(all_breakers) == 2
        assert all_breakers["breaker1"] is breaker1
        assert all_breakers["breaker2"] is breaker2
        
        # Test reset_circuit_breaker
        breaker1.state = CircuitState.OPEN
        breaker1.failure_count = 5
        
        assert reset_circuit_breaker("breaker1") is True
        assert breaker1.state == CircuitState.CLOSED
        assert breaker1.failure_count == 0
        
        assert reset_circuit_breaker("nonexistent") is False


class TestErrorHandler:
    """Test error handler functionality."""
    
    def test_error_collector_registration(self):
        """Test error collector registration and execution."""
        handler = ErrorHandler()
        collector_mock = Mock()
        
        handler.register_collector(collector_mock)
        
        # Handle an error
        error = NetworkError("Test network error")
        context = {"operation": "test"}
        
        with pytest.raises(NetworkError):
            handler.handle_error(error, context)
        
        # Collector should have been called
        collector_mock.assert_called_once_with(error, context)
    
    def test_global_fallback_registration(self):
        """Test global fallback registration and execution."""
        handler = ErrorHandler()
        fallback_mock = Mock(return_value="fallback_result")
        
        handler.register_fallback(NetworkError, fallback_mock)
        
        # Handle an error with registered fallback
        error = NetworkError("Test network error")
        result = handler.handle_error(error)
        
        assert result == "fallback_result"
        fallback_mock.assert_called_once()
    
    def test_specific_fallback_priority(self):
        """Test that specific fallback takes priority over global fallback."""
        handler = ErrorHandler()
        global_fallback = Mock(return_value="global")
        specific_fallback = Mock(return_value="specific")
        
        handler.register_fallback(NetworkError, global_fallback)
        
        error = NetworkError("Test error")
        result = handler.handle_error(error, fallback=specific_fallback)
        
        assert result == "specific"
        specific_fallback.assert_called_once()
        global_fallback.assert_not_called()


class TestHandleErrorsDecorator:
    """Test handle_errors decorator."""
    
    def test_basic_error_handling(self):
        """Test basic error handling with logging."""
        with patch('memmimic.errors.handlers.logger') as mock_logger:
            
            @handle_errors(catch=[ValueError], reraise=False)
            def failing_function():
                raise ValueError("Test error")
            
            result = failing_function()
            assert result is None
            mock_logger.log.assert_called()
    
    def test_selective_catching(self):
        """Test selective exception catching."""
        
        @handle_errors(catch=[ValueError], reraise=True)
        def selective_function(error_type):
            if error_type == "value":
                raise ValueError("Value error")
            elif error_type == "type":
                raise TypeError("Type error")
        
        # ValueError should be caught and re-raised as MemMimicError
        with pytest.raises(MemMimicError):
            selective_function("value")
        
        # TypeError should pass through unchanged
        with pytest.raises(TypeError):
            selective_function("type")
    
    def test_ignore_exceptions(self):
        """Test ignoring specific exceptions."""
        
        @handle_errors(ignore=[ValueError])
        def function_with_ignore():
            raise ValueError("Should be ignored")
        
        # Ignored exception should pass through
        with pytest.raises(ValueError):
            function_with_ignore()
    
    def test_fallback_execution(self):
        """Test fallback function execution."""
        fallback_mock = Mock(return_value="fallback_result")
        
        @handle_errors(fallback=fallback_mock, reraise=False)
        def failing_function():
            raise RuntimeError("Test error")
        
        result = failing_function()
        assert result == "fallback_result"
        fallback_mock.assert_called_once()


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def test_successful_retry(self):
        """Test successful operation after retries."""
        attempt_count = 0
        
        @retry(max_attempts=3)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3
    
    def test_max_attempts_exceeded(self):
        """Test behavior when max attempts are exceeded."""
        
        @retry(max_attempts=2)
        def always_failing():
            raise NetworkError("Always fails")
        
        with pytest.raises(NetworkError):
            always_failing()
    
    def test_non_retriable_exception(self):
        """Test that non-retriable exceptions are not retried."""
        attempt_count = 0
        
        @retry(max_attempts=3, retriable_exceptions=(NetworkError,))
        def function_with_validation_error():
            nonlocal attempt_count
            attempt_count += 1
            raise ValidationError("Validation failed")
        
        with pytest.raises(ValidationError):
            function_with_validation_error()
        
        # Should only attempt once
        assert attempt_count == 1
    
    def test_stop_on_exception(self):
        """Test stop_on_exceptions functionality."""
        policy = RetryPolicy(
            max_attempts=3,
            retriable_exceptions=(Exception,),
            stop_on_exceptions=(ValidationError,)
        )
        
        attempt_count = 0
        
        @retry(policy=policy)
        def function_with_stop_exception():
            nonlocal attempt_count
            attempt_count += 1
            raise ValidationError("Should stop immediately")
        
        with pytest.raises(ValidationError):
            function_with_stop_exception()
        
        # Should only attempt once due to stop_on_exceptions
        assert attempt_count == 1
    
    @patch('time.sleep')
    def test_backoff_timing(self, mock_sleep):
        """Test that backoff delays are applied correctly."""
        
        @retry(
            max_attempts=3,
            backoff=ConstantBackoff(delay=1.0)
        )
        def failing_function():
            raise NetworkError("Always fails")
        
        with pytest.raises(NetworkError):
            failing_function()
        
        # Should have slept twice (between 3 attempts)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1.0)


class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator."""
    
    def setup_method(self):
        """Clear circuit breaker registry before each test."""
        CircuitBreaker._instances.clear()
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        
        @circuit_breaker("test_circuit", failure_threshold=2)
        def normal_function():
            return "success"
        
        # Should work normally
        result = normal_function()
        assert result == "success"
        
        # Circuit should be closed
        breaker = get_circuit_breaker("test_circuit")
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker with failures."""
        
        @circuit_breaker("test_circuit", failure_threshold=2)
        def failing_function():
            raise NetworkError("Network failure")
        
        # First two failures should be allowed
        with pytest.raises(NetworkError):
            failing_function()
        
        with pytest.raises(NetworkError):
            failing_function()
        
        # Third call should trigger circuit breaker
        with pytest.raises(CircuitBreakerError):
            failing_function()
        
        # Circuit should be open
        breaker = get_circuit_breaker("test_circuit")
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        
        # First function to trigger the circuit breaker
        @circuit_breaker(
            "recovery_test_circuit", 
            failure_threshold=1, 
            recovery_timeout=0.1,
            expected_exception=NetworkError
        )
        def failing_function():
            raise NetworkError("Failure")
        
        # Second function to test recovery
        @circuit_breaker(
            "recovery_test_circuit", 
            failure_threshold=1, 
            recovery_timeout=0.1,
            expected_exception=NetworkError
        )
        def function_that_recovers():
            return "recovered"
        
        # Trigger circuit breaker
        with pytest.raises(NetworkError):
            failing_function()
        
        # Verify circuit is open
        breaker = get_circuit_breaker("recovery_test_circuit")
        assert breaker.state == CircuitState.OPEN
        
        # Should be blocked immediately
        with pytest.raises(CircuitBreakerError):
            function_that_recovers()
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should work again
        result = function_that_recovers()
        assert result == "recovered"


class TestFallbackDecorator:
    """Test fallback decorator."""
    
    def test_fallback_on_specified_exception(self):
        """Test fallback execution on specified exceptions."""
        
        @fallback(
            fallback_func=lambda: "fallback_result",
            catch=[NetworkError]
        )
        def network_function():
            raise NetworkError("Network failure")
        
        result = network_function()
        assert result == "fallback_result"
    
    def test_no_fallback_on_unspecified_exception(self):
        """Test that fallback is not used for unspecified exceptions."""
        
        @fallback(
            fallback_func=lambda: "fallback_result",
            catch=[NetworkError]
        )
        def function_with_other_error():
            raise ValueError("Different error")
        
        with pytest.raises(ValueError):
            function_with_other_error()
    
    def test_fallback_with_arguments(self):
        """Test fallback function receiving original arguments."""
        
        def fallback_with_args(*args, **kwargs):
            return f"fallback: args={args}, kwargs={kwargs}"
        
        @fallback(fallback_func=fallback_with_args)
        def function_with_args(a, b, c=None):
            raise RuntimeError("Error")
        
        result = function_with_args(1, 2, c=3)
        assert result == "fallback: args=(1, 2), kwargs={'c': 3}"
    
    def test_fallback_function_signature_mismatch(self):
        """Test fallback when function signatures don't match."""
        
        @fallback(fallback_func=lambda: "simple_fallback")
        def function_with_args(a, b):
            raise RuntimeError("Error")
        
        result = function_with_args(1, 2)
        assert result == "simple_fallback"


class TestLogErrorsDecorator:
    """Test log_errors decorator."""
    
    def test_basic_error_logging(self):
        """Test basic error logging functionality."""
        with patch('memmimic.errors.handlers.logger') as mock_logger:
            
            @log_errors()
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            mock_logger.log.assert_called()
            args, kwargs = mock_logger.log.call_args
            assert args[0] == logging.ERROR  # log level
            assert "Error in failing_function: Test error" in args[1]
            assert kwargs['exc_info'] is True
    
    def test_custom_log_level(self):
        """Test custom log level configuration."""
        with patch('memmimic.errors.handlers.logger') as mock_logger:
            
            @log_errors(log_level=logging.WARNING)
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            args, kwargs = mock_logger.log.call_args
            assert args[0] == logging.WARNING
    
    def test_include_arguments(self):
        """Test including function arguments in logs."""
        with patch('memmimic.errors.handlers.logger') as mock_logger:
            
            @log_errors(include_args=True)
            def failing_function(a, b, c=None):
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function(1, 2, c=3)
            
            args, kwargs = mock_logger.log.call_args
            extra = kwargs['extra']
            assert 'args' in extra
            assert 'kwargs' in extra
    
    def test_no_stack_trace(self):
        """Test logging without stack trace."""
        with patch('memmimic.errors.handlers.logger') as mock_logger:
            
            @log_errors(include_stack_trace=False)
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            args, kwargs = mock_logger.log.call_args
            assert kwargs.get('exc_info') is not True


class TestCombineDecorators:
    """Test decorator combination utility."""
    
    def test_decorator_combination_order(self):
        """Test that decorators are applied in correct order."""
        execution_order = []
        
        def decorator_a(func):
            def wrapper(*args, **kwargs):
                execution_order.append('a_start')
                try:
                    result = func(*args, **kwargs)
                    execution_order.append('a_end')
                    return result
                except Exception as e:
                    execution_order.append('a_exception')
                    raise
            return wrapper
        
        def decorator_b(func):
            def wrapper(*args, **kwargs):
                execution_order.append('b_start')
                try:
                    result = func(*args, **kwargs)
                    execution_order.append('b_end')
                    return result
                except Exception as e:
                    execution_order.append('b_exception')
                    raise
            return wrapper
        
        @combine_decorators(decorator_a, decorator_b)
        def test_function():
            execution_order.append('function')
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Should execute in order: a_start, b_start, function, b_end, a_end
        expected_order = ['a_start', 'b_start', 'function', 'b_end', 'a_end']
        assert execution_order == expected_order
    
    def test_complex_decorator_stack(self):
        """Test complex decorator stack with error handling."""
        attempt_count = 0
        
        @combine_decorators(
            handle_errors(reraise=False, fallback=lambda: "handled"),
            retry(max_attempts=2),
            log_errors(log_level=logging.INFO)
        )
        def complex_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise NetworkError("First attempt fails")
            return "success"
        
        with patch('memmimic.errors.handlers.logger'):
            result = complex_function()
        
        assert result == "success"
        assert attempt_count == 2


class TestUtilityFunctions:
    """Test utility functions for error handler management."""
    
    def test_register_error_collector(self):
        """Test registering global error collector."""
        collector_mock = Mock()
        
        register_error_collector(collector_mock)
        
        # Verify collector is registered by triggering an error
        @handle_errors(reraise=False)
        def test_function():
            raise ValueError("Test error")
        
        test_function()
        
        # Collector should have been called
        assert collector_mock.call_count >= 1
    
    def test_register_global_fallback(self):
        """Test registering global fallback."""
        fallback_mock = Mock(return_value="global_fallback")
        
        register_global_fallback(ValueError, fallback_mock)
        
        # Verify fallback is used
        @handle_errors(reraise=False)
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        assert result == "global_fallback"
        fallback_mock.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])