#!/usr/bin/env python3
"""
Security Decorators Tests

Tests for security validation decorators and their integration.
"""

import pytest
import time
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Import security framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memmimic.security import (
    validate_input, sanitize_output, rate_limit, audit_security,
    validate_memory_content, validate_tale_input, validate_query_input,
    SecurityValidationError, ValidationError, RateLimitExceeded,
    SecurityDecoratorError
)


class TestValidationDecorators:
    """Test input validation decorators."""
    
    def test_validate_memory_content_decorator(self):
        """Test memory content validation decorator."""
        
        @validate_memory_content(strict=True)
        def store_memory(content: str, memory_type: str = "interaction"):
            return f"Stored: {content[:20]}..."
        
        # Valid content should work
        result = store_memory("This is normal content")
        assert "Stored:" in result
        
        # Invalid content should raise exception
        with pytest.raises(SecurityDecoratorError):
            store_memory("'; DROP TABLE memories; --")  # SQL injection
    
    def test_validate_tale_input_decorator(self):
        """Test tale input validation decorator."""
        
        @validate_tale_input(strict=True)
        def save_tale(name: str, content: str, category: str = "claude/core"):
            return f"Tale saved: {name}"
        
        # Valid input should work
        result = save_tale("normal_name", "Normal content")
        assert "Tale saved:" in result
        
        # Invalid name should raise exception
        with pytest.raises(SecurityDecoratorError):
            save_tale("../../../malicious", "content")  # Path traversal
    
    def test_validate_query_input_decorator(self):
        """Test query input validation decorator."""
        
        @validate_query_input(strict=True)
        def search_memories(query: str, limit: int = 10):
            return f"Found results for: {query}"
        
        # Valid query should work
        result = search_memories("normal search")
        assert "Found results for:" in result
        
        # Invalid query should raise exception
        with pytest.raises(SecurityDecoratorError):
            search_memories("")  # Empty query
        
        with pytest.raises(SecurityDecoratorError):
            search_memories("x" * 1001)  # Too long
    
    def test_validation_with_strict_false(self):
        """Test validation decorator with strict=False."""
        
        @validate_memory_content(strict=False)
        def store_memory_lenient(content: str):
            return f"Stored: {content[:20]}..."
        
        # Should not raise exception even with malicious content
        result = store_memory_lenient("'; DROP TABLE memories; --")
        assert "Stored:" in result
    
    def test_sanitized_input_passed_to_function(self):
        """Test that sanitized input is passed to decorated function."""
        
        @validate_memory_content(strict=False)  # Allow but sanitize
        def store_memory_check_input(content: str):
            # This should receive sanitized content
            return content
        
        malicious_input = "<script>alert('xss')</script>"
        result = store_memory_check_input(malicious_input)
        
        # Result should be sanitized (HTML escaped)
        assert result != malicious_input
        assert "&lt;" in result or "script" not in result.lower()


class TestSanitizationDecorators:
    """Test output sanitization decorators."""
    
    def test_sanitize_output_decorator(self):
        """Test output sanitization decorator."""
        
        @sanitize_output(sanitization_type="memory")
        def get_memory_content():
            return "<script>alert('xss')</script>Normal content"
        
        result = get_memory_content()
        
        # Output should be sanitized
        assert result != "<script>alert('xss')</script>Normal content"
        assert "script" not in result.lower() or "&lt;" in result
    
    def test_sanitize_json_output(self):
        """Test JSON output sanitization."""
        
        @sanitize_output(sanitization_type="json")
        def get_json_data():
            return {
                "safe_data": "normal content",
                "dangerous_data": "<script>alert('xss')</script>"
            }
        
        result = get_json_data()
        
        # Should be a sanitized dictionary
        assert isinstance(result, dict)
        assert "safe_data" in result
        # Dangerous content should be sanitized
        dangerous_value = str(result.get("dangerous_data", ""))
        assert "script" not in dangerous_value.lower() or "&lt;" in dangerous_value
    
    def test_preserve_structure_setting(self):
        """Test preserve_structure setting in sanitization."""
        
        @sanitize_output(sanitization_type="memory", preserve_structure=True)
        def get_structured_data():
            return {
                "title": "Normal Title",
                "content": "<script>alert('xss')</script>",
                "metadata": {"type": "test"}
            }
        
        result = get_structured_data()
        
        # Structure should be preserved
        assert isinstance(result, dict)
        assert "title" in result
        assert "metadata" in result
        # But content should be sanitized
        content = str(result.get("content", ""))
        assert content != "<script>alert('xss')</script>"


class TestRateLimitingDecorators:
    """Test rate limiting decorators."""
    
    def test_rate_limit_decorator(self):
        """Test basic rate limiting."""
        
        @rate_limit(max_calls=3, window_seconds=1)
        def limited_function():
            return "success"
        
        # First 3 calls should succeed
        for i in range(3):
            result = limited_function()
            assert result == "success"
        
        # 4th call should raise rate limit exception
        with pytest.raises(RateLimitExceeded):
            limited_function()
        
        # After waiting, should work again
        time.sleep(1.1)  # Wait for window to expire
        result = limited_function()
        assert result == "success"
    
    def test_per_user_rate_limiting(self):
        """Test per-user rate limiting."""
        
        @rate_limit(max_calls=2, window_seconds=1, per_user=True)
        def limited_per_user(user_id: str):
            return f"success for {user_id}"
        
        # User1 should be able to make 2 calls
        limited_per_user("user1")
        limited_per_user("user1")
        
        # User1's 3rd call should fail
        with pytest.raises(RateLimitExceeded):
            limited_per_user("user1")
        
        # But user2 should still be able to call
        result = limited_per_user("user2")
        assert result == "success for user2"
    
    def test_custom_rate_limit_key(self):
        """Test custom rate limiting key function."""
        
        def custom_key_func(action: str, resource_id: str):
            return f"{action}:{resource_id}"
        
        @rate_limit(max_calls=1, window_seconds=1, key_func=custom_key_func)
        def resource_action(action: str, resource_id: str):
            return f"{action} on {resource_id}"
        
        # Different resources should have separate limits
        resource_action("read", "resource1")
        resource_action("read", "resource2")  # Should work (different resource)
        
        # Same resource should hit limit
        with pytest.raises(RateLimitExceeded):
            resource_action("read", "resource1")  # Should fail (same resource)


class TestAuditDecorators:
    """Test audit logging decorators."""
    
    @patch('memmimic.security.audit.get_security_audit_logger')
    def test_audit_security_decorator(self, mock_get_logger):
        """Test security audit logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @audit_security(event_type="test_function", log_inputs=True)
        def test_function(param1: str, param2: int = 42):
            return "test result"
        
        result = test_function("test_value", 123)
        
        assert result == "test result"
        
        # Should have logged the security event
        mock_logger.log_security_event.assert_called()
        
        # Check the logged event
        call_args = mock_logger.log_security_event.call_args[0][0]
        assert call_args.event_type == "test_function"
        assert call_args.component == "decorator"
        assert call_args.function_name == "test_function"
        assert "inputs" in call_args.metadata
    
    @patch('memmimic.security.audit.get_security_audit_logger')
    def test_audit_with_sensitive_params(self, mock_get_logger):
        """Test audit logging with sensitive parameter masking."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @audit_security(event_type="sensitive_function", log_inputs=True, 
                        sensitive_params=["password", "api_key"])
        def login_function(username: str, password: str, api_key: str):
            return "logged in"
        
        login_function("user123", "secret_pass", "api_secret_key")
        
        # Should have logged the event with masked sensitive parameters
        mock_logger.log_security_event.assert_called()
        
        call_args = mock_logger.log_security_event.call_args[0][0]
        inputs = call_args.metadata["inputs"]
        
        assert inputs["username"] == "user123"  # Not sensitive
        assert inputs["password"] == "[MASKED]"  # Sensitive
        assert inputs["api_key"] == "[MASKED]"  # Sensitive
    
    @patch('memmimic.security.audit.get_security_audit_logger')
    def test_audit_function_exception(self, mock_get_logger):
        """Test audit logging when function raises exception."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @audit_security(event_type="failing_function")
        def failing_function():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Should have logged the error event
        mock_logger.log_security_event.assert_called()
        
        call_args = mock_logger.log_security_event.call_args[0][0]
        assert call_args.event_type == "failing_function_error"
        assert call_args.severity == "HIGH"


class TestDecoratorCombination:
    """Test combinations of multiple decorators."""
    
    def test_validation_and_rate_limiting(self):
        """Test combining validation and rate limiting."""
        
        @validate_memory_content(strict=True)
        @rate_limit(max_calls=2, window_seconds=1)
        def store_memory_limited(content: str):
            return f"Stored: {content[:10]}..."
        
        # Valid content within rate limit should work
        result = store_memory_limited("Normal content")
        assert "Stored:" in result
        
        result = store_memory_limited("More content")
        assert "Stored:" in result
        
        # Rate limit should prevent 3rd call
        with pytest.raises(RateLimitExceeded):
            store_memory_limited("Third content")
    
    @patch('memmimic.security.audit.get_security_audit_logger')
    def test_full_security_stack(self, mock_get_logger):
        """Test full security stack with all decorators."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @validate_memory_content(strict=False)  # Allow but sanitize
        @sanitize_output(sanitization_type="memory")
        @rate_limit(max_calls=5, window_seconds=1)
        @audit_security(event_type="secure_function", log_inputs=False)
        def fully_secured_function(content: str):
            return f"Processed: {content}"
        
        # Should work with normal content
        result = fully_secured_function("Normal content")
        assert "Processed:" in result
        
        # Should sanitize malicious content but continue processing
        result = fully_secured_function("<script>alert('xss')</script>")
        assert "Processed:" in result
        # Output should be sanitized
        assert "script" not in result.lower() or "&lt;" in result
        
        # Should have logged audit events
        assert mock_logger.log_security_event.called
    
    def test_async_function_decorators(self):
        """Test decorators with async functions."""
        
        @validate_memory_content(strict=True)
        @rate_limit(max_calls=3, window_seconds=1)
        async def async_store_memory(content: str):
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Async stored: {content[:10]}..."
        
        # Should work with async functions
        async def run_test():
            result = await async_store_memory("Normal content")
            assert "Async stored:" in result
            
            # Validation should still work
            with pytest.raises(SecurityDecoratorError):
                await async_store_memory("'; DROP TABLE memories; --")
        
        asyncio.run(run_test())


class TestDecoratorConfiguration:
    """Test decorator configuration and error handling."""
    
    def test_decorator_error_handling(self):
        """Test decorator error handling."""
        
        @validate_input(validation_type="unknown_type", strict=True)
        def test_function(content: str):
            return content
        
        # Should handle unknown validation type gracefully
        # (Either work with auto-detection or fail gracefully)
        try:
            result = test_function("test content")
            # If it works, that's fine
            assert isinstance(result, str)
        except (SecurityDecoratorError, ValidationError):
            # If it fails with proper error, that's also fine
            pass
    
    def test_decorator_with_none_input(self):
        """Test decorators with None inputs."""
        
        @validate_memory_content(strict=False)
        def handle_none_content(content):
            return f"Handled: {content}"
        
        # Should handle None gracefully
        result = handle_none_content(None)
        assert "Handled:" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])