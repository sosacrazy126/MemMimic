#!/usr/bin/env python3
"""
Security Input Validation Tests

Tests for comprehensive input validation to prevent injection attacks.
"""

import pytest
import asyncio
from typing import List, Dict, Any

# Import security framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memmimic.security import (
    InputValidator, ValidationError, SecurityValidationError,
    SecuritySanitizer, SanitizationResult,
    get_input_validator, get_security_sanitizer
)


class TestInputValidator:
    """Test the core input validation engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = InputValidator()
    
    def test_memory_content_validation_success(self):
        """Test successful memory content validation."""
        content = "This is a normal memory content"
        result = self.validator.validate_memory_content(content, "interaction")
        
        assert result.is_valid
        assert result.sanitized_value == content
        assert len(result.errors) == 0
        assert result.metadata['original_length'] == len(content)
    
    def test_memory_content_size_limit(self):
        """Test memory content size limit validation."""
        # Create content larger than default limit (100KB)
        large_content = "x" * (100 * 1024 + 1)
        result = self.validator.validate_memory_content(large_content)
        
        assert not result.is_valid
        assert any("exceeds maximum length" in str(error) for error in result.errors)
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        malicious_inputs = [
            "'; DROP TABLE memories; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM memories",
            "'; DELETE FROM memories WHERE id=1; --",
            "admin'/**/OR/**/1=1#",
        ]
        
        for malicious_input in malicious_inputs:
            result = self.validator.validate_memory_content(malicious_input)
            
            # Should detect SQL injection patterns
            assert any(error.threat_type == 'sql_injection' 
                      for error in result.errors 
                      if isinstance(error, SecurityValidationError))
    
    def test_xss_detection(self):
        """Test XSS pattern detection."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "javascript:alert('XSS')",
            "<img onerror='alert(1)' src='x'>",
            "<div onload='alert(1)'>content</div>",
        ]
        
        for malicious_input in malicious_inputs:
            result = self.validator.validate_memory_content(malicious_input)
            
            # Should detect XSS patterns and sanitize
            assert result.sanitized_value != malicious_input  # Content should be modified
            assert any(issue['type'] == 'xss' 
                      for issue in result.metadata.get('security_issues_found', []))
    
    def test_path_traversal_detection(self):
        """Test path traversal pattern detection."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
        ]
        
        for malicious_path in malicious_paths:
            result = self.validator.validate_tale_input("test", "content", malicious_path)
            
            # Should detect path traversal patterns
            assert any(error.threat_type == 'path_traversal' 
                      for error in result.errors 
                      if isinstance(error, SecurityValidationError))
    
    def test_command_injection_detection(self):
        """Test command injection pattern detection."""
        malicious_inputs = [
            "; cat /etc/passwd",
            "| whoami",
            "&& rm -rf /",
            "$(cat /etc/passwd)",
            "`ls -la`",
        ]
        
        for malicious_input in malicious_inputs:
            result = self.validator.validate_memory_content(malicious_input)
            
            # Should detect command injection patterns
            assert any(issue['type'] == 'command_injection' 
                      for issue in result.metadata.get('security_issues_found', []))
    
    def test_tale_name_validation(self):
        """Test tale name validation and sanitization."""
        test_cases = [
            ("normal_name", True),
            ("name with spaces", True),  # Should be sanitized
            ("name/with/slashes", False),  # Path traversal attempt
            ("../../../malicious", False),  # Path traversal
            ("<script>alert('xss')</script>", False),  # XSS attempt
            ("", False),  # Empty name
            ("x" * 300, False),  # Too long
        ]
        
        for test_name, should_be_valid in test_cases:
            result = self.validator.validate_tale_input(test_name, "content")
            
            if should_be_valid:
                assert result.is_valid or len(result.warnings) > 0  # Might have warnings
            else:
                assert not result.is_valid
    
    def test_query_validation(self):
        """Test search query validation."""
        # Valid queries
        valid_queries = ["normal search", "python programming", "memory recall"]
        
        for query in valid_queries:
            result = self.validator.validate_query_input(query)
            assert result.is_valid
        
        # Invalid queries
        invalid_queries = [
            "",  # Empty
            "x" * 1001,  # Too long
            "'; DROP TABLE memories; --",  # SQL injection
        ]
        
        for query in invalid_queries:
            result = self.validator.validate_query_input(query)
            assert not result.is_valid
    
    def test_json_validation(self):
        """Test JSON input validation."""
        # Valid JSON
        valid_json = '{"test": "value", "number": 123}'
        result = self.validator.validate_json_input(valid_json)
        assert result.is_valid
        assert isinstance(result.sanitized_value, dict)
        
        # Invalid JSON
        invalid_json = '{"test": "value", "invalid"}'
        result = self.validator.validate_json_input(invalid_json)
        assert not result.is_valid
        
        # Excessively nested JSON
        deeply_nested = '{"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": "deep"}}}}}}}}}}}'
        result = self.validator.validate_json_input(deeply_nested)
        # Should detect excessive nesting
        assert any("excessive nesting" in str(error).lower() 
                  for error in result.errors)


class TestSecuritySanitizer:
    """Test the security sanitization engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sanitizer = SecuritySanitizer()
    
    def test_memory_content_sanitization(self):
        """Test memory content sanitization."""
        test_content = "Normal content with <script>alert('xss')</script>"
        result = self.sanitizer.sanitize_memory_content(test_content)
        
        assert result.sanitized_value != test_content
        assert "script" not in result.sanitized_value.lower()
        assert len(result.security_issues_found) > 0
        assert "xss" in [issue['type'] for issue in result.security_issues_found]
    
    def test_filename_sanitization(self):
        """Test filename sanitization for filesystem safety."""
        dangerous_filenames = [
            "file<name>.txt",
            "file|name.txt", 
            "file:name.txt",
            "file\"name\".txt",
            "file*name.txt",
            "file?name.txt",
            "../../../etc/passwd",
            "con.txt",  # Windows reserved name
            "aux.txt",  # Windows reserved name
        ]
        
        for filename in dangerous_filenames:
            result = self.sanitizer.sanitize_filename(filename)
            
            # Should be sanitized to safe filename
            assert result.sanitized_value != filename
            assert not any(char in result.sanitized_value for char in '<>:"|?*\\')
    
    def test_sql_value_sanitization(self):
        """Test SQL value sanitization."""
        malicious_sql = "'; DROP TABLE memories; --"
        result = self.sanitizer.sanitize_sql_value(malicious_sql)
        
        # Should escape single quotes
        assert "''" in result.sanitized_value  # Single quotes should be doubled
        assert len(result.security_issues_found) > 0
    
    def test_json_sanitization(self):
        """Test JSON input sanitization."""
        malicious_json = {
            "normal_key": "normal_value",
            "xss_key": "<script>alert('xss')</script>",
            "sql_key": "'; DROP TABLE memories; --",
        }
        
        result = self.sanitizer.sanitize_json_input(malicious_json)
        
        # Should sanitize dangerous content
        assert result.sanitized_value != malicious_json
        sanitized_values = str(result.sanitized_value).lower()
        assert "script" not in sanitized_values
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # Unicode content with different representations
        unicode_content = "café"  # é might be composed differently
        result = self.sanitizer.sanitize_memory_content(unicode_content)
        
        # Should normalize Unicode
        assert 'unicode_normalize' in result.operations_applied
    
    def test_control_character_removal(self):
        """Test control character removal."""
        content_with_controls = "Normal text\x00\x01\x02with control chars"
        result = self.sanitizer.sanitize_memory_content(content_with_controls)
        
        # Should remove control characters
        assert '\x00' not in result.sanitized_value
        assert '\x01' not in result.sanitized_value
        assert '\x02' not in result.sanitized_value


class TestSecurityIntegration:
    """Integration tests for the complete security framework."""
    
    def test_end_to_end_validation(self):
        """Test end-to-end validation flow."""
        # Simulate malicious input through the validation pipeline
        malicious_input = "'; DROP TABLE memories; <script>alert('xss')</script>"
        
        validator = get_input_validator()
        sanitizer = get_security_sanitizer()
        
        # Validate input
        validation_result = validator.validate_memory_content(malicious_input)
        
        # Should detect multiple security issues
        assert not validation_result.is_valid  # SQL injection should fail validation
        assert validation_result.sanitized_value != malicious_input  # Should be sanitized
        
        # Further sanitize if needed
        sanitization_result = sanitizer.sanitize_memory_content(validation_result.sanitized_value)
        
        # Final content should be safe
        final_content = sanitization_result.sanitized_value
        assert "drop table" not in final_content.lower()
        assert "script" not in final_content.lower()
    
    def test_performance_impact(self):
        """Test that security validation doesn't severely impact performance."""
        import time
        
        validator = get_input_validator()
        test_content = "This is a normal test content for performance measurement"
        
        # Measure validation time
        start_time = time.perf_counter()
        
        # Run multiple validations
        for _ in range(100):
            result = validator.validate_memory_content(test_content)
            assert result.is_valid
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100
        
        # Should complete validation in reasonable time (less than 10ms per validation)
        assert avg_time < 0.01, f"Validation too slow: {avg_time:.4f}s per validation"
    
    def test_configuration_override(self):
        """Test security configuration override."""
        # Create validator with custom configuration
        custom_config = {
            'max_content_length': 50,  # Very small limit for testing
            'enable_sql_injection_detection': False,
            'strict_mode': False
        }
        
        validator = InputValidator(custom_config)
        
        # Test small limit
        large_content = "x" * 51
        result = validator.validate_memory_content(large_content)
        assert not result.is_valid
        
        # Test disabled SQL injection detection
        sql_injection = "'; DROP TABLE memories; --"
        result = validator.validate_memory_content(sql_injection)
        # Should not detect SQL injection due to disabled detection
        assert not any(error.threat_type == 'sql_injection' 
                      for error in result.errors 
                      if isinstance(error, SecurityValidationError))


# Performance and stress tests
class TestSecurityPerformance:
    """Performance tests for security validation."""
    
    def test_large_input_handling(self):
        """Test handling of large inputs."""
        validator = get_input_validator()
        
        # Test with various input sizes
        sizes = [1000, 10000, 50000]  # Up to default limit
        
        for size in sizes:
            large_content = "x" * size
            result = validator.validate_memory_content(large_content)
            
            if size <= 100 * 1024:  # Within default limit
                assert result.is_valid
            else:
                assert not result.is_valid
    
    def test_regex_performance(self):
        """Test regex pattern matching performance."""
        validator = get_input_validator()
        
        # Test content that might cause regex backtracking
        problematic_content = "a" * 1000 + "!"  # Shouldn't match SQL patterns
        
        import time
        start_time = time.perf_counter()
        
        result = validator.validate_memory_content(problematic_content)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Should complete quickly even with potential regex backtracking
        assert processing_time < 0.1, f"Regex processing too slow: {processing_time:.4f}s"
        assert result.is_valid


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])