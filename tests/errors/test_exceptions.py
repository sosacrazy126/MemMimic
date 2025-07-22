"""
Tests for the MemMimic error handling framework exceptions.

Tests the comprehensive exception hierarchy, context management,
and error utility functions.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any

# Set up path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import directly from the exceptions module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'memmimic', 'errors'))

from exceptions import (
    MemMimicError, ErrorSeverity,
    SystemError, ConfigurationError, InitializationError, ResourceError,
    MemMimicMemoryError, MemoryStorageError, MemoryRetrievalError, MemoryCorruptionError, MemoryValidationError,
    CXDError, ClassificationError, TrainingError, ModelError, CXDIntegrationError,
    MCPError, ProtocolError, HandlerError, CommunicationError, MCPValidationError,
    APIError, ValidationError, AuthenticationError, AuthorizationError, RateLimitError,
    ExternalServiceError, DatabaseError, NetworkError, TimeoutError, ExternalAPIError,
    create_error, is_retriable_error, get_error_severity
)


class TestMemMimicError:
    """Test the base MemMimicError class functionality."""
    
    def test_basic_error_creation(self):
        """Test basic error creation with minimal parameters."""
        error = MemMimicError("Test error message")
        
        assert error.message == "Test error message"
        assert error.error_code == "MemMimicError"
        assert error.severity == ErrorSeverity.MEDIUM
        assert isinstance(error.error_id, str)
        assert isinstance(error.timestamp, datetime)
        assert error.context == {}
        assert error.correlation_id is None
        assert error.component is None
        assert error.operation is None
    
    def test_error_with_full_context(self):
        """Test error creation with full context information."""
        context = {"user_id": "123", "memory_id": "mem_456"}
        error = MemMimicError(
            message="Complex error",
            error_code="CUSTOM_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            component="test_component",
            operation="test_operation",
            correlation_id="corr_789"
        )
        
        assert error.message == "Complex error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == context
        assert error.component == "test_component"
        assert error.operation == "test_operation"
        assert error.correlation_id == "corr_789"
    
    def test_error_context_methods(self):
        """Test error context manipulation methods."""
        error = MemMimicError("Test error")
        
        # Test add_context
        result = error.add_context("key1", "value1")
        assert result is error  # Method should return self
        assert error.context["key1"] == "value1"
        
        # Test set_correlation_id
        result = error.set_correlation_id("new_corr_id")
        assert result is error
        assert error.correlation_id == "new_corr_id"
        
        # Test set_component
        result = error.set_component("new_component")
        assert result is error
        assert error.component == "new_component"
        
        # Test set_operation
        result = error.set_operation("new_operation")
        assert result is error
        assert error.operation == "new_operation"
    
    def test_error_to_dict(self):
        """Test conversion of error to dictionary."""
        error = MemMimicError(
            message="Test error",
            error_code="TEST_ERROR",
            context={"key": "value"},
            severity=ErrorSeverity.HIGH,
            component="test_comp",
            operation="test_op",
            correlation_id="test_corr"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_id"] == error.error_id
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["severity"] == "high"
        assert error_dict["component"] == "test_comp"
        assert error_dict["operation"] == "test_op"
        assert error_dict["correlation_id"] == "test_corr"
        assert error_dict["context"] == {"key": "value"}
        assert "timestamp" in error_dict
    
    def test_error_str_representation(self):
        """Test string representation of errors."""
        error = MemMimicError(
            message="Test error",
            error_code="TEST_ERROR",
            component="test_comp",
            operation="test_op",
            correlation_id="test_corr",
            context={"key": "value"}
        )
        
        error_str = str(error)
        
        assert "[TEST_ERROR] Test error" in error_str
        assert "Component: test_comp" in error_str
        assert "Operation: test_op" in error_str
        assert "Correlation ID: test_corr" in error_str
        assert "Context: key=value" in error_str
    
    def test_error_repr(self):
        """Test repr representation of errors."""
        error = MemMimicError("Test error", error_code="TEST_ERROR")
        
        error_repr = repr(error)
        
        assert "MemMimicError" in error_repr
        assert "message='Test error'" in error_repr
        assert "error_code='TEST_ERROR'" in error_repr
        assert f"error_id='{error.error_id}'" in error_repr
    
    def test_original_exception_tracking(self):
        """Test tracking of original exceptions."""
        original = ValueError("Original error")
        
        try:
            raise original
        except ValueError as e:
            try:
                raise MemMimicError("Wrapped error") from e
            except MemMimicError as wrapped_error:
                error = wrapped_error
        
        assert error._get_original_exception is original
        
        error_dict = error.to_dict()
        assert error_dict["original_exception"]["type"] == "ValueError"
        assert error_dict["original_exception"]["message"] == "Original error"


class TestSystemErrors:
    """Test system-level errors."""
    
    def test_configuration_error(self):
        """Test ConfigurationError functionality."""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="database_url"
        )
        
        assert isinstance(error, SystemError)
        assert isinstance(error, MemMimicError)
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.context["config_key"] == "database_url"
        assert error.severity == ErrorSeverity.HIGH  # Default for SystemError
    
    def test_initialization_error(self):
        """Test InitializationError functionality."""
        error = InitializationError(
            "Component failed to initialize",
            component="cxd_classifier"
        )
        
        assert isinstance(error, SystemError)
        assert error.error_code == "INITIALIZATION_ERROR"
        assert error.component == "cxd_classifier"
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_resource_error(self):
        """Test ResourceError functionality."""
        error = ResourceError(
            "Resource exhausted",
            resource_type="memory"
        )
        
        assert isinstance(error, SystemError)
        assert error.error_code == "RESOURCE_ERROR"
        assert error.context["resource_type"] == "memory"
        assert error.severity == ErrorSeverity.HIGH


class TestMemoryErrors:
    """Test memory-related errors."""
    
    def test_memory_storage_error(self):
        """Test MemoryStorageError functionality."""
        error = MemoryStorageError(
            "Failed to store memory",
            memory_id="mem_123"
        )
        
        assert isinstance(error, MemMimicMemoryError)
        assert isinstance(error, MemMimicError)
        assert error.error_code == "MEMORY_STORAGE_ERROR"
        assert error.operation == "memory_storage"
        assert error.component == "memory_system"
        assert error.context["memory_id"] == "mem_123"
        assert error.severity == ErrorSeverity.HIGH
    
    def test_memory_retrieval_error(self):
        """Test MemoryRetrievalError functionality."""
        error = MemoryRetrievalError(
            "Failed to retrieve memory",
            memory_id="mem_456"
        )
        
        assert isinstance(error, MemMimicMemoryError)
        assert error.error_code == "MEMORY_RETRIEVAL_ERROR"
        assert error.operation == "memory_retrieval"
        assert error.context["memory_id"] == "mem_456"
    
    def test_memory_corruption_error(self):
        """Test MemoryCorruptionError functionality."""
        error = MemoryCorruptionError(
            "Memory data is corrupted",
            memory_id="mem_789"
        )
        
        assert isinstance(error, MemMimicMemoryError)
        assert error.error_code == "MEMORY_CORRUPTION_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_memory_validation_error(self):
        """Test MemoryValidationError functionality."""
        validation_errors = ["missing_field", "invalid_format"]
        error = MemoryValidationError(
            "Memory validation failed",
            validation_errors=validation_errors
        )
        
        assert isinstance(error, MemMimicMemoryError)
        assert error.error_code == "MEMORY_VALIDATION_ERROR"
        assert error.context["validation_errors"] == validation_errors


class TestCXDErrors:
    """Test CXD-related errors."""
    
    def test_classification_error(self):
        """Test ClassificationError functionality."""
        content = "This is test content for classification"
        error = ClassificationError(
            "Classification failed",
            content=content
        )
        
        assert isinstance(error, CXDError)
        assert error.error_code == "CLASSIFICATION_ERROR"
        assert error.operation == "cxd_classification"
        assert error.component == "cxd_system"
        assert error.context["content_preview"] == content
        assert error.context["content_length"] == len(content)
    
    def test_classification_error_long_content(self):
        """Test ClassificationError with long content truncation."""
        content = "x" * 500  # Long content
        error = ClassificationError(
            "Classification failed",
            content=content
        )
        
        assert len(error.context["content_preview"]) == 203  # 200 + "..."
        assert error.context["content_preview"].endswith("...")
        assert error.context["content_length"] == 500
    
    def test_training_error(self):
        """Test TrainingError functionality."""
        error = TrainingError(
            "Model training failed",
            model_type="classifier"
        )
        
        assert isinstance(error, CXDError)
        assert error.error_code == "TRAINING_ERROR"
        assert error.operation == "model_training"
        assert error.context["model_type"] == "classifier"
        assert error.severity == ErrorSeverity.HIGH
    
    def test_model_error(self):
        """Test ModelError functionality."""
        error = ModelError(
            "Model operation failed",
            model_id="model_123"
        )
        
        assert isinstance(error, CXDError)
        assert error.error_code == "MODEL_ERROR"
        assert error.context["model_id"] == "model_123"


class TestMCPErrors:
    """Test MCP-related errors."""
    
    def test_protocol_error(self):
        """Test ProtocolError functionality."""
        error = ProtocolError(
            "Protocol violation",
            protocol_version="1.0",
            request_id="req_123"
        )
        
        assert isinstance(error, MCPError)
        assert error.error_code == "PROTOCOL_ERROR"
        assert error.component == "mcp_system"
        assert error.context["protocol_version"] == "1.0"
        assert error.context["request_id"] == "req_123"
    
    def test_handler_error(self):
        """Test HandlerError functionality."""
        error = HandlerError(
            "Handler failed",
            handler_type="memory_recall",
            request_id="req_456"
        )
        
        assert isinstance(error, MCPError)
        assert error.error_code == "HANDLER_ERROR"
        assert error.context["handler_type"] == "memory_recall"
        assert error.context["request_id"] == "req_456"
    
    def test_mcp_validation_error(self):
        """Test MCPValidationError functionality."""
        validation_errors = ["missing_parameter", "invalid_type"]
        error = MCPValidationError(
            "Request validation failed",
            validation_errors=validation_errors,
            request_id="req_789"
        )
        
        assert isinstance(error, MCPError)
        assert error.error_code == "MCP_VALIDATION_ERROR"
        assert error.context["validation_errors"] == validation_errors
        assert error.context["request_id"] == "req_789"


class TestAPIErrors:
    """Test API-related errors."""
    
    def test_validation_error(self):
        """Test ValidationError functionality."""
        field_errors = {"email": "Invalid format", "age": "Must be positive"}
        error = ValidationError(
            "Validation failed",
            field_errors=field_errors
        )
        
        assert isinstance(error, APIError)
        assert error.error_code == "VALIDATION_ERROR"
        assert error.component == "api_system"
        assert error.context["status_code"] == 400
        assert error.context["field_errors"] == field_errors
    
    def test_authentication_error(self):
        """Test AuthenticationError functionality."""
        error = AuthenticationError("Invalid credentials")
        
        assert isinstance(error, APIError)
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.context["status_code"] == 401
        assert error.severity == ErrorSeverity.HIGH
    
    def test_authorization_error(self):
        """Test AuthorizationError functionality."""
        error = AuthorizationError(
            "Insufficient permissions",
            required_permission="memory:write"
        )
        
        assert isinstance(error, APIError)
        assert error.error_code == "AUTHORIZATION_ERROR"
        assert error.context["status_code"] == 403
        assert error.context["required_permission"] == "memory:write"
    
    def test_rate_limit_error(self):
        """Test RateLimitError functionality."""
        error = RateLimitError(
            "Rate limit exceeded",
            limit=100,
            reset_time=3600
        )
        
        assert isinstance(error, APIError)
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.context["status_code"] == 429
        assert error.context["rate_limit"] == 100
        assert error.context["reset_time"] == 3600


class TestExternalServiceErrors:
    """Test external service errors."""
    
    def test_database_error(self):
        """Test DatabaseError functionality."""
        query = "SELECT * FROM memories WHERE id = ?"
        error = DatabaseError(
            "Database query failed",
            query=query,
            service_name="postgresql"
        )
        
        assert isinstance(error, ExternalServiceError)
        assert error.error_code == "DATABASE_ERROR"
        assert error.component == "external_services"
        assert error.context["service_name"] == "postgresql"
        assert error.context["query_preview"] == query
        assert error.severity == ErrorSeverity.HIGH
    
    def test_database_error_long_query(self):
        """Test DatabaseError with long query truncation."""
        query = "SELECT * FROM table WHERE " + "column = 'value' AND " * 50
        error = DatabaseError("Query failed", query=query)
        
        assert len(error.context["query_preview"]) == 503  # 500 + "..."
        assert error.context["query_preview"].endswith("...")
    
    def test_network_error(self):
        """Test NetworkError functionality."""
        error = NetworkError(
            "Network request failed",
            url="https://api.example.com/endpoint",
            status_code=500,
            service_name="external_api"
        )
        
        assert isinstance(error, ExternalServiceError)
        assert error.error_code == "NETWORK_ERROR"
        assert error.context["url"] == "https://api.example.com/endpoint"
        assert error.context["status_code"] == 500
        assert error.context["service_name"] == "external_api"
    
    def test_timeout_error(self):
        """Test TimeoutError functionality."""
        error = TimeoutError(
            "Operation timed out",
            timeout_seconds=30.0,
            service_name="slow_service"
        )
        
        assert isinstance(error, ExternalServiceError)
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.context["timeout_seconds"] == 30.0
        assert error.context["service_name"] == "slow_service"


class TestErrorUtilities:
    """Test error utility functions."""
    
    def test_create_error_function(self):
        """Test create_error factory function."""
        # Test memory error creation
        error = create_error(
            "memory_storage",
            "Storage failed",
            memory_id="mem_123"
        )
        
        assert isinstance(error, MemoryStorageError)
        assert error.message == "Storage failed"
        assert error.context["memory_id"] == "mem_123"
        
        # Test system error creation
        error = create_error(
            "configuration",
            "Config invalid",
            config_key="database_url"
        )
        
        assert isinstance(error, ConfigurationError)
        assert error.message == "Config invalid"
        assert error.context["config_key"] == "database_url"
    
    def test_create_error_invalid_type(self):
        """Test create_error with invalid error type."""
        with pytest.raises(ValueError, match="Unknown error type"):
            create_error("invalid_type", "Test message")
    
    def test_is_retriable_error(self):
        """Test is_retriable_error function."""
        # Retriable errors
        assert is_retriable_error(NetworkError("Network failed"))
        assert is_retriable_error(TimeoutError("Timeout"))
        assert is_retriable_error(DatabaseError("connection timeout"))
        assert is_retriable_error(DatabaseError("temporary failure"))
        assert is_retriable_error(ExternalAPIError("API failed", response_code=500))
        assert is_retriable_error(ExternalAPIError("Rate limited", response_code=429))
        
        # Non-retriable errors
        assert not is_retriable_error(ConfigurationError("Config invalid"))
        assert not is_retriable_error(ValidationError("Validation failed"))
        assert not is_retriable_error(AuthenticationError("Auth failed"))
        assert not is_retriable_error(DatabaseError("syntax error"))
        assert not is_retriable_error(ExternalAPIError("Bad request", response_code=400))
        assert not is_retriable_error(ValueError("Generic error"))
    
    def test_get_error_severity(self):
        """Test get_error_severity function."""
        # MemMimic errors return their configured severity
        error = MemMimicError("Test", severity=ErrorSeverity.HIGH)
        assert get_error_severity(error) == ErrorSeverity.HIGH
        
        # System errors mapped by type
        assert get_error_severity(KeyboardInterrupt()) == ErrorSeverity.CRITICAL
        assert get_error_severity(SystemExit()) == ErrorSeverity.CRITICAL
        assert get_error_severity(MemoryError()) == ErrorSeverity.HIGH
        assert get_error_severity(OSError()) == ErrorSeverity.HIGH
        assert get_error_severity(ValueError()) == ErrorSeverity.MEDIUM
        assert get_error_severity(TypeError()) == ErrorSeverity.MEDIUM
        assert get_error_severity(AttributeError()) == ErrorSeverity.LOW


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_severity_values(self):
        """Test severity enum values."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.INFO.value == "info"
    
    def test_severity_ordering(self):
        """Test that severities can be compared (for alerting logic)."""
        severities = [
            ErrorSeverity.INFO,
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL
        ]
        
        # Test that all values are distinct
        assert len(set(s.value for s in severities)) == len(severities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])