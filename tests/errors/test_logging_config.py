"""
Tests for the MemMimic structured logging configuration.

Tests comprehensive logging functionality including structured formatting,
correlation ID management, error context integration, and log filtering.
"""

import pytest
import json
import logging
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# Set up path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memmimic.errors.logging_config import (
    # Data classes and enums
    LogLevel, LogContext,
    
    # Formatters and filters
    StructuredFormatter, CorrelationIDFilter, ErrorContextFilter,
    SeverityLevelFilter, ErrorLoggingHandler,
    
    # Configuration functions
    configure_error_logging, get_error_logger, setup_error_handler_logging,
    create_logger_with_context, log_error_with_context, enhance_error_with_logging
)

from memmimic.errors.exceptions import (
    MemMimicError, ErrorSeverity, NetworkError, DatabaseError
)

from memmimic.errors.context import (
    create_error_context, with_error_context
)


class TestLogContext:
    """Test LogContext data class."""
    
    def test_log_context_creation(self):
        """Test basic LogContext creation."""
        context = LogContext(
            timestamp="2024-01-15T10:30:45.123Z",
            level="ERROR",
            logger="memmimic.test",
            message="Test message",
            correlation_id="corr_123",
            error_id="err_456"
        )
        
        assert context.timestamp == "2024-01-15T10:30:45.123Z"
        assert context.level == "ERROR"
        assert context.correlation_id == "corr_123"
        assert context.error_id == "err_456"
    
    def test_log_context_to_dict(self):
        """Test LogContext to_dict method excludes None values."""
        context = LogContext(
            timestamp="2024-01-15T10:30:45.123Z",
            level="ERROR",
            logger="memmimic.test",
            message="Test message",
            correlation_id="corr_123",
            error_id=None,  # Should be excluded
            component=None   # Should be excluded
        )
        
        result = context.to_dict()
        
        assert "timestamp" in result
        assert "correlation_id" in result
        assert "error_id" not in result
        assert "component" not in result
        assert result["level"] == "ERROR"


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = StructuredFormatter(
            include_correlation_ids=True,
            include_performance=True,
            include_context=True,
            json_indent=None
        )
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name="memmimic.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Test error message",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "ERROR"
        assert log_data["logger"] == "memmimic.test"
        assert log_data["message"] == "Test error message"
        assert "timestamp" in log_data
    
    def test_memmimic_error_formatting(self):
        """Test formatting with MemMimic error."""
        error = NetworkError(
            "Network failure",
            url="https://api.example.com",
            status_code=500,
            operation="fetch_data",
            context={"retry_count": 3}
        )
        
        record = logging.LogRecord(
            name="memmimic.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Network operation failed",
            args=(),
            exc_info=(NetworkError, error, None)
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["error_id"] == error.error_id
        assert log_data["error_code"] == "NETWORK_ERROR"
        assert log_data["component"] == "external_services"  # From ExternalServiceError
        assert log_data["operation"] == "fetch_data"
        # Check that network-specific context is included
        assert "context" in log_data
        assert log_data["context"]["url"] == "https://api.example.com"
        assert log_data["context"]["status_code"] == 500
        assert log_data["context"]["retry_count"] == 3
    
    def test_context_integration(self):
        """Test integration with error context."""
        with with_error_context(
            operation="test_operation",
            component="test_component",
            correlation_id="test_corr_123",
            metadata={"test_key": "test_value"}
        ) as ctx:
            record = logging.LogRecord(
                name="memmimic.test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test with context",
                args=(),
                exc_info=None
            )
            
            formatted = self.formatter.format(record)
            log_data = json.loads(formatted)
            
            assert log_data["operation"] == "test_operation"
            assert log_data["component"] == "test_component"
            assert log_data["correlation_id"] == "test_corr_123"
    
    def test_custom_fields_exclusion(self):
        """Test excluding specific fields from output."""
        formatter = StructuredFormatter(exclude_fields=["thread_id", "process_id"])
        
        record = logging.LogRecord(
            name="memmimic.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "thread_id" not in log_data
        assert "process_id" not in log_data
        assert "message" in log_data  # Should still have other fields
    
    def test_performance_data_inclusion(self):
        """Test inclusion of performance data."""
        with with_error_context(operation="perf_test") as ctx:
            ctx.duration_ms = 150.5
            ctx.memory_usage_mb = 25.3
            
            record = logging.LogRecord(
                name="memmimic.test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Performance test",
                args=(),
                exc_info=None
            )
            
            formatted = self.formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "performance" in log_data
            assert log_data["performance"]["duration_ms"] == 150.5
            assert log_data["performance"]["memory_usage_mb"] == 25.3
    
    def test_json_serialization_fallback(self):
        """Test fallback when JSON serialization fails."""
        # Create record with non-serializable data
        record = logging.LogRecord(
            name="memmimic.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add non-serializable object
        record.non_serializable = object()
        
        # Mock json.dumps to raise an error
        with patch('json.dumps', side_effect=TypeError("Object not serializable")):
            formatted = self.formatter.format(record)
            
            assert "[JSON_ERROR]" in formatted
            assert "Test message" in formatted
            assert "serialization failed" in formatted


class TestCorrelationIDFilter:
    """Test CorrelationIDFilter functionality."""
    
    def test_auto_generation(self):
        """Test automatic correlation ID generation."""
        filter_obj = CorrelationIDFilter(auto_generate=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        # Should add correlation ID
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'correlation_id')
        assert record.correlation_id is not None
        assert len(record.correlation_id) > 0
    
    def test_no_auto_generation(self):
        """Test disabled auto-generation."""
        filter_obj = CorrelationIDFilter(auto_generate=False)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        # Should not add correlation ID if not present
        assert not hasattr(record, 'correlation_id')
    
    def test_existing_correlation_id_preservation(self):
        """Test preservation of existing correlation ID."""
        filter_obj = CorrelationIDFilter(auto_generate=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.correlation_id = "existing_id"
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.correlation_id == "existing_id"
    
    def test_thread_local_correlation_id(self):
        """Test thread-local correlation ID management."""
        filter_obj = CorrelationIDFilter()
        
        # Set correlation ID
        filter_obj.set_correlation_id("thread_local_id")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        
        assert record.correlation_id == "thread_local_id"
        
        # Test retrieval
        assert filter_obj.get_correlation_id() == "thread_local_id"
        
        # Test clearing
        filter_obj.clear_correlation_id()
        assert filter_obj.get_correlation_id() is None
    
    def test_context_correlation_id_integration(self):
        """Test integration with error context correlation ID."""
        filter_obj = CorrelationIDFilter()
        
        with with_error_context(correlation_id="context_corr_id"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test",
                args=(),
                exc_info=None
            )
            
            filter_obj.filter(record)
            
            assert record.correlation_id == "context_corr_id"


class TestErrorContextFilter:
    """Test ErrorContextFilter functionality."""
    
    def test_context_enrichment(self):
        """Test enrichment of log records with error context."""
        filter_obj = ErrorContextFilter()
        
        with with_error_context(
            operation="test_operation",
            component="test_component",
            user_id="user_123"
        ) as ctx:
            ctx.duration_ms = 100.5
            ctx.metadata = {"key": "value"}
            
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test",
                args=(),
                exc_info=None
            )
            
            result = filter_obj.filter(record)
            
            assert result is True
            assert record.component == "test_component"
            assert record.operation == "test_operation"
            assert record.user_id == "user_123"
            assert record.duration_ms == 100.5
            assert record.context == {"key": "value"}
    
    def test_no_context_override(self):
        """Test that existing record attributes are not overridden."""
        filter_obj = ErrorContextFilter()
        
        with with_error_context(component="context_component"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test",
                args=(),
                exc_info=None
            )
            record.component = "existing_component"
            
            filter_obj.filter(record)
            
            # Should not override existing value
            assert record.component == "existing_component"
    
    def test_no_context_available(self):
        """Test behavior when no error context is available."""
        filter_obj = ErrorContextFilter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        # Should not add any attributes
        assert not hasattr(record, 'component')
        assert not hasattr(record, 'operation')


class TestSeverityLevelFilter:
    """Test SeverityLevelFilter functionality."""
    
    def test_memmimic_error_severity_mapping(self):
        """Test mapping of MemMimic error severities to log levels."""
        filter_obj = SeverityLevelFilter()
        
        error = NetworkError("Test error")
        error.severity = ErrorSeverity.CRITICAL
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,  # Original level
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=(NetworkError, error, None)
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.levelno == logging.CRITICAL
        assert record.levelname == "CRITICAL"
        assert record.severity == "critical"
    
    def test_severity_mapping_table(self):
        """Test all severity mappings."""
        filter_obj = SeverityLevelFilter()
        
        severity_mappings = [
            (ErrorSeverity.CRITICAL, logging.CRITICAL),
            (ErrorSeverity.HIGH, logging.ERROR),
            (ErrorSeverity.MEDIUM, logging.WARNING),
            (ErrorSeverity.LOW, logging.INFO),
            (ErrorSeverity.INFO, logging.DEBUG),
        ]
        
        for severity, expected_level in severity_mappings:
            error = MemMimicError("Test", severity=severity)
            
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test",
                args=(),
                exc_info=(MemMimicError, error, None)
            )
            
            filter_obj.filter(record)
            
            assert record.levelno == expected_level
            assert record.severity == severity.value
    
    def test_severity_from_record_extras(self):
        """Test severity mapping from record extras."""
        filter_obj = SeverityLevelFilter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.severity = "high"
        
        filter_obj.filter(record)
        
        assert record.levelno == logging.ERROR
        assert record.levelname == "ERROR"
    
    def test_invalid_severity_handling(self):
        """Test handling of invalid severity values."""
        filter_obj = SeverityLevelFilter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.severity = "invalid_severity"
        
        original_level = record.levelno
        filter_obj.filter(record)
        
        # Should keep original level for invalid severity
        assert record.levelno == original_level


class TestErrorLoggingHandler:
    """Test ErrorLoggingHandler functionality."""
    
    def test_memmimic_error_enhancement(self):
        """Test enhancement of records with MemMimic error information."""
        with patch('logging.Handler.emit') as mock_emit:
            handler = ErrorLoggingHandler()
            
            error = DatabaseError(
                "Database connection failed",
                query="SELECT * FROM users",
                context={"retry_count": 2}
            )
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Database error occurred",
                args=(),
                exc_info=(DatabaseError, error, None)
            )
            
            handler.emit(record)
            
            # Check that record was enhanced
            assert record.error_id == error.error_id
            assert record.error_code == "DATABASE_ERROR"
            assert record.component == error.component
            assert record.context == error.context
            
            mock_emit.assert_called_once()
    
    def test_message_truncation(self):
        """Test truncation of long log messages."""
        with patch('logging.Handler.emit') as mock_emit:
            handler = ErrorLoggingHandler(max_message_length=50)
            
            long_message = "This is a very long message that exceeds the maximum length limit"
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg=long_message,
                args=(),
                exc_info=None
            )
            
            handler.emit(record)
            
            assert len(record.msg) <= 50 + len("... [TRUNCATED]")
            assert record.msg.endswith("... [TRUNCATED]")
            
            mock_emit.assert_called_once()


class TestConfigurationFunctions:
    """Test logging configuration functions."""
    
    def test_configure_error_logging_basic(self):
        """Test basic error logging configuration."""
        # Clear any existing configuration
        if hasattr(configure_error_logging, '_configured'):
            delattr(configure_error_logging, '_configured')
        
        configure_error_logging(
            level=logging.WARNING,
            format_type="structured",
            include_correlation_ids=True
        )
        
        logger = logging.getLogger('memmimic')
        
        assert logger.level == logging.WARNING
        assert len(logger.handlers) > 0
        assert hasattr(configure_error_logging, '_configured')
        assert configure_error_logging._configured is True
    
    def test_configure_with_file_output(self):
        """Test configuration with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            configure_error_logging(
                level=logging.INFO,
                format_type="structured",
                output_file=log_file
            )
            
            logger = logging.getLogger('memmimic')
            
            # Should have both console and file handlers
            assert len(logger.handlers) >= 2
            
            # Test that logging to file works
            logger.error("Test error message")
            
            # Check that file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test error message" in content
    
    def test_get_error_logger(self):
        """Test get_error_logger function."""
        logger = get_error_logger("test_module")
        
        assert logger.name == "memmimic.test_module"
        assert isinstance(logger, logging.Logger)
    
    def test_create_logger_with_context(self):
        """Test logger adapter creation with context."""
        logger_adapter = create_logger_with_context(
            "test_module",
            component="test_component",
            operation="test_operation",
            correlation_id="test_corr_123"
        )
        
        assert isinstance(logger_adapter, logging.LoggerAdapter)
        assert logger_adapter.extra["component"] == "test_component"
        assert logger_adapter.extra["operation"] == "test_operation"
        assert logger_adapter.extra["correlation_id"] == "test_corr_123"
    
    def test_log_error_with_context(self):
        """Test logging with context utility function."""
        with patch('logging.Logger.log') as mock_log:
            logger = get_error_logger("test")
            error = NetworkError("Test error")
            
            log_error_with_context(
                logger,
                logging.ERROR,
                "Operation failed",
                error=error,
                user_id="123",
                operation="test_op"
            )
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            
            assert args[0] == logging.ERROR
            assert args[1] == "Operation failed"
            assert kwargs['exc_info'] is True
            assert kwargs['extra']['user_id'] == "123"
            assert kwargs['extra']['operation'] == "test_op"
            assert kwargs['extra']['error_id'] == error.error_id
    
    def test_enhance_error_with_logging(self):
        """Test error enhancement with automatic logging."""
        with patch('memmimic.errors.logging_config.log_error_with_context') as mock_log:
            error = DatabaseError("Database error")
            
            result = enhance_error_with_logging(error)
            
            assert result is error  # Should return same instance
            mock_log.assert_called_once()
            
            # Check call arguments
            args = mock_log.call_args[0]
            kwargs = mock_log.call_args[1]
            
            assert args[1] == logging.ERROR
            assert error.error_code in args[2]
            assert kwargs['error'] is error


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_complete_error_logging_flow(self):
        """Test complete error logging flow with all components."""
        # Clear existing handlers to avoid interference
        memmimic_logger = logging.getLogger('memmimic')
        memmimic_logger.handlers.clear()
        
        # Configure logging
        configure_error_logging(
            level=logging.INFO,
            format_type="structured",
            include_correlation_ids=True,
            include_performance=True,
            include_context=True
        )
        
        logger = get_error_logger("integration_test")
        
        with with_error_context(
            operation="integration_test",
            component="test_component",
            user_id="test_user_123"
        ):
            # Create and log an error
            error = NetworkError("Network operation failed")
            error.context.update({"retry_count": 3, "endpoint": "https://api.example.com"})
            
            # This should trigger all the logging enhancements
            # Capture specific handler for testing
            handler = logger.handlers[0] if logger.handlers else memmimic_logger.handlers[0]
            
            with patch.object(handler, 'emit') as mock_emit:
                logger.error("Integration test error occurred", exc_info=(NetworkError, error, None))
                
                mock_emit.assert_called_once()
                record = mock_emit.call_args[0][0]
                
                # Verify the record was properly enhanced
                assert hasattr(record, 'correlation_id')
                assert record.component == "test_component"
                assert record.operation == "integration_test"
                assert record.user_id == "test_user_123"
    
    def test_performance_logging_integration(self):
        """Test performance data logging integration."""
        # Clear existing handlers to avoid interference
        memmimic_logger = logging.getLogger('memmimic')
        memmimic_logger.handlers.clear()
        
        configure_error_logging(include_performance=True)
        logger = get_error_logger("performance_test")
        
        with with_error_context(operation="performance_test") as ctx:
            # Simulate operation timing
            ctx.duration_ms = 250.7
            ctx.memory_usage_mb = 15.3
            
            # Capture specific handler for testing
            handler = logger.handlers[0] if logger.handlers else memmimic_logger.handlers[0]
            
            with patch.object(handler, 'emit') as mock_emit:
                logger.info("Operation completed", extra={"response_time": 250})
                
                mock_emit.assert_called_once()
                record = mock_emit.call_args[0][0]
                
                # Should have performance data from context
                assert record.duration_ms == 250.7
                # Should also have extra performance data
                assert record.response_time == 250


class TestJsonOutputValidation:
    """Test JSON output format validation."""
    
    def test_structured_json_output_format(self):
        """Test that structured output produces valid JSON with expected fields."""
        formatter = StructuredFormatter()
        
        with with_error_context(
            operation="json_test",
            component="json_component",
            correlation_id="json_corr_123"
        ):
            record = logging.LogRecord(
                name="memmimic.json_test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="JSON format test",
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)  # Should not raise
            
            # Check required fields
            required_fields = ["timestamp", "level", "logger", "message"]
            for field in required_fields:
                assert field in log_data
            
            # Check context fields
            assert log_data["operation"] == "json_test"
            assert log_data["component"] == "json_component"
            assert log_data["correlation_id"] == "json_corr_123"
            
            # Check timestamp format
            timestamp = log_data["timestamp"]
            # Should be ISO format
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])