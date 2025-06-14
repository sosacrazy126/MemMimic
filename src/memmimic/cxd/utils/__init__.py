"""
Utilities module for CXD Classifier.

Contains utility functions and classes for logging, metrics,
validation, and other common functionality.
"""

from .logging import (
    StructuredLogger,
    setup_logging,
    get_logger,
)

from .metrics import (
    PerformanceMetrics,
    ClassificationMetrics,
    MetricsCollector,
)

from .validation import (
    TextValidator,
    ConfigValidator,
    validate_input,
)

from .helpers import (
    ensure_directory,
    safe_json_serialize,
    compute_checksum,
    format_time_delta,
)

__all__ = [
    # Logging
    "StructuredLogger",
    "setup_logging",
    "get_logger",
    
    # Metrics
    "PerformanceMetrics",
    "ClassificationMetrics", 
    "MetricsCollector",
    
    # Validation
    "TextValidator",
    "ConfigValidator",
    "validate_input",
    
    # Helpers
    "ensure_directory",
    "safe_json_serialize",
    "compute_checksum", 
    "format_time_delta",
]
