"""
Utilities module for CXD Classifier.

Contains utility functions and classes for logging, metrics,
validation, and other common functionality.
"""

from .helpers import (
    compute_checksum,
    ensure_directory,
    format_time_delta,
    safe_json_serialize,
)
from .logging import StructuredLogger, get_logger, setup_logging
from .metrics import ClassificationMetrics, MetricsCollector, PerformanceMetrics
from .validation import ConfigValidator, TextValidator, validate_input

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

