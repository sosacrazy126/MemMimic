"""
Core module for CXD Classifier.

Contains fundamental types, interfaces, and configuration classes
that form the foundation of the CXD classification system.
"""

# Import configuration
from .config import (  # Main configuration class; Configuration sections; Enums; Factory functions
    AlgorithmsConfig,
    APIConfig,
    CLIConfig,
    CXDConfig,
    Device,
    ExperimentalConfig,
    FeaturesConfig,
    LoggingConfig,
    LogLevel,
    MetricType,
    ModelsConfig,
    OptimizationMetric,
    OutputFormat,
    PathsConfig,
    PerformanceConfig,
    ValidationConfig,
    create_default_config,
    create_development_config,
    create_production_config,
    load_config_from_file,
)

# Import interfaces
from .interfaces import (  # Core interfaces; Support interfaces
    CacheProvider,
    CanonicalExampleProvider,
    ConfigProvider,
    CXDClassifier,
    EmbeddingModel,
    MetricsCollector,
    StructuredLogger,
    VectorStore,
)

# Import core types
from .types import (  # Enums; Data structures; Utility functions
    CXDFunction,
    CXDSequence,
    CXDTag,
    ExecutionState,
    MetaClassificationResult,
    calculate_sequence_hash,
    create_simple_sequence,
    merge_sequences,
    parse_cxd_pattern,
)

__all__ = [
    # Core types
    "CXDFunction",
    "ExecutionState",
    "CXDTag",
    "CXDSequence",
    "MetaClassificationResult",
    # Type utilities
    "create_simple_sequence",
    "parse_cxd_pattern",
    "calculate_sequence_hash",
    "merge_sequences",
    # Core interfaces
    "EmbeddingModel",
    "CXDClassifier",
    "VectorStore",
    "CanonicalExampleProvider",
    # Support interfaces
    "ConfigProvider",
    "MetricsCollector",
    "CacheProvider",
    "StructuredLogger",
    # Configuration
    "CXDConfig",
    "PathsConfig",
    "ModelsConfig",
    "AlgorithmsConfig",
    "FeaturesConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "ValidationConfig",
    "CLIConfig",
    "APIConfig",
    "ExperimentalConfig",
    # Configuration enums
    "LogLevel",
    "Device",
    "OutputFormat",
    "MetricType",
    "OptimizationMetric",
    # Configuration factories
    "create_default_config",
    "create_development_config",
    "create_production_config",
    "load_config_from_file",
]
