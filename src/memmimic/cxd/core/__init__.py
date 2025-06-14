"""
Core module for CXD Classifier.

Contains fundamental types, interfaces, and configuration classes
that form the foundation of the CXD classification system.
"""

# Import core types
from .types import (
    # Enums
    CXDFunction,
    ExecutionState,
    
    # Data structures
    CXDTag,
    CXDSequence,
    MetaClassificationResult,
    
    # Utility functions
    create_simple_sequence,
    parse_cxd_pattern,
    calculate_sequence_hash,
    merge_sequences,
)

# Import interfaces
from .interfaces import (
    # Core interfaces
    EmbeddingModel,
    CXDClassifier,
    VectorStore,
    CanonicalExampleProvider,
    
    # Support interfaces
    ConfigProvider,
    MetricsCollector,
    CacheProvider,
    StructuredLogger,
)

# Import configuration
from .config import (
    # Main configuration class
    CXDConfig,
    
    # Configuration sections
    PathsConfig,
    ModelsConfig,
    AlgorithmsConfig,
    FeaturesConfig,
    PerformanceConfig,
    LoggingConfig,
    ValidationConfig,
    CLIConfig,
    APIConfig,
    ExperimentalConfig,
    
    # Enums
    LogLevel,
    Device,
    OutputFormat,
    MetricType,
    OptimizationMetric,
    
    # Factory functions
    create_default_config,
    create_development_config,
    create_production_config,
    load_config_from_file,
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
