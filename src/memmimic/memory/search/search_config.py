"""
Configuration management for the Memory Search System.

Provides centralized configuration with validation and environment-based overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from .interfaces import SearchConfig, SimilarityMetric, ConfigurationError


@dataclass
class DefaultSearchConfig(SearchConfig):
    """Default configuration for memory search operations"""
    
    # Performance settings
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    max_results: int = 100
    default_limit: int = 10
    timeout_ms: int = 5000
    
    # Caching configuration
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000
    enable_cache_warming: bool = True
    
    # Search behavior
    min_confidence_threshold: float = 0.0
    enable_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # CXD integration
    enable_cxd_classification: bool = True
    cxd_cache_ttl: int = 7200  # 2 hours
    cxd_timeout_ms: int = 1000
    
    # Performance tuning
    batch_size: int = 32
    max_concurrent_requests: int = 100
    enable_query_preprocessing: bool = True
    
    # Environment overrides
    _env_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load environment overrides and validate configuration"""
        self._load_environment_overrides()
        self._validate_configuration()
    
    def _load_environment_overrides(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'MEMMIMIC_SEARCH_TIMEOUT_MS': ('timeout_ms', int),
            'MEMMIMIC_SEARCH_MAX_RESULTS': ('max_results', int),
            'MEMMIMIC_SEARCH_CACHE_TTL': ('cache_ttl', int),
            'MEMMIMIC_SEARCH_SIMILARITY_METRIC': ('similarity_metric', self._parse_similarity_metric),
            'MEMMIMIC_SEARCH_SEMANTIC_WEIGHT': ('semantic_weight', float),
            'MEMMIMIC_SEARCH_KEYWORD_WEIGHT': ('keyword_weight', float),
            'MEMMIMIC_SEARCH_ENABLE_CXD': ('enable_cxd_classification', self._parse_bool),
            'MEMMIMIC_SEARCH_BATCH_SIZE': ('batch_size', int),
        }
        
        for env_var, (attr_name, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(self, attr_name, converted_value)
                    self._env_overrides[attr_name] = converted_value
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(
                        f"Invalid environment variable {env_var}={env_value}: {e}",
                        error_code="INVALID_ENV_CONFIG"
                    )
    
    def _parse_similarity_metric(self, value: str) -> SimilarityMetric:
        """Parse similarity metric from string"""
        try:
            return SimilarityMetric(value.lower())
        except ValueError:
            valid_metrics = [m.value for m in SimilarityMetric]
            raise ValueError(f"Invalid similarity metric '{value}'. Valid options: {valid_metrics}")
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string"""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def _validate_configuration(self):
        """Validate configuration values"""
        errors = []
        
        # Validate numeric ranges
        if self.max_results <= 0 or self.max_results > 10000:
            errors.append("max_results must be between 1 and 10000")
        
        if self.default_limit <= 0 or self.default_limit > self.max_results:
            errors.append(f"default_limit must be between 1 and {self.max_results}")
        
        if self.timeout_ms <= 0 or self.timeout_ms > 60000:
            errors.append("timeout_ms must be between 1 and 60000")
        
        if self.cache_ttl <= 0:
            errors.append("cache_ttl must be positive")
        
        # Validate weights
        if not (0.0 <= self.semantic_weight <= 1.0):
            errors.append("semantic_weight must be between 0.0 and 1.0")
        
        if not (0.0 <= self.keyword_weight <= 1.0):
            errors.append("keyword_weight must be between 0.0 and 1.0")
        
        if abs(self.semantic_weight + self.keyword_weight - 1.0) > 0.001:
            errors.append("semantic_weight + keyword_weight must equal 1.0")
        
        # Validate confidence threshold
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            errors.append("min_confidence_threshold must be between 0.0 and 1.0")
        
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                error_code="INVALID_CONFIG",
                context={"validation_errors": errors}
            )
    
    # SearchConfig interface implementation
    def get_similarity_metric(self) -> SimilarityMetric:
        """Get configured similarity metric"""
        return self.similarity_metric
    
    def get_cache_ttl(self) -> int:
        """Get cache time-to-live in seconds"""
        return self.cache_ttl
    
    def get_max_results(self) -> int:
        """Get maximum results to return"""
        return self.max_results
    
    # Additional helper methods
    def get_weighted_scores(self) -> tuple[float, float]:
        """Get semantic and keyword weights as tuple"""
        return self.semantic_weight, self.keyword_weight
    
    def is_cxd_enabled(self) -> bool:
        """Check if CXD classification is enabled"""
        return self.enable_cxd_classification
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        return {
            'batch_size': self.batch_size,
            'max_concurrent_requests': self.max_concurrent_requests,
            'timeout_ms': self.timeout_ms,
            'cache_max_size': self.cache_max_size,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'similarity_metric': self.similarity_metric.value,
            'max_results': self.max_results,
            'default_limit': self.default_limit,
            'timeout_ms': self.timeout_ms,
            'cache_ttl': self.cache_ttl,
            'cache_max_size': self.cache_max_size,
            'enable_cache_warming': self.enable_cache_warming,
            'min_confidence_threshold': self.min_confidence_threshold,
            'enable_hybrid_search': self.enable_hybrid_search,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'enable_cxd_classification': self.enable_cxd_classification,
            'cxd_cache_ttl': self.cxd_cache_ttl,
            'cxd_timeout_ms': self.cxd_timeout_ms,
            'batch_size': self.batch_size,
            'max_concurrent_requests': self.max_concurrent_requests,
            'enable_query_preprocessing': self.enable_query_preprocessing,
            'env_overrides': self._env_overrides,
        }


def create_search_config(**overrides) -> DefaultSearchConfig:
    """Factory function to create search configuration with overrides"""
    return DefaultSearchConfig(**overrides)


def load_config_from_file(config_path: str) -> DefaultSearchConfig:
    """Load configuration from JSON/YAML file"""
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            error_code="CONFIG_FILE_NOT_FOUND"
        )
    
    try:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                # Try YAML if available
                try:
                    import yaml
                    config_data = yaml.safe_load(f)
                except ImportError:
                    raise ConfigurationError(
                        "YAML support not available. Install PyYAML to use YAML config files.",
                        error_code="YAML_NOT_AVAILABLE"
                    )
        
        # Convert similarity metric string to enum if present
        if 'similarity_metric' in config_data:
            config_data['similarity_metric'] = SimilarityMetric(config_data['similarity_metric'])
        
        return DefaultSearchConfig(**config_data)
        
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ConfigurationError(
            f"Failed to parse configuration file: {e}",
            error_code="CONFIG_PARSE_ERROR",
            context={"file_path": str(config_path)}
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {e}",
            error_code="CONFIG_LOAD_ERROR",
            context={"file_path": str(config_path)}
        )