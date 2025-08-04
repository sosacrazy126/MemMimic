"""
DSPy Configuration Management

Configuration system for DSPy consciousness optimization with safety constraints
and performance monitoring.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml

@dataclass
class DSPyPerformanceConfig:
    """Performance constraints and monitoring configuration"""
    
    # Response time limits (milliseconds)
    biological_reflex_max_time: int = 5  # Hard limit for biological reflexes
    consciousness_pattern_max_time: int = 50  # Limit for consciousness operations
    optimization_max_time: int = 200  # Limit for DSPy optimization
    
    # Resource limits
    max_token_budget_per_hour: int = 10000  # Token consumption limit
    max_concurrent_optimizations: int = 3  # Parallel optimization limit
    cache_size_mb: int = 100  # Memory cache size limit
    
    # Circuit breaker settings
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: int = 30  # Seconds before attempting recovery
    
    # Monitoring intervals
    performance_check_interval: int = 60  # Seconds between performance checks
    metrics_collection_interval: int = 10  # Seconds between metric collections

@dataclass  
class DSPyOptimizationConfig:
    """DSPy optimization algorithm configuration"""
    
    # MIPROv2 settings
    min_training_examples: int = 30  # Minimum examples for optimization
    preferred_training_examples: int = 300  # Preferred examples for best results
    max_bootstrapped_demos: int = 10  # Bootstrap examples for MIPROv2
    num_candidate_programs: int = 6  # Candidate programs to evaluate
    
    # Optimization scheduling
    optimization_interval_hours: int = 24  # Hours between optimization cycles
    max_optimization_attempts: int = 3  # Max attempts per pattern
    
    # Learning parameters
    confidence_threshold: float = 0.8  # Minimum confidence for pattern application
    improvement_threshold: float = 0.1  # Minimum improvement to accept optimization
    
    # Assertion settings
    max_backtrack_attempts: int = 2  # Max retry attempts for assertions
    assertion_timeout: int = 5  # Seconds before assertion timeout

@dataclass
class DSPyModelConfig:
    """LLM model configuration for DSPy"""
    
    # Primary model for optimization
    primary_model: str = "anthropic/claude-3-sonnet-20240229"
    
    # Fallback model for cost efficiency
    fallback_model: str = "anthropic/claude-3-haiku-20240307"
    
    # Model parameters
    temperature: float = 0.1  # Low temperature for consistency
    max_tokens: int = 2048  # Token limit per response
    
    # API configuration
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    timeout: int = 30  # Request timeout in seconds
    retry_attempts: int = 3  # Number of retry attempts
    
    # Cost management
    cost_per_1k_tokens: float = 0.003  # Approximate cost per 1k tokens
    daily_cost_limit: float = 50.0  # Daily spending limit

@dataclass
class DSPyIntegrationConfig:
    """Integration settings with MemMimic consciousness vault"""
    
    # Feature flags
    enable_dspy_optimization: bool = False  # Master switch for DSPy
    enable_biological_reflex_optimization: bool = False  # Biological reflex enhancement
    enable_pattern_learning: bool = True  # Pattern learning and adaptation
    enable_assertion_validation: bool = True  # Assertion-based validation
    
    # Integration modes
    optimization_mode: str = "hybrid"  # "fast_only", "optimization_only", "hybrid"
    fallback_strategy: str = "graceful"  # "graceful", "immediate", "disabled"
    
    # Consciousness pattern integration
    cxd_classification_enhancement: bool = True  # Enhance CXD with DSPy
    pattern_storage_location: str = "amms"  # "amms", "separate", "both"
    
    # A/B testing configuration
    enable_ab_testing: bool = True  # Enable gradual rollout
    ab_test_percentage: float = 0.1  # Percentage of requests to optimize
    ab_test_duration_days: int = 7  # Duration of A/B test

@dataclass
class DSPyConfig:
    """Complete DSPy configuration for consciousness optimization"""
    
    performance: DSPyPerformanceConfig = field(default_factory=DSPyPerformanceConfig)
    optimization: DSPyOptimizationConfig = field(default_factory=DSPyOptimizationConfig)
    model: DSPyModelConfig = field(default_factory=DSPyModelConfig)
    integration: DSPyIntegrationConfig = field(default_factory=DSPyIntegrationConfig)
    
    # Configuration metadata
    config_version: str = "1.0.0"
    environment: str = "development"  # "development", "staging", "production"
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "DSPyConfig":
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls(
                performance=DSPyPerformanceConfig(**config_data.get('performance', {})),
                optimization=DSPyOptimizationConfig(**config_data.get('optimization', {})),
                model=DSPyModelConfig(**config_data.get('model', {})),
                integration=DSPyIntegrationConfig(**config_data.get('integration', {})),
                config_version=config_data.get('config_version', '1.0.0'),
                environment=config_data.get('environment', 'development')
            )
        except Exception as e:
            raise ValueError(f"Failed to load DSPy config from {config_path}: {e}")
    
    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file"""
        config_data = {
            'performance': self.performance.__dict__,
            'optimization': self.optimization.__dict__,
            'model': self.model.__dict__,
            'integration': self.integration.__dict__,
            'config_version': self.config_version,
            'environment': self.environment
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Performance validation
        if self.performance.biological_reflex_max_time > 10:
            issues.append("Biological reflex max time should be ≤10ms for optimal performance")
        
        if self.performance.max_token_budget_per_hour < 1000:
            issues.append("Token budget may be too low for effective optimization")
        
        # Optimization validation
        if self.optimization.min_training_examples < 10:
            issues.append("Minimum training examples should be at least 10")
        
        if self.optimization.confidence_threshold > 0.95:
            issues.append("Confidence threshold may be too high, reducing optimization effectiveness")
        
        # Model validation
        api_key = os.environ.get(self.model.api_key_env_var)
        if not api_key:
            issues.append(f"API key not found in environment variable: {self.model.api_key_env_var}")
        
        # Integration validation
        if self.integration.ab_test_percentage > 0.5:
            issues.append("A/B test percentage >50% may be too aggressive for initial rollout")
        
        return issues
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables"""
        overrides = {}
        
        # Environment-based overrides
        env_mappings = {
            'DSPY_ENABLE_OPTIMIZATION': ('integration', 'enable_dspy_optimization'),
            'DSPY_BIOLOGICAL_REFLEX_MAX_TIME': ('performance', 'biological_reflex_max_time'),
            'DSPY_TOKEN_BUDGET': ('performance', 'max_token_budget_per_hour'),
            'DSPY_OPTIMIZATION_MODE': ('integration', 'optimization_mode'),
            'DSPY_AB_TEST_PERCENTAGE': ('integration', 'ab_test_percentage'),
            'DSPY_ENVIRONMENT': (None, 'environment')
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value:
                if section:
                    if section not in overrides:
                        overrides[section] = {}
                    overrides[section][key] = self._convert_env_value(env_value)
                else:
                    overrides[key] = env_value
        
        return overrides
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

def create_default_config() -> DSPyConfig:
    """Create default DSPy configuration for development"""
    return DSPyConfig(
        environment="development",
        integration=DSPyIntegrationConfig(
            enable_dspy_optimization=False,  # Start disabled for safety
            enable_biological_reflex_optimization=False,
            optimization_mode="hybrid",
            ab_test_percentage=0.05  # Very conservative initial rollout
        )
    )

def create_production_config() -> DSPyConfig:
    """Create production-ready DSPy configuration"""
    return DSPyConfig(
        environment="production",
        performance=DSPyPerformanceConfig(
            biological_reflex_max_time=3,  # Stricter timing for production
            max_token_budget_per_hour=50000,  # Higher budget for production
            failure_threshold=3  # Lower tolerance for failures
        ),
        integration=DSPyIntegrationConfig(
            enable_dspy_optimization=True,
            optimization_mode="hybrid",
            ab_test_percentage=0.1,
            ab_test_duration_days=14  # Longer testing period
        )
    )