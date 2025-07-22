"""
Security Configuration Management

Centralized security configuration with runtime adjustment capabilities.
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration container with validation and defaults."""
    
    # Input validation settings
    max_content_length: int = 100 * 1024  # 100KB
    max_request_size: int = 1024 * 1024   # 1MB
    max_query_length: int = 1000
    max_tale_name_length: int = 200
    max_category_length: int = 100
    max_json_nesting_depth: int = 10
    max_json_string_length: int = 10000
    
    # Content validation settings
    enable_content_sanitization: bool = True
    enable_path_validation: bool = True
    enable_sql_injection_detection: bool = True
    enable_xss_detection: bool = True
    enable_command_injection_detection: bool = True
    normalize_unicode: bool = True
    strip_control_characters: bool = True
    preserve_formatting: bool = True
    
    # Rate limiting settings
    default_rate_limit_calls: int = 100
    default_rate_limit_window: int = 60  # seconds
    enable_per_user_rate_limiting: bool = True
    rate_limit_storage_size: int = 10000
    
    # Audit and logging settings
    log_security_violations: bool = True
    log_input_validation: bool = True
    log_output_sanitization: bool = False
    log_rate_limiting: bool = True
    audit_log_retention_days: int = 30
    
    # File system security
    allowed_tale_categories: list = field(default_factory=lambda: [
        'claude/core', 'claude/contexts', 'claude/insights', 
        'claude/current', 'claude/archive', 'projects/', 'misc/'
    ])
    filesystem_safe_mode: bool = True
    file_path_jail_root: Optional[str] = None
    
    # Database security
    enable_parameterized_queries: bool = True
    sql_query_timeout: float = 30.0  # seconds
    max_sql_query_complexity: int = 100
    
    # MCP security
    validate_mcp_requests: bool = True
    mcp_request_timeout: float = 30.0
    max_mcp_payload_size: int = 1024 * 1024  # 1MB
    
    # Development and debugging
    strict_mode: bool = True
    debug_security_validation: bool = False
    allow_unsafe_operations: bool = False
    
    # Advanced security features
    enable_content_scanning: bool = True
    enable_threat_detection: bool = True
    quarantine_suspicious_content: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        
        # Validate size limits
        if self.max_content_length <= 0:
            errors.append("max_content_length must be positive")
        
        if self.max_request_size <= 0:
            errors.append("max_request_size must be positive")
        
        if self.max_content_length > self.max_request_size:
            errors.append("max_content_length cannot exceed max_request_size")
        
        # Validate rate limiting
        if self.default_rate_limit_calls <= 0:
            errors.append("default_rate_limit_calls must be positive")
        
        if self.default_rate_limit_window <= 0:
            errors.append("default_rate_limit_window must be positive")
        
        # Validate file path jail if set
        if self.file_path_jail_root:
            jail_path = Path(self.file_path_jail_root)
            if not jail_path.exists():
                errors.append(f"file_path_jail_root does not exist: {self.file_path_jail_root}")
        
        if errors:
            error_msg = "Security configuration validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            else:
                logger.warning("Continuing with invalid configuration (strict_mode=False)")
    
    def get_validation_limits(self) -> Dict[str, int]:
        """Get input validation size limits."""
        return {
            'max_content_length': self.max_content_length,
            'max_request_size': self.max_request_size,
            'max_query_length': self.max_query_length,
            'max_tale_name_length': self.max_tale_name_length,
            'max_category_length': self.max_category_length,
            'max_json_nesting_depth': self.max_json_nesting_depth,
            'max_json_string_length': self.max_json_string_length
        }
    
    def get_detection_settings(self) -> Dict[str, bool]:
        """Get threat detection settings."""
        return {
            'enable_sql_injection_detection': self.enable_sql_injection_detection,
            'enable_xss_detection': self.enable_xss_detection,
            'enable_command_injection_detection': self.enable_command_injection_detection,
            'enable_path_validation': self.enable_path_validation,
            'enable_content_scanning': self.enable_content_scanning,
            'enable_threat_detection': self.enable_threat_detection
        }
    
    def get_sanitization_settings(self) -> Dict[str, bool]:
        """Get content sanitization settings."""
        return {
            'enable_content_sanitization': self.enable_content_sanitization,
            'normalize_unicode': self.normalize_unicode,
            'strip_control_characters': self.strip_control_characters,
            'preserve_formatting': self.preserve_formatting
        }
    
    def get_rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            'default_calls': self.default_rate_limit_calls,
            'default_window': self.default_rate_limit_window,
            'per_user_enabled': self.enable_per_user_rate_limiting,
            'storage_size': self.rate_limit_storage_size
        }
    
    def get_audit_config(self) -> Dict[str, Any]:
        """Get audit and logging configuration."""
        return {
            'log_security_violations': self.log_security_violations,
            'log_input_validation': self.log_input_validation,
            'log_output_sanitization': self.log_output_sanitization,
            'log_rate_limiting': self.log_rate_limiting,
            'retention_days': self.audit_log_retention_days
        }
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode with relaxed security."""
        return not self.strict_mode or self.allow_unsafe_operations or self.debug_security_validation
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)


class SecurityConfigManager:
    """Manager for security configuration with environment variable support."""
    
    def __init__(self):
        self._config = None
        self._config_file_path = None
    
    def load_config(self, config_file: Optional[str] = None, 
                   env_prefix: str = "MEMMIMIC_SECURITY_") -> SecurityConfig:
        """
        Load security configuration from file and environment variables.
        
        Args:
            config_file: Path to configuration file (JSON/YAML)
            env_prefix: Prefix for environment variables
            
        Returns:
            SecurityConfig instance
        """
        config_data = {}
        
        # Load from configuration file if provided
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                config_data = self._load_config_file(config_path)
                self._config_file_path = config_path
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        
        # Override with environment variables
        env_config = self._load_from_environment(env_prefix)
        config_data.update(env_config)
        
        # Create configuration instance
        try:
            self._config = SecurityConfig(**config_data)
        except TypeError as e:
            # Handle unknown configuration keys
            logger.error(f"Invalid configuration parameters: {e}")
            # Create with defaults and update
            self._config = SecurityConfig()
            valid_keys = {k for k in config_data.keys() if hasattr(self._config, k)}
            valid_config = {k: v for k, v in config_data.items() if k in valid_keys}
            self._config.update_from_dict(valid_config)
        
        logger.info("Security configuration loaded successfully")
        return self._config
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    return json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        return yaml.safe_load(f)
                except ImportError:
                    logger.error("PyYAML not available, cannot load YAML config")
                    return {}
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_path}: {e}")
            return {}
    
    def _load_from_environment(self, env_prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'MAX_CONTENT_LENGTH': ('max_content_length', int),
            'MAX_REQUEST_SIZE': ('max_request_size', int),
            'MAX_QUERY_LENGTH': ('max_query_length', int),
            'ENABLE_STRICT_MODE': ('strict_mode', self._parse_bool),
            'ENABLE_SQL_INJECTION_DETECTION': ('enable_sql_injection_detection', self._parse_bool),
            'ENABLE_XSS_DETECTION': ('enable_xss_detection', self._parse_bool),
            'LOG_SECURITY_VIOLATIONS': ('log_security_violations', self._parse_bool),
            'RATE_LIMIT_CALLS': ('default_rate_limit_calls', int),
            'RATE_LIMIT_WINDOW': ('default_rate_limit_window', int),
            'DEBUG_SECURITY': ('debug_security_validation', self._parse_bool),
            'FILESYSTEM_JAIL_ROOT': ('file_path_jail_root', str)
        }
        
        for env_suffix, (config_key, converter) in env_mappings.items():
            env_var = f"{env_prefix}{env_suffix}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                try:
                    config[config_key] = converter(env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
        
        return config
    
    def _parse_bool(self, value: str) -> bool:
        """Parse string value to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_config(self) -> SecurityConfig:
        """Get current security configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> SecurityConfig:
        """Reload configuration from file and environment."""
        return self.load_config(
            str(self._config_file_path) if self._config_file_path else None
        )
    
    def save_config(self, config_file: str) -> None:
        """Save current configuration to file."""
        if self._config is None:
            logger.error("No configuration to save")
            return
        
        config_path = Path(config_file)
        config_dict = self._config.to_dict()
        
        try:
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'w') as f:
                        yaml.safe_dump(config_dict, f, indent=2)
                except ImportError:
                    logger.error("PyYAML not available, cannot save YAML config")
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")


# Global configuration manager
_global_config_manager = SecurityConfigManager()


def get_security_config() -> SecurityConfig:
    """Get the global security configuration."""
    return _global_config_manager.get_config()


def initialize_security_config(config_file: Optional[str] = None,
                             env_prefix: str = "MEMMIMIC_SECURITY_") -> SecurityConfig:
    """Initialize security configuration."""
    return _global_config_manager.load_config(config_file, env_prefix)


def reload_security_config() -> SecurityConfig:
    """Reload security configuration."""
    return _global_config_manager.reload_config()


# Development helpers
def create_development_config() -> SecurityConfig:
    """Create a development-friendly security configuration."""
    return SecurityConfig(
        strict_mode=False,
        debug_security_validation=True,
        log_security_violations=True,
        log_input_validation=True,
        allow_unsafe_operations=False,  # Still maintain some security
        max_content_length=1024 * 1024,  # Larger limits for development
        default_rate_limit_calls=1000
    )


def create_production_config() -> SecurityConfig:
    """Create a production security configuration."""
    return SecurityConfig(
        strict_mode=True,
        debug_security_validation=False,
        log_security_violations=True,
        log_input_validation=False,  # Reduce log volume
        allow_unsafe_operations=False,
        enable_threat_detection=True,
        quarantine_suspicious_content=True,
        default_rate_limit_calls=100,
        audit_log_retention_days=90
    )