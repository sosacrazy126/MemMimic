"""
MemMimic Configuration Management

Central configuration system for MemMimic performance and behavior settings,
including secure credential management.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Security module removed - basic config only

class PerformanceConfig:
    """Performance configuration manager for MemMimic"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        # Look for config in project root
        current_dir = Path(__file__).parent.parent.parent.parent
        config_file = current_dir / "config" / "performance_config.yaml"
        
        if config_file.exists():
            return str(config_file)
        
        # Fallback to embedded defaults
        return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'database': {
                'connection_pool_size': 5,
                'connection_timeout': 5.0,
                'wal_mode': True,
                'cache_size': 10000,
                'temp_store': 'MEMORY'
            },
            'memory': {
                'active_pool_size': 1000,
                'max_pool_size': 1500,
                'importance_threshold': 0.3,
                'auto_approve_threshold': 0.8,
                'auto_reject_threshold': 0.3,
                'duplicate_threshold': 0.85,
                'min_content_length': 10,
                'default_search_limit': 10,
                'max_search_limit': 100
            },
            'cxd': {
                'enable_cache_persistence': True,
                'rebuild_cache_on_start': False,
                'embedding_cache_size': 1000,
                'vector_index_type': 'flat',
                'confidence_threshold': 0.5,
                'similarity_threshold': 0.7
            },
            'tales': {
                'cache_size': 100,
                'cache_ttl_hours': 24,
                'backup_on_delete': True,
                'max_file_size_mb': 10
            },
            'monitoring': {
                'enable_performance_metrics': True,
                'metrics_retention_hours': 168,
                'target_response_time_ms': 5.0,
                'warning_threshold_ms': 10.0,
                'error_threshold_ms': 50.0
            },
            'async_bridge': {
                'shared_loop_enabled': True,
                'thread_pool_size': 4,
                'operation_timeout': 30.0
            },
            'logging': {
                'performance_logging': True,
                'slow_query_threshold_ms': 100.0,
                'log_level': 'INFO'
            }
        }
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path (e.g., 'database.connection_pool_size')"""
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def reload(self):
        """Reload configuration from file"""
        self._config = self._load_config()
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get_section('database')
    
    @property
    def memory_config(self) -> Dict[str, Any]:
        """Get memory configuration"""
        return self.get_section('memory')
    
    @property
    def cxd_config(self) -> Dict[str, Any]:
        """Get CXD configuration"""
        return self.get_section('cxd')
    
    @property
    def tales_config(self) -> Dict[str, Any]:
        """Get tales configuration"""
        return self.get_section('tales')
    
    @property
    def monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get_section('monitoring')


# Global configuration instance
_global_config = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = PerformanceConfig()
    return _global_config

def reload_config():
    """Reload global configuration"""
    global _global_config
    if _global_config is not None:
        _global_config.reload()


# Expose all configuration components
__all__ = [
    'PerformanceConfig',
    'get_performance_config',
    'reload_config'
]