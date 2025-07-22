#!/usr/bin/env python3
"""
MemMimic Configuration Management System
Handles loading and validation of AMMS configuration from YAML files
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ActiveMemoryPoolConfig:
    """Configuration for active memory pool management"""

    target_size: int = 1000
    max_size: int = 1500
    importance_threshold: float = 0.3

    # Performance settings
    batch_size: int = 100
    max_query_time_ms: int = 100
    cache_size: int = 500


@dataclass
class CleanupPoliciesConfig:
    """Configuration for memory cleanup policies"""

    stale_threshold_days: int = 30
    archive_threshold: float = 0.2
    prune_threshold: float = 0.1

    # Advanced thresholds
    archive_threshold_days: int = 90
    prune_threshold_days: int = 180
    min_access_frequency: float = 0.01


@dataclass
class RetentionPolicyConfig:
    """Configuration for memory type retention policies"""

    min_retention: str = "30_days"  # "permanent", "90_days", etc.
    importance_boost: float = 0.0
    archive_after: str = "90_days"


@dataclass
class ScoringWeightsConfig:
    """Configuration for importance scoring weights"""

    cxd_classification: float = 0.40
    access_frequency: float = 0.25
    recency_temporal: float = 0.20
    confidence_quality: float = 0.10
    memory_type: float = 0.05

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total = (
            self.cxd_classification
            + self.access_frequency
            + self.recency_temporal
            + self.confidence_quality
            + self.memory_type
        )
        return abs(total - 1.0) < 0.01


@dataclass
class MemMimicConfig:
    """Main MemMimic configuration"""

    active_memory_pool: ActiveMemoryPoolConfig
    cleanup_policies: CleanupPoliciesConfig
    scoring_weights: ScoringWeightsConfig
    retention_policies: Dict[str, RetentionPolicyConfig]


class ConfigLoader:
    """Configuration loader with validation and defaults"""

    DEFAULT_CONFIG_PATHS = [
        "config/memmimic_config.yaml",
        "memmimic_config.yaml",
        "~/.memmimic/config.yaml",
        "/etc/memmimic/config.yaml",
    ]

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self._config: Optional[MemMimicConfig] = None

    def load_config(self) -> MemMimicConfig:
        """Load configuration from file or use defaults"""
        if self._config is not None:
            return self._config

        config_data = self._load_config_file()
        self._config = self._parse_config(config_data)

        # Validate configuration
        if not self._validate_config(self._config):
            self.logger.warning(
                "Configuration validation failed, using validated defaults"
            )
            self._config = self._get_default_config()

        return self._config

    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_paths = (
            [self.config_path] if self.config_path else self.DEFAULT_CONFIG_PATHS
        )

        for path in config_paths:
            if path is None:
                continue

            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                try:
                    with open(expanded_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                        self.logger.info(f"Loaded configuration from {expanded_path}")
                        return config_data or {}
                except Exception as e:
                    self.logger.error(
                        f"Failed to load config from {expanded_path}: {e}"
                    )
                    continue

        self.logger.info("No configuration file found, using defaults")
        return {}

    def _parse_config(self, config_data: Dict[str, Any]) -> MemMimicConfig:
        """Parse configuration data into structured config objects"""

        # Parse active memory pool config
        pool_config = config_data.get("active_memory_pool", {})
        active_memory_pool = ActiveMemoryPoolConfig(
            target_size=pool_config.get("target_size", 1000),
            max_size=pool_config.get("max_size", 1500),
            importance_threshold=pool_config.get("importance_threshold", 0.3),
            batch_size=pool_config.get("batch_size", 100),
            max_query_time_ms=pool_config.get("max_query_time_ms", 100),
            cache_size=pool_config.get("cache_size", 500),
        )

        # Parse cleanup policies config
        cleanup_config = config_data.get("cleanup_policies", {})
        cleanup_policies = CleanupPoliciesConfig(
            stale_threshold_days=cleanup_config.get("stale_threshold_days", 30),
            archive_threshold=cleanup_config.get("archive_threshold", 0.2),
            prune_threshold=cleanup_config.get("prune_threshold", 0.1),
            archive_threshold_days=cleanup_config.get("archive_threshold_days", 90),
            prune_threshold_days=cleanup_config.get("prune_threshold_days", 180),
            min_access_frequency=cleanup_config.get("min_access_frequency", 0.01),
        )

        # Parse scoring weights config
        weights_config = config_data.get("scoring_weights", {})
        scoring_weights = ScoringWeightsConfig(
            cxd_classification=weights_config.get("cxd_classification", 0.40),
            access_frequency=weights_config.get("access_frequency", 0.25),
            recency_temporal=weights_config.get("recency_temporal", 0.20),
            confidence_quality=weights_config.get("confidence_quality", 0.10),
            memory_type=weights_config.get("memory_type", 0.05),
        )

        # Parse retention policies config
        retention_config = config_data.get("retention_policies", {})
        retention_policies = {}

        # Default retention policies
        default_policies = {
            "synthetic_wisdom": {"min_retention": "permanent", "importance_boost": 0.2},
            "milestone": {"min_retention": "permanent", "importance_boost": 0.15},
            "interaction": {"min_retention": "90_days", "archive_after": "90_days"},
            "reflection": {"min_retention": "60_days", "archive_after": "60_days"},
            "project_info": {"min_retention": "180_days", "archive_after": "180_days"},
        }

        # Merge with user configuration
        for policy_name, policy_data in {
            **default_policies,
            **retention_config,
        }.items():
            retention_policies[policy_name] = RetentionPolicyConfig(
                min_retention=policy_data.get("min_retention", "30_days"),
                importance_boost=policy_data.get("importance_boost", 0.0),
                archive_after=policy_data.get("archive_after", "90_days"),
            )

        return MemMimicConfig(
            active_memory_pool=active_memory_pool,
            cleanup_policies=cleanup_policies,
            scoring_weights=scoring_weights,
            retention_policies=retention_policies,
        )

    def _validate_config(self, config: MemMimicConfig) -> bool:
        """Validate configuration values"""
        try:
            # Validate scoring weights sum to 1.0
            if not config.scoring_weights.validate():
                self.logger.error("Scoring weights do not sum to 1.0")
                return False

            # Validate pool sizes
            if (
                config.active_memory_pool.target_size
                >= config.active_memory_pool.max_size
            ):
                self.logger.error("Target pool size must be less than max pool size")
                return False

            # Validate thresholds are between 0 and 1
            if not (0 <= config.active_memory_pool.importance_threshold <= 1):
                self.logger.error("Importance threshold must be between 0 and 1")
                return False

            if not (0 <= config.cleanup_policies.archive_threshold <= 1):
                self.logger.error("Archive threshold must be between 0 and 1")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    def _get_default_config(self) -> MemMimicConfig:
        """Get default configuration with validated values"""
        return MemMimicConfig(
            active_memory_pool=ActiveMemoryPoolConfig(),
            cleanup_policies=CleanupPoliciesConfig(),
            scoring_weights=ScoringWeightsConfig(),
            retention_policies={
                "synthetic_wisdom": RetentionPolicyConfig(
                    "permanent", 0.2, "permanent"
                ),
                "milestone": RetentionPolicyConfig("permanent", 0.15, "permanent"),
                "interaction": RetentionPolicyConfig("90_days", 0.0, "90_days"),
                "reflection": RetentionPolicyConfig("60_days", 0.0, "60_days"),
                "project_info": RetentionPolicyConfig("180_days", 0.1, "180_days"),
            },
        )


# Global configuration instance
_config_loader = ConfigLoader()


def get_config(config_path: Optional[str] = None) -> MemMimicConfig:
    """Get the global MemMimic configuration"""
    global _config_loader
    if config_path and config_path != _config_loader.config_path:
        _config_loader = ConfigLoader(config_path)
    return _config_loader.load_config()


def reload_config(config_path: Optional[str] = None) -> MemMimicConfig:
    """Reload configuration from file"""
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    return _config_loader.load_config()

