"""
Simple Governance Framework for MemMimic v2.0
Lightweight, configurable governance with <10ms performance target.
"""

import json
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..errors import (
    MemoryStorageError, handle_errors, with_error_context, get_error_logger
)
from .enhanced_memory import EnhancedMemory


class GovernanceConfigError(Exception):
    """Governance configuration error"""
    pass


class GovernanceValidationError(Exception):
    """Governance validation error"""
    pass


@dataclass
class GovernanceViolation:
    """Detailed governance violation information"""
    type: str
    message: str
    severity: str  # critical, high, medium, low
    value: Optional[Union[int, float, str]] = None
    threshold: Optional[Union[int, float, str]] = None
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type,
            'message': self.message,
            'severity': self.severity,
            'value': self.value,
            'threshold': self.threshold,
            'remediation': self.remediation
        }


@dataclass
class GovernanceResult:
    """Comprehensive governance validation result"""
    approved: bool
    status: str  # approved, approved_with_warnings, rejected, error
    violations: List[GovernanceViolation]
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    operation_context: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'approved': self.approved,
            'status': self.status,
            'violations': [v.to_dict() for v in self.violations],
            'warnings': self.warnings,
            'processing_time': self.processing_time,
            'operation_context': self.operation_context,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }


@dataclass
class GovernanceConfig:
    """Configuration dataclass for governance thresholds and settings"""
    # Core content limits
    content_size: int = 1_000_000  # 1MB maximum content
    summary_length: int = 1000     # 1000 characters
    metadata_size: int = 10_000    # 10KB metadata
    
    # Tag management
    tag_count: int = 100           # Maximum tags per memory
    tag_length: int = 50           # Maximum characters per tag
    
    # Relationship governance
    relationship_depth: int = 3    # Maximum relationship depth
    
    # Performance governance
    governance_timeout: int = 10   # Maximum governance time (ms)
    
    # Enforcement settings
    enabled: bool = True
    enforcement_mode: str = "strict"  # strict, permissive, audit_only
    
    # Environment-specific overrides
    environment: str = "development"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GovernanceConfig':
        """Create config from dictionary"""
        # Extract thresholds if nested
        if 'thresholds' in data:
            thresholds = data['thresholds']
        else:
            thresholds = data
            
        return cls(
            content_size=thresholds.get('content_size', 1_000_000),
            summary_length=thresholds.get('summary_length', 1000),
            metadata_size=thresholds.get('metadata_size', 10_000),
            tag_count=thresholds.get('tag_count', 100),
            tag_length=thresholds.get('tag_length', 50),
            relationship_depth=thresholds.get('relationship_depth', 3),
            governance_timeout=thresholds.get('governance_timeout', 10),
            enabled=data.get('enabled', True),
            enforcement_mode=data.get('enforcement_mode', 'strict'),
            environment=data.get('environment', 'development')
        )
    
    def validate(self):
        """Validate configuration values"""
        for attr_name in ['content_size', 'summary_length', 'metadata_size', 
                         'tag_count', 'tag_length', 'relationship_depth', 'governance_timeout']:
            value = getattr(self, attr_name)
            if not isinstance(value, int) or value <= 0:
                raise GovernanceConfigError(f"Invalid threshold {attr_name}: {value}")
        
        if self.enforcement_mode not in ['strict', 'permissive', 'audit_only']:
            raise GovernanceConfigError(f"Invalid enforcement_mode: {self.enforcement_mode}")


class ThresholdManager:
    """Dynamic threshold adjustment and management"""
    
    def __init__(self, base_config: GovernanceConfig):
        self.base_config = base_config
        self.dynamic_adjustments = {}
        self.logger = get_error_logger("governance.threshold_manager")
    
    def get_effective_thresholds(self, context: Optional[str] = None) -> GovernanceConfig:
        """Get effective thresholds considering dynamic adjustments"""
        # Start with base configuration
        effective_config = self.base_config
        
        # Apply context-specific adjustments if available
        if context and context in self.dynamic_adjustments:
            adjustments = self.dynamic_adjustments[context]
            config_dict = {
                'content_size': effective_config.content_size,
                'summary_length': effective_config.summary_length,
                'metadata_size': effective_config.metadata_size,
                'tag_count': effective_config.tag_count,
                'tag_length': effective_config.tag_length,
                'relationship_depth': effective_config.relationship_depth,
                'governance_timeout': effective_config.governance_timeout,
                'enabled': effective_config.enabled,
                'enforcement_mode': effective_config.enforcement_mode,
                'environment': effective_config.environment
            }
            config_dict.update(adjustments)
            effective_config = GovernanceConfig.from_dict(config_dict)
        
        return effective_config
    
    def adjust_thresholds(self, context: str, adjustments: Dict[str, Any]):
        """Adjust thresholds for specific context"""
        self.dynamic_adjustments[context] = adjustments
        self.logger.info(f"Applied threshold adjustments for context '{context}': {adjustments}")
    
    def reset_adjustments(self, context: Optional[str] = None):
        """Reset threshold adjustments"""
        if context:
            self.dynamic_adjustments.pop(context, None)
            self.logger.info(f"Reset adjustments for context '{context}'")
        else:
            self.dynamic_adjustments.clear()
            self.logger.info("Reset all threshold adjustments")


class GovernanceMetrics:
    """Performance and compliance tracking for governance operations"""
    
    def __init__(self):
        self.metrics = {
            'total_validations': 0,
            'approvals': 0,
            'rejections': 0,
            'warnings': 0,
            'errors': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'violation_types': {},
            'performance_percentiles': {
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        }
        self.processing_times = []
        self.logger = get_error_logger("governance.metrics")
    
    def record_validation(self, result: GovernanceResult):
        """Record governance validation metrics"""
        self.metrics['total_validations'] += 1
        self.metrics['total_processing_time_ms'] += result.processing_time
        self.processing_times.append(result.processing_time)
        
        # Keep only last 1000 processing times for percentile calculation
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Update status counts
        if result.approved and not result.warnings:
            self.metrics['approvals'] += 1
        elif result.approved and result.warnings:
            self.metrics['approvals'] += 1
            self.metrics['warnings'] += 1
        else:
            self.metrics['rejections'] += 1
        
        if result.error:
            self.metrics['errors'] += 1
        
        # Track violation types
        for violation in result.violations:
            violation_type = violation.type
            if violation_type not in self.metrics['violation_types']:
                self.metrics['violation_types'][violation_type] = 0
            self.metrics['violation_types'][violation_type] += 1
        
        # Update average processing time
        self.metrics['avg_processing_time_ms'] = (
            self.metrics['total_processing_time_ms'] / self.metrics['total_validations']
        )
        
        # Update percentiles
        if self.processing_times:
            sorted_times = sorted(self.processing_times)
            count = len(sorted_times)
            self.metrics['performance_percentiles'] = {
                'p50': sorted_times[int(count * 0.5)] if count > 0 else 0.0,
                'p95': sorted_times[int(count * 0.95)] if count > 20 else sorted_times[-1] if count > 0 else 0.0,
                'p99': sorted_times[int(count * 0.99)] if count > 100 else sorted_times[-1] if count > 0 else 0.0
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_validations': self.metrics['total_validations'],
            'approval_rate': (
                self.metrics['approvals'] / self.metrics['total_validations'] * 100
                if self.metrics['total_validations'] > 0 else 0.0
            ),
            'rejection_rate': (
                self.metrics['rejections'] / self.metrics['total_validations'] * 100
                if self.metrics['total_validations'] > 0 else 0.0
            ),
            'warning_rate': (
                self.metrics['warnings'] / self.metrics['total_validations'] * 100
                if self.metrics['total_validations'] > 0 else 0.0
            ),
            'error_rate': (
                self.metrics['errors'] / self.metrics['total_validations'] * 100
                if self.metrics['total_validations'] > 0 else 0.0
            ),
            'performance': {
                'avg_processing_time_ms': self.metrics['avg_processing_time_ms'],
                'percentiles': self.metrics['performance_percentiles']
            },
            'violation_types': self.metrics['violation_types']
        }
    
    def reset_metrics(self):
        """Reset all metrics for testing/maintenance"""
        self.__init__()
        self.logger.info("Governance metrics reset")


class GovernanceValidator:
    """Real-time validation engine with <10ms performance target"""
    
    def __init__(self, config: GovernanceConfig, threshold_manager: Optional[ThresholdManager] = None):
        self.base_config = config
        self.threshold_manager = threshold_manager or ThresholdManager(config)
        self.metrics = GovernanceMetrics()
        self.logger = get_error_logger("governance.validator")
        
        # Pre-compile validation patterns for performance
        self._validation_cache = {}
        
        self.logger.info(f"Governance validator initialized with enforcement_mode: {config.enforcement_mode}")
    
    @handle_errors(catch=[Exception], reraise=False)
    async def validate_memory_governance(
        self,
        memory: EnhancedMemory,
        operation_context: str = "store"
    ) -> GovernanceResult:
        """
        Comprehensive governance validation with <10ms target.
        
        Args:
            memory: EnhancedMemory object to validate
            operation_context: Context of the operation (store, update, retrieve)
            
        Returns:
            GovernanceResult with validation outcome
        """
        start_time = time.perf_counter()
        
        # Get effective thresholds for this validation context
        thresholds = self.threshold_manager.get_effective_thresholds(operation_context)
        
        # Skip validation if governance is disabled
        if not thresholds.enabled:
            return GovernanceResult(
                approved=True,
                status="approved",
                violations=[],
                processing_time=(time.perf_counter() - start_time) * 1000,
                operation_context=operation_context,
                warnings=["Governance validation disabled"]
            )
        
        violations = []
        warnings = []
        
        try:
            # Performance timeout check
            timeout_start = time.perf_counter()
            
            # 1. Content size validation (most critical)
            content_size = len(memory.full_context or memory.content or "")
            if content_size > thresholds.content_size:
                violations.append(GovernanceViolation(
                    type="content_size_exceeded",
                    message=f"Content size {content_size} exceeds limit {thresholds.content_size}",
                    severity="critical",
                    value=content_size,
                    threshold=thresholds.content_size,
                    remediation="Reduce content size or increase limit"
                ))
            elif content_size > thresholds.content_size * 0.8:
                warnings.append(
                    f"Content size approaching limit: {content_size}/{thresholds.content_size} (80%)"
                )
            
            # 2. Tag validation
            if len(memory.tags) > thresholds.tag_count:
                violations.append(GovernanceViolation(
                    type="tag_count_exceeded", 
                    message=f"Tag count {len(memory.tags)} exceeds limit {thresholds.tag_count}",
                    severity="high",
                    value=len(memory.tags),
                    threshold=thresholds.tag_count,
                    remediation="Remove unnecessary tags or increase limit"
                ))
            
            # 3. Individual tag validation (with early exit for performance)
            for i, tag in enumerate(memory.tags):
                if len(tag) > thresholds.tag_length:
                    violations.append(GovernanceViolation(
                        type="tag_length_exceeded",
                        message=f"Tag '{tag[:20]}...' length {len(tag)} exceeds limit {thresholds.tag_length}",
                        severity="medium",
                        value=len(tag),
                        threshold=thresholds.tag_length,
                        remediation="Shorten tag name or increase limit"
                    ))
                
                # Performance check: if we're approaching timeout, stop tag validation
                if (time.perf_counter() - timeout_start) * 1000 > thresholds.governance_timeout * 0.7:
                    remaining_tags = len(memory.tags) - i - 1
                    if remaining_tags > 0:
                        warnings.append(f"Tag validation incomplete due to timeout, {remaining_tags} tags not checked")
                    break
            
            # 4. Summary validation
            if memory.summary and len(memory.summary) > thresholds.summary_length:
                violations.append(GovernanceViolation(
                    type="summary_length_exceeded",
                    message=f"Summary length {len(memory.summary)} exceeds limit {thresholds.summary_length}",
                    severity="medium",
                    value=len(memory.summary),
                    threshold=thresholds.summary_length,
                    remediation="Shorten summary or increase limit"
                ))
            
            # 5. Metadata validation
            metadata_size = len(json.dumps(memory.metadata)) if memory.metadata else 0
            if metadata_size > thresholds.metadata_size:
                violations.append(GovernanceViolation(
                    type="metadata_size_exceeded",
                    message=f"Metadata size {metadata_size} exceeds limit {thresholds.metadata_size}",
                    severity="low",
                    value=metadata_size,
                    threshold=thresholds.metadata_size,
                    remediation="Reduce metadata or increase limit"
                ))
            
            # Check processing time against governance timeout
            processing_time = (time.perf_counter() - start_time) * 1000
            if processing_time > thresholds.governance_timeout:
                warnings.append(f"Governance validation took {processing_time:.2f}ms > {thresholds.governance_timeout}ms timeout")
            
            # Determine overall status based on enforcement mode
            if thresholds.enforcement_mode == "audit_only":
                status = "approved"
                approved = True
                if violations:
                    warnings.extend([f"AUDIT: {v.message}" for v in violations])
            else:
                # strict or permissive mode
                critical_violations = [v for v in violations if v.severity == "critical"]
                high_violations = [v for v in violations if v.severity == "high"]
                
                if critical_violations:
                    status = "rejected"
                    approved = False
                elif high_violations and thresholds.enforcement_mode == "strict":
                    status = "rejected"
                    approved = False
                elif violations:
                    status = "approved_with_warnings"
                    approved = True
                    if thresholds.enforcement_mode == "permissive":
                        warnings.extend([f"PERMISSIVE: {v.message}" for v in violations])
                else:
                    status = "approved"
                    approved = True
            
            # Final warnings check
            if warnings and status == "approved":
                status = "approved_with_warnings"
            
            result = GovernanceResult(
                approved=approved,
                status=status,
                violations=violations,
                warnings=warnings,
                processing_time=processing_time,
                operation_context=operation_context,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Governance validation failed: {e}")
            
            result = GovernanceResult(
                approved=False,
                status="error",
                violations=[GovernanceViolation(
                    type="governance_error",
                    message=f"Governance validation failed: {e}",
                    severity="critical",
                    remediation="Check governance configuration and memory data"
                )],
                processing_time=processing_time,
                operation_context=operation_context,
                error=str(e)
            )
        
        # Record metrics
        self.metrics.record_validation(result)
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get governance performance metrics"""
        return self.metrics.get_performance_summary()
    
    def update_config(self, new_config: GovernanceConfig):
        """Update governance configuration with hot-reload capability"""
        self.base_config = new_config
        self.threshold_manager = ThresholdManager(new_config)
        self.logger.info(f"Governance configuration updated: enforcement_mode={new_config.enforcement_mode}")


class SimpleGovernance:
    """
    Simple Governance Framework with <10ms performance target.
    
    Provides lightweight, configurable governance with:
    - Threshold-based validation
    - Hot-reload configuration
    - Dynamic threshold adjustment
    - Performance metrics tracking
    - Multiple enforcement modes
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], GovernanceConfig]] = None):
        # Initialize configuration
        if isinstance(config, GovernanceConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = GovernanceConfig.from_dict(config)
        else:
            self.config = GovernanceConfig()  # Use defaults
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.threshold_manager = ThresholdManager(self.config)
        self.validator = GovernanceValidator(self.config, self.threshold_manager)
        self.logger = get_error_logger("governance.simple")
        
        self.logger.info(
            f"Simple Governance Framework initialized: "
            f"enforcement_mode={self.config.enforcement_mode}, "
            f"environment={self.config.environment}"
        )
    
    async def validate_memory(
        self,
        memory: EnhancedMemory,
        operation_context: str = "store"
    ) -> GovernanceResult:
        """Main validation entry point"""
        return await self.validator.validate_memory_governance(memory, operation_context)
    
    def get_thresholds(self, context: Optional[str] = None) -> GovernanceConfig:
        """Get effective thresholds for given context"""
        return self.threshold_manager.get_effective_thresholds(context)
    
    def adjust_thresholds(self, context: str, adjustments: Dict[str, Any]):
        """Dynamically adjust thresholds for specific context"""
        self.threshold_manager.adjust_thresholds(context, adjustments)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive governance metrics"""
        validator_metrics = self.validator.get_performance_metrics()
        
        return {
            'governance_framework': 'simple',
            'version': '2.0',
            'configuration': {
                'enabled': self.config.enabled,
                'enforcement_mode': self.config.enforcement_mode,
                'environment': self.config.environment,
                'governance_timeout_ms': self.config.governance_timeout
            },
            'thresholds': {
                'content_size': self.config.content_size,
                'summary_length': self.config.summary_length,
                'metadata_size': self.config.metadata_size,
                'tag_count': self.config.tag_count,
                'tag_length': self.config.tag_length
            },
            'performance': validator_metrics
        }
    
    def reload_config(self, new_config: Union[Dict[str, Any], GovernanceConfig]):
        """Hot-reload governance configuration"""
        if isinstance(new_config, dict):
            new_config = GovernanceConfig.from_dict(new_config)
        
        new_config.validate()
        self.config = new_config
        self.validator.update_config(new_config)
        self.threshold_manager = ThresholdManager(new_config)
        
        self.logger.info("Governance configuration reloaded successfully")
    
    @classmethod
    def from_yaml_file(cls, yaml_path: Union[str, Path]) -> 'SimpleGovernance':
        """Create governance framework from YAML configuration file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise GovernanceConfigError(f"Governance config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract governance section if nested
            if 'governance' in config_data:
                governance_config = config_data['governance']
            else:
                governance_config = config_data
            
            return cls(governance_config)
            
        except Exception as e:
            raise GovernanceConfigError(f"Failed to load governance config from {yaml_path}: {e}") from e
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"SimpleGovernance("
            f"enabled={self.config.enabled}, "
            f"mode={self.config.enforcement_mode}, "
            f"env={self.config.environment})"
        )