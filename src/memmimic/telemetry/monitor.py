"""
Live Performance Monitor for MemMimic v2.0

Real-time performance monitoring with intelligent thresholds and automated responses.
Provides continuous monitoring, threshold validation, and proactive performance management.
"""

import time
import threading
import psutil
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum

from ..errors import get_error_logger
from .collector import TelemetryCollector, get_telemetry_collector
from .aggregator import MetricsAggregator, get_metrics_aggregator

logger = get_error_logger("telemetry.monitor")


class ThresholdType(Enum):
    """Threshold types for different monitoring scenarios"""
    PERFORMANCE = "performance"
    RESOURCE = "resource" 
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    name: str
    operation: str
    metric: str  # p50, p95, p99, mean, etc.
    threshold_value: float
    threshold_type: ThresholdType
    severity: AlertSeverity
    enabled: bool = True
    grace_period_seconds: int = 60
    consecutive_violations_required: int = 3
    auto_recovery_enabled: bool = False
    recovery_action: Optional[str] = None


@dataclass 
class ThresholdViolation:
    """Threshold violation event"""
    threshold: PerformanceThreshold
    current_value: float
    violation_time: datetime
    violation_count: int
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResourceMetrics:
    """System resource usage metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_available_gb: float
    load_average_1m: Optional[float] = None
    load_average_5m: Optional[float] = None
    load_average_15m: Optional[float] = None


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: datetime
    system_resources: SystemResourceMetrics
    operation_metrics: Dict[str, Dict[str, float]]
    telemetry_overhead: Dict[str, float]
    active_violations: List[ThresholdViolation]
    health_score: float


class PerformanceMonitor:
    """
    Live Performance Monitor with intelligent thresholds and automated responses.
    
    Features:
    - Real-time threshold monitoring
    - Multi-dimensional performance tracking
    - System resource monitoring
    - Intelligent threshold adjustment
    - Proactive alerting and recovery
    - Performance trend analysis
    - Automated remediation actions
    """
    
    def __init__(
        self, 
        collector: Optional[TelemetryCollector] = None,
        aggregator: Optional[MetricsAggregator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.collector = collector or get_telemetry_collector()
        self.aggregator = aggregator or get_metrics_aggregator()
        self.config = config or {}
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 10.0)  # 10 seconds
        self.resource_monitoring_enabled = self.config.get('resource_monitoring_enabled', True)
        self.auto_threshold_adjustment = self.config.get('auto_threshold_adjustment', True)
        self.performance_history_size = self.config.get('performance_history_size', 1000)
        
        # Threshold management
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._violation_counters: Dict[str, int] = defaultdict(int)
        self._active_violations: Dict[str, ThresholdViolation] = {}
        self._violation_history: deque = deque(maxlen=1000)
        
        # Performance history
        self._performance_history: deque = deque(maxlen=self.performance_history_size)
        
        # System resource tracking
        self._resource_history: deque = deque(maxlen=1000)
        self._process = psutil.Process(os.getpid())
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Start monitoring
        self._start_background_monitoring()
        
        logger.info("PerformanceMonitor initialized")
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds based on MemMimic v2.0 targets"""
        default_thresholds = [
            # Core performance targets from PLAN.md
            PerformanceThreshold(
                name="summary_retrieval_p95",
                operation="storage_retrieve_summary_optimized",
                metric="p95_duration_ms",
                threshold_value=5.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=2
            ),
            PerformanceThreshold(
                name="summary_retrieval_p99", 
                operation="storage_retrieve_summary_optimized",
                metric="p99_duration_ms",
                threshold_value=10.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                consecutive_violations_required=1
            ),
            PerformanceThreshold(
                name="full_context_retrieval_p95",
                operation="storage_retrieve_full_context_optimized", 
                metric="p95_duration_ms",
                threshold_value=50.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=2
            ),
            PerformanceThreshold(
                name="enhanced_remember_p95",
                operation="memory_remember_with_context",
                metric="p95_duration_ms", 
                threshold_value=15.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=2
            ),
            PerformanceThreshold(
                name="governance_validation_p95",
                operation="governance_validate_memory_governance",
                metric="p95_duration_ms",
                threshold_value=10.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=2
            ),
            PerformanceThreshold(
                name="telemetry_overhead_p95",
                operation="telemetry_overhead",
                metric="p95_duration_ms",
                threshold_value=1.0,
                threshold_type=ThresholdType.PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                consecutive_violations_required=1
            ),
            # Resource thresholds
            PerformanceThreshold(
                name="system_memory_usage",
                operation="system_resources",
                metric="memory_percent",
                threshold_value=85.0,
                threshold_type=ThresholdType.RESOURCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=3
            ),
            PerformanceThreshold(
                name="system_cpu_usage",
                operation="system_resources", 
                metric="cpu_percent",
                threshold_value=80.0,
                threshold_type=ThresholdType.RESOURCE,
                severity=AlertSeverity.WARNING,
                consecutive_violations_required=3
            ),
            # Success rate thresholds
            PerformanceThreshold(
                name="operation_success_rate",
                operation="all_operations",
                metric="success_rate",
                threshold_value=0.95,  # 95% success rate
                threshold_type=ThresholdType.SUCCESS_RATE,
                severity=AlertSeverity.CRITICAL,
                consecutive_violations_required=1
            )
        ]
        
        for threshold in default_thresholds:
            self.add_threshold(threshold)
        
        logger.info(f"Initialized {len(default_thresholds)} default performance thresholds")
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add performance threshold"""
        with self._lock:
            self._thresholds[threshold.name] = threshold
            logger.info(f"Added threshold: {threshold.name} - {threshold.operation}:{threshold.metric} <= {threshold.threshold_value}")
    
    def remove_threshold(self, threshold_name: str) -> bool:
        """Remove performance threshold"""
        with self._lock:
            if threshold_name in self._thresholds:
                del self._thresholds[threshold_name]
                # Clean up related data
                if threshold_name in self._violation_counters:
                    del self._violation_counters[threshold_name]
                if threshold_name in self._active_violations:
                    del self._active_violations[threshold_name]
                logger.info(f"Removed threshold: {threshold_name}")
                return True
        return False
    
    def update_threshold(self, threshold_name: str, **updates) -> bool:
        """Update existing threshold"""
        with self._lock:
            if threshold_name in self._thresholds:
                threshold = self._thresholds[threshold_name]
                for key, value in updates.items():
                    if hasattr(threshold, key):
                        setattr(threshold, key, value)
                logger.info(f"Updated threshold {threshold_name}: {updates}")
                return True
        return False
    
    def get_threshold(self, threshold_name: str) -> Optional[PerformanceThreshold]:
        """Get threshold by name"""
        with self._lock:
            return self._thresholds.get(threshold_name)
    
    def collect_system_resources(self) -> SystemResourceMetrics:
        """Collect current system resource metrics"""
        try:
            # System-wide metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Load average (Unix/Linux only)
            load_avg_1m = load_avg_5m = load_avg_15m = None
            try:
                load_avg = psutil.getloadavg()
                load_avg_1m, load_avg_5m, load_avg_15m = load_avg
            except AttributeError:
                pass  # Windows doesn't have load average
            
            return SystemResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_used_gb=disk.used / 1024 / 1024 / 1024,
                disk_available_gb=disk.free / 1024 / 1024 / 1024,
                load_average_1m=load_avg_1m,
                load_average_5m=load_avg_5m,
                load_average_15m=load_avg_15m
            )
        
        except Exception as e:
            logger.error(f"Failed to collect system resources: {e}")
            # Return minimal metrics
            return SystemResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=1024.0,
                disk_usage_percent=0.0,
                disk_used_gb=0.0,
                disk_available_gb=100.0
            )
    
    def check_thresholds(self) -> List[ThresholdViolation]:
        """Check all thresholds and return violations"""
        violations = []
        current_time = datetime.now()
        
        with self._lock:
            for threshold in self._thresholds.values():
                if not threshold.enabled:
                    continue
                
                violation = self._check_individual_threshold(threshold, current_time)
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_individual_threshold(self, threshold: PerformanceThreshold, check_time: datetime) -> Optional[ThresholdViolation]:
        """Check individual threshold for violations"""
        try:
            current_value = self._get_current_metric_value(threshold)
            if current_value is None:
                return None
            
            # Check if threshold is violated
            is_violation = False
            if threshold.threshold_type in [ThresholdType.PERFORMANCE, ThresholdType.RESOURCE]:
                is_violation = current_value > threshold.threshold_value
            elif threshold.threshold_type in [ThresholdType.SUCCESS_RATE]:
                is_violation = current_value < threshold.threshold_value
            elif threshold.threshold_type == ThresholdType.ERROR_RATE:
                is_violation = current_value > threshold.threshold_value
            
            if is_violation:
                # Increment violation counter
                self._violation_counters[threshold.name] += 1
                
                # Check if consecutive violations threshold is met
                if self._violation_counters[threshold.name] >= threshold.consecutive_violations_required:
                    violation = ThresholdViolation(
                        threshold=threshold,
                        current_value=current_value,
                        violation_time=check_time,
                        violation_count=self._violation_counters[threshold.name],
                        message=f"Threshold '{threshold.name}' violated: {current_value} vs {threshold.threshold_value}",
                        metadata={
                            'operation': threshold.operation,
                            'metric': threshold.metric,
                            'threshold_type': threshold.threshold_type.value,
                            'severity': threshold.severity.value
                        }
                    )
                    
                    # Store active violation
                    self._active_violations[threshold.name] = violation
                    self._violation_history.append(violation)
                    
                    logger.warning(f"Threshold violation: {violation.message}")
                    return violation
            else:
                # Reset violation counter and clear active violation if resolved
                if threshold.name in self._violation_counters:
                    self._violation_counters[threshold.name] = 0
                if threshold.name in self._active_violations:
                    resolved_violation = self._active_violations.pop(threshold.name)
                    logger.info(f"Threshold violation resolved: {threshold.name}")
        
        except Exception as e:
            logger.error(f"Failed to check threshold {threshold.name}: {e}")
        
        return None
    
    def _get_current_metric_value(self, threshold: PerformanceThreshold) -> Optional[float]:
        """Get current value for threshold metric"""
        if threshold.operation == "system_resources":
            # System resource metrics
            if not hasattr(self, '_last_resource_metrics'):
                return None
            
            resources = self._last_resource_metrics
            if threshold.metric == "cpu_percent":
                return resources.cpu_percent
            elif threshold.metric == "memory_percent":
                return resources.memory_percent
            elif threshold.metric == "disk_usage_percent":
                return resources.disk_usage_percent
            
        elif threshold.operation == "telemetry_overhead":
            # Telemetry overhead metrics
            system_stats = self.collector.get_system_stats()
            overhead_stats = system_stats.get('overhead_stats', {})
            if threshold.metric == "p95_duration_ms":
                return overhead_stats.get('p95_ms', 0.0)
            elif threshold.metric == "p99_duration_ms":
                return overhead_stats.get('p99_ms', 0.0)
            elif threshold.metric == "mean_duration_ms":
                return overhead_stats.get('mean_ms', 0.0)
        
        elif threshold.operation == "all_operations":
            # Global operation metrics
            system_stats = self.collector.get_system_stats()
            if threshold.metric == "success_rate":
                return system_stats.get('success_rate_percent', 100.0) / 100.0
        
        else:
            # Operation-specific metrics
            op_stats = self.collector.get_operation_stats(threshold.operation)
            if not op_stats:
                return None
            
            if threshold.metric == "p50_duration_ms":
                return op_stats.get('p50_duration_ms', 0.0)
            elif threshold.metric == "p95_duration_ms":
                return op_stats.get('p95_duration_ms', 0.0)
            elif threshold.metric == "p99_duration_ms":
                return op_stats.get('p99_duration_ms', 0.0)
            elif threshold.metric == "mean_duration_ms":
                return op_stats.get('mean_duration_ms', 0.0)
            elif threshold.metric == "success_rate":
                return op_stats.get('success_rate', 1.0)
            elif threshold.metric == "error_rate":
                return 1.0 - op_stats.get('success_rate', 1.0)
        
        return None
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get comprehensive performance snapshot"""
        # Collect system resources
        system_resources = self.collect_system_resources()
        
        # Collect operation metrics
        operation_metrics = {}
        for operation in self.collector._operation_buffers.keys():
            op_stats = self.collector.get_operation_stats(operation)
            if op_stats:
                operation_metrics[operation] = {
                    'p50_ms': op_stats.get('p50_duration_ms', 0.0),
                    'p95_ms': op_stats.get('p95_duration_ms', 0.0),
                    'p99_ms': op_stats.get('p99_duration_ms', 0.0),
                    'mean_ms': op_stats.get('mean_duration_ms', 0.0),
                    'success_rate': op_stats.get('success_rate', 1.0),
                    'total_count': op_stats.get('total_count', 0),
                    'error_count': op_stats.get('error_count', 0)
                }
        
        # Get telemetry overhead metrics
        system_stats = self.collector.get_system_stats()
        telemetry_overhead = system_stats.get('overhead_stats', {})
        
        # Check for active violations
        violations = self.check_thresholds()
        
        # Calculate health score
        health_score = self._calculate_health_score(system_resources, operation_metrics, violations)
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            system_resources=system_resources,
            operation_metrics=operation_metrics,
            telemetry_overhead=telemetry_overhead,
            active_violations=violations,
            health_score=health_score
        )
        
        # Store in history
        with self._lock:
            self._performance_history.append(snapshot)
            self._resource_history.append(system_resources)
            self._last_resource_metrics = system_resources
        
        return snapshot
    
    def _calculate_health_score(
        self, 
        resources: SystemResourceMetrics, 
        operations: Dict[str, Dict[str, float]], 
        violations: List[ThresholdViolation]
    ) -> float:
        """Calculate overall system health score (0-100)"""
        base_score = 100.0
        
        # Penalties for resource usage
        if resources.memory_percent > 90:
            base_score -= 20
        elif resources.memory_percent > 80:
            base_score -= 10
        elif resources.memory_percent > 70:
            base_score -= 5
        
        if resources.cpu_percent > 90:
            base_score -= 15
        elif resources.cpu_percent > 80:
            base_score -= 8
        elif resources.cpu_percent > 70:
            base_score -= 3
        
        # Penalties for performance violations
        for violation in violations:
            if violation.threshold.severity == AlertSeverity.EMERGENCY:
                base_score -= 30
            elif violation.threshold.severity == AlertSeverity.CRITICAL:
                base_score -= 20
            elif violation.threshold.severity == AlertSeverity.WARNING:
                base_score -= 10
            else:
                base_score -= 5
        
        # Penalties for poor operation performance
        critical_operations = ['storage_retrieve_summary_optimized', 'memory_remember_with_context']
        for op_name in critical_operations:
            if op_name in operations:
                op_metrics = operations[op_name]
                success_rate = op_metrics.get('success_rate', 1.0)
                if success_rate < 0.95:
                    base_score -= (0.95 - success_rate) * 100
        
        # Penalty for telemetry overhead
        system_stats = self.collector.get_system_stats()
        overhead_stats = system_stats.get('overhead_stats', {})
        if not overhead_stats.get('target_met', True):
            base_score -= 15
        
        return max(0.0, min(100.0, base_score))
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        current_snapshot = self.get_performance_snapshot()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'health_score': current_snapshot.health_score,
            'health_status': self._get_health_status(current_snapshot.health_score),
            'active_violations': [
                {
                    'name': v.threshold.name,
                    'severity': v.threshold.severity.value,
                    'message': v.message,
                    'current_value': v.current_value,
                    'threshold_value': v.threshold.threshold_value,
                    'violation_count': v.violation_count
                }
                for v in current_snapshot.active_violations
            ],
            'system_resources': {
                'cpu_percent': current_snapshot.system_resources.cpu_percent,
                'memory_percent': current_snapshot.system_resources.memory_percent,
                'disk_usage_percent': current_snapshot.system_resources.disk_usage_percent,
                'memory_used_mb': current_snapshot.system_resources.memory_used_mb,
                'memory_available_mb': current_snapshot.system_resources.memory_available_mb
            },
            'performance_metrics': current_snapshot.operation_metrics,
            'telemetry_overhead': current_snapshot.telemetry_overhead,
            'threshold_summary': {
                'total_thresholds': len(self._thresholds),
                'active_thresholds': len([t for t in self._thresholds.values() if t.enabled]),
                'violated_thresholds': len(current_snapshot.active_violations)
            },
            'performance_targets': {
                'summary_retrieval_5ms': self._check_target('storage_retrieve_summary_optimized', 'p95_duration_ms', 5.0),
                'full_context_50ms': self._check_target('storage_retrieve_full_context_optimized', 'p95_duration_ms', 50.0),
                'enhanced_remember_15ms': self._check_target('memory_remember_with_context', 'p95_duration_ms', 15.0),
                'governance_10ms': self._check_target('governance_validate_memory_governance', 'p95_duration_ms', 10.0),
                'telemetry_overhead_1ms': current_snapshot.telemetry_overhead.get('target_met', False)
            }
        }
        
        return summary
    
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status string"""
        if health_score >= 95:
            return 'excellent'
        elif health_score >= 85:
            return 'good'
        elif health_score >= 70:
            return 'fair'
        elif health_score >= 50:
            return 'poor'
        else:
            return 'critical'
    
    def _check_target(self, operation: str, metric: str, target_value: float) -> bool:
        """Check if specific target is being met"""
        op_stats = self.collector.get_operation_stats(operation)
        if not op_stats:
            return False
        
        current_value = op_stats.get(metric, float('inf'))
        return current_value <= target_value
    
    def get_performance_history(self, minutes: int = 60) -> List[PerformanceSnapshot]:
        """Get performance history for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                snapshot for snapshot in self._performance_history
                if snapshot.timestamp >= cutoff_time
            ]
    
    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        def monitoring_worker():
            while not self._stop_monitoring.wait(self.monitoring_interval):
                try:
                    # Collect performance snapshot
                    snapshot = self.get_performance_snapshot()
                    
                    # Log significant issues
                    if snapshot.health_score < 70:
                        logger.warning(f"Performance health degraded: {snapshot.health_score:.1f}/100")
                    
                    if snapshot.active_violations:
                        logger.warning(f"Active threshold violations: {len(snapshot.active_violations)}")
                
                except Exception as e:
                    logger.error(f"Background performance monitoring failed: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self._monitoring_thread.start()
        logger.info("Background performance monitoring started")
    
    def shutdown(self):
        """Shutdown performance monitor"""
        try:
            self._stop_monitoring.set()
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            logger.info("PerformanceMonitor shutdown completed")
            
        except Exception as e:
            logger.error(f"PerformanceMonitor shutdown failed: {e}")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor()
    
    return _global_monitor