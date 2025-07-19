#!/usr/bin/env python3
"""
MCP Performance Monitor - Enhanced Monitoring for MemMimic MCP Tools
Real-time performance tracking, metrics collection, and optimization insights
"""

import json
import logging
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PerformanceMetrics:
    """Performance metrics for MCP operations"""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Operation-specific metrics
    memories_processed: int = 0
    results_returned: int = 0
    database_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Quality metrics
    relevance_score: Optional[float] = None
    user_satisfaction: Optional[float] = None

    def finalize(self):
        """Finalize metrics calculation"""
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class OperationStats:
    """Aggregated statistics for an operation type"""

    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    median_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    # Recent performance (last 100 operations)
    recent_durations: List[float] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)

    # Resource usage
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0

    # Quality metrics
    avg_relevance_score: float = 0.0
    avg_user_satisfaction: float = 0.0

    def update(self, metrics: PerformanceMetrics):
        """Update stats with new metrics"""
        self.total_calls += 1

        if metrics.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if metrics.error_message:
                self.recent_errors.append(metrics.error_message)
                if len(self.recent_errors) > 20:  # Keep last 20 errors
                    self.recent_errors.pop(0)

        if metrics.duration_ms:
            self.total_duration_ms += metrics.duration_ms
            self.min_duration_ms = min(self.min_duration_ms, metrics.duration_ms)
            self.max_duration_ms = max(self.max_duration_ms, metrics.duration_ms)

            # Update recent durations for percentile calculation
            self.recent_durations.append(metrics.duration_ms)
            if len(self.recent_durations) > 100:  # Keep last 100 operations
                self.recent_durations.pop(0)

            # Calculate percentiles
            if self.recent_durations:
                sorted_durations = sorted(self.recent_durations)
                self.median_duration_ms = statistics.median(sorted_durations)
                self.p95_duration_ms = (
                    statistics.quantiles(sorted_durations, n=20)[18]
                    if len(sorted_durations) >= 20
                    else sorted_durations[-1]
                )
                self.p99_duration_ms = (
                    statistics.quantiles(sorted_durations, n=100)[98]
                    if len(sorted_durations) >= 100
                    else sorted_durations[-1]
                )

        # Update averages
        if self.successful_calls > 0:
            self.avg_duration_ms = self.total_duration_ms / self.successful_calls

        # Resource usage
        if metrics.memory_usage_mb:
            self.avg_memory_mb = (
                (self.avg_memory_mb * (self.total_calls - 1)) + metrics.memory_usage_mb
            ) / self.total_calls

        if metrics.cpu_usage_percent:
            self.avg_cpu_percent = (
                (self.avg_cpu_percent * (self.total_calls - 1))
                + metrics.cpu_usage_percent
            ) / self.total_calls


class MCPPerformanceMonitor:
    """
    Comprehensive performance monitoring system for MemMimic MCP tools

    Features:
    - Real-time performance tracking
    - Aggregated statistics with percentiles
    - Resource usage monitoring
    - Error tracking and analysis
    - Performance alerts and recommendations
    - Metrics export for dashboards
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Cache directory for metrics storage
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = (
                Path(__file__).parent.parent / "memmimic_cache" / "mcp_metrics"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory metrics storage
        self.operation_stats: Dict[str, OperationStats] = {}
        self.active_operations: Dict[str, PerformanceMetrics] = {}

        # Performance thresholds
        self.performance_thresholds = {
            "warning_duration_ms": 500,
            "critical_duration_ms": 2000,
            "error_rate_threshold": 0.05,  # 5%
            "memory_warning_mb": 100,
            "memory_critical_mb": 500,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Load existing metrics
        self._load_metrics()

        self.logger.info("MCP Performance Monitor initialized")

    def start_operation(
        self, operation_name: str, operation_id: Optional[str] = None
    ) -> str:
        """
        Start tracking an operation

        Args:
            operation_name: Name of the operation (e.g., 'recall_cxd', 'status')
            operation_id: Optional unique ID for this operation instance

        Returns:
            Unique operation ID for tracking
        """
        if not operation_id:
            operation_id = f"{operation_name}_{int(time.time()*1000000)}"

        with self._lock:
            # Get current resource usage
            memory_mb = self._get_memory_usage()
            cpu_percent = self._get_cpu_usage()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=time.perf_counter(),
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
            )

            self.active_operations[operation_id] = metrics

            # Initialize operation stats if not exists
            if operation_name not in self.operation_stats:
                self.operation_stats[operation_name] = OperationStats(operation_name)

        return operation_id

    def finish_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Finish tracking an operation

        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            error_message: Error message if failed
            **kwargs: Additional metrics (memories_processed, results_returned, etc.)

        Returns:
            Completed performance metrics
        """
        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(
                    f"Operation {operation_id} not found in active operations"
                )
                return None

            metrics = self.active_operations[operation_id]

            # Finalize metrics
            metrics.end_time = time.perf_counter()
            metrics.success = success
            metrics.error_message = error_message
            metrics.finalize()

            # Update with additional metrics
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)

            # Update operation stats
            self.operation_stats[metrics.operation_name].update(metrics)

            # Remove from active operations
            del self.active_operations[operation_id]

            # Check for performance alerts
            self._check_performance_alerts(metrics)

            # Periodic metrics save
            if self.operation_stats[metrics.operation_name].total_calls % 10 == 0:
                self._save_metrics()

            return metrics

    def get_operation_stats(
        self, operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics

        Args:
            operation_name: Specific operation to get stats for, or None for all

        Returns:
            Performance statistics dictionary
        """
        with self._lock:
            if operation_name:
                if operation_name in self.operation_stats:
                    return self._format_operation_stats(
                        self.operation_stats[operation_name]
                    )
                else:
                    return {}
            else:
                return {
                    name: self._format_operation_stats(stats)
                    for name, stats in self.operation_stats.items()
                }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        with self._lock:
            total_calls = sum(
                stats.total_calls for stats in self.operation_stats.values()
            )
            total_errors = sum(
                stats.failed_calls for stats in self.operation_stats.values()
            )

            if total_calls == 0:
                return {
                    "status": "NO_DATA",
                    "total_operations": 0,
                    "error_rate": 0.0,
                    "avg_response_time_ms": 0.0,
                    "active_operations": len(self.active_operations),
                    "recommendations": [],
                }

            error_rate = total_errors / total_calls
            avg_response_time = sum(
                stats.avg_duration_ms for stats in self.operation_stats.values()
            ) / len(self.operation_stats)

            # Determine system status
            status = "HEALTHY"
            if error_rate > self.performance_thresholds["error_rate_threshold"]:
                status = "DEGRADED"
            if avg_response_time > self.performance_thresholds["critical_duration_ms"]:
                status = "CRITICAL"
            elif avg_response_time > self.performance_thresholds["warning_duration_ms"]:
                status = "WARNING"

            # Generate recommendations
            recommendations = []
            if error_rate > self.performance_thresholds["error_rate_threshold"]:
                recommendations.append(
                    "High error rate detected - investigate failed operations"
                )
            if avg_response_time > self.performance_thresholds["warning_duration_ms"]:
                recommendations.append(
                    "Response times elevated - consider optimization"
                )

            return {
                "status": status,
                "total_operations": total_calls,
                "error_rate": error_rate,
                "avg_response_time_ms": avg_response_time,
                "active_operations": len(self.active_operations),
                "recommendations": recommendations,
                "operation_count": len(self.operation_stats),
            }

    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export metrics in specified format

        Args:
            format_type: Format to export ('json', 'csv', 'prometheus')

        Returns:
            Formatted metrics string
        """
        with self._lock:
            if format_type == "json":
                return json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "system_health": self.get_system_health(),
                        "operation_stats": self.get_operation_stats(),
                        "thresholds": self.performance_thresholds,
                    },
                    indent=2,
                )
            elif format_type == "prometheus":
                return self._export_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported format: {format_type}")

    def _format_operation_stats(self, stats: OperationStats) -> Dict[str, Any]:
        """Format operation stats for output"""
        success_rate = (
            stats.successful_calls / stats.total_calls if stats.total_calls > 0 else 0.0
        )

        return {
            "operation_name": stats.operation_name,
            "total_calls": stats.total_calls,
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate,
            "performance": {
                "avg_duration_ms": stats.avg_duration_ms,
                "median_duration_ms": stats.median_duration_ms,
                "p95_duration_ms": stats.p95_duration_ms,
                "p99_duration_ms": stats.p99_duration_ms,
                "min_duration_ms": (
                    stats.min_duration_ms
                    if stats.min_duration_ms != float("inf")
                    else 0
                ),
                "max_duration_ms": stats.max_duration_ms,
            },
            "resource_usage": {
                "avg_memory_mb": stats.avg_memory_mb,
                "avg_cpu_percent": stats.avg_cpu_percent,
            },
            "quality": {
                "avg_relevance_score": stats.avg_relevance_score,
                "avg_user_satisfaction": stats.avg_user_satisfaction,
            },
            "recent_errors": stats.recent_errors[-5:] if stats.recent_errors else [],
            "last_updated": datetime.now().isoformat(),
        }

    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts and log warnings"""
        if not metrics.success:
            self.logger.warning(
                f"Operation {metrics.operation_name} failed: {metrics.error_message}"
            )

        if metrics.duration_ms:
            if (
                metrics.duration_ms
                > self.performance_thresholds["critical_duration_ms"]
            ):
                self.logger.error(
                    f"CRITICAL: {metrics.operation_name} took {metrics.duration_ms:.1f}ms"
                )
            elif (
                metrics.duration_ms > self.performance_thresholds["warning_duration_ms"]
            ):
                self.logger.warning(
                    f"SLOW: {metrics.operation_name} took {metrics.duration_ms:.1f}ms"
                )

        if metrics.memory_usage_mb:
            if (
                metrics.memory_usage_mb
                > self.performance_thresholds["memory_critical_mb"]
            ):
                self.logger.error(
                    f"CRITICAL: {metrics.operation_name} used {metrics.memory_usage_mb:.1f}MB memory"
                )
            elif (
                metrics.memory_usage_mb
                > self.performance_thresholds["memory_warning_mb"]
            ):
                self.logger.warning(
                    f"HIGH MEMORY: {metrics.operation_name} used {metrics.memory_usage_mb:.1f}MB"
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
        except Exception as e:
            self.logger.debug(f"Memory usage check failed: {e}")
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
        except Exception as e:
            self.logger.debug(f"CPU usage check failed: {e}")
            return 0.0

    def _save_metrics(self):
        """Save metrics to disk"""
        try:
            metrics_file = self.cache_dir / "mcp_performance_metrics.json"

            # Convert to serializable format
            serializable_stats = {}
            for name, stats in self.operation_stats.items():
                serializable_stats[name] = {
                    "operation_name": stats.operation_name,
                    "total_calls": stats.total_calls,
                    "successful_calls": stats.successful_calls,
                    "failed_calls": stats.failed_calls,
                    "total_duration_ms": stats.total_duration_ms,
                    "min_duration_ms": (
                        stats.min_duration_ms
                        if stats.min_duration_ms != float("inf")
                        else 0
                    ),
                    "max_duration_ms": stats.max_duration_ms,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "median_duration_ms": stats.median_duration_ms,
                    "p95_duration_ms": stats.p95_duration_ms,
                    "p99_duration_ms": stats.p99_duration_ms,
                    "recent_durations": stats.recent_durations[-50:],  # Keep last 50
                    "recent_errors": stats.recent_errors[-10:],  # Keep last 10
                    "avg_memory_mb": stats.avg_memory_mb,
                    "avg_cpu_percent": stats.avg_cpu_percent,
                    "avg_relevance_score": stats.avg_relevance_score,
                    "avg_user_satisfaction": stats.avg_user_satisfaction,
                }

            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "operation_stats": serializable_stats,
                    },
                    f,
                    indent=2,
                )

            self.logger.debug(f"Metrics saved to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def _load_metrics(self):
        """Load metrics from disk"""
        try:
            metrics_file = self.cache_dir / "mcp_performance_metrics.json"

            if not metrics_file.exists():
                return

            with open(metrics_file, "r") as f:
                data = json.load(f)

            # Restore operation stats
            for name, stats_data in data.get("operation_stats", {}).items():
                stats = OperationStats(name)

                # Restore basic stats
                stats.total_calls = stats_data.get("total_calls", 0)
                stats.successful_calls = stats_data.get("successful_calls", 0)
                stats.failed_calls = stats_data.get("failed_calls", 0)
                stats.total_duration_ms = stats_data.get("total_duration_ms", 0.0)
                stats.min_duration_ms = stats_data.get("min_duration_ms", float("inf"))
                stats.max_duration_ms = stats_data.get("max_duration_ms", 0.0)
                stats.avg_duration_ms = stats_data.get("avg_duration_ms", 0.0)
                stats.median_duration_ms = stats_data.get("median_duration_ms", 0.0)
                stats.p95_duration_ms = stats_data.get("p95_duration_ms", 0.0)
                stats.p99_duration_ms = stats_data.get("p99_duration_ms", 0.0)
                stats.recent_durations = stats_data.get("recent_durations", [])
                stats.recent_errors = stats_data.get("recent_errors", [])
                stats.avg_memory_mb = stats_data.get("avg_memory_mb", 0.0)
                stats.avg_cpu_percent = stats_data.get("avg_cpu_percent", 0.0)
                stats.avg_relevance_score = stats_data.get("avg_relevance_score", 0.0)
                stats.avg_user_satisfaction = stats_data.get(
                    "avg_user_satisfaction", 0.0
                )

                self.operation_stats[name] = stats

            self.logger.info(
                f"Loaded metrics for {len(self.operation_stats)} operations"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load metrics: {e}")

    def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Add help and type information
        lines.append(
            "# HELP mcp_operation_duration_seconds Time spent on MCP operations"
        )
        lines.append("# TYPE mcp_operation_duration_seconds histogram")

        lines.append("# HELP mcp_operation_total Total number of MCP operations")
        lines.append("# TYPE mcp_operation_total counter")

        lines.append(
            "# HELP mcp_operation_errors_total Total number of MCP operation errors"
        )
        lines.append("# TYPE mcp_operation_errors_total counter")

        # Export metrics for each operation
        for name, stats in self.operation_stats.items():
            # Operation totals
            lines.append(
                f'mcp_operation_total{{operation="{name}"}} {stats.total_calls}'
            )
            lines.append(
                f'mcp_operation_errors_total{{operation="{name}"}} {stats.failed_calls}'
            )

            # Duration metrics
            lines.append(
                f'mcp_operation_duration_seconds{{operation="{name}",quantile="0.5"}} {stats.median_duration_ms/1000}'
            )
            lines.append(
                f'mcp_operation_duration_seconds{{operation="{name}",quantile="0.95"}} {stats.p95_duration_ms/1000}'
            )
            lines.append(
                f'mcp_operation_duration_seconds{{operation="{name}",quantile="0.99"}} {stats.p99_duration_ms/1000}'
            )

        return "\n".join(lines)


# Global monitor instance
_monitor_instance = None


def get_performance_monitor() -> MCPPerformanceMonitor:
    """Get global performance monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = MCPPerformanceMonitor()
    return _monitor_instance


def track_operation(operation_name: str):
    """Decorator to track MCP operation performance"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            operation_id = monitor.start_operation(operation_name)

            try:
                result = func(*args, **kwargs)

                # Extract metrics from result if available
                extra_metrics = {}
                if isinstance(result, dict):
                    if "memories_processed" in result:
                        extra_metrics["memories_processed"] = result[
                            "memories_processed"
                        ]
                    if "results_returned" in result:
                        extra_metrics["results_returned"] = result["results_returned"]

                monitor.finish_operation(operation_id, success=True, **extra_metrics)
                return result

            except Exception as e:
                monitor.finish_operation(
                    operation_id, success=False, error_message=str(e)
                )
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    # Test the performance monitor
    monitor = MCPPerformanceMonitor()

    # Simulate some operations
    for i in range(5):
        op_id = monitor.start_operation("test_operation")
        time.sleep(0.1)  # Simulate work
        monitor.finish_operation(
            op_id, success=True, memories_processed=10, results_returned=5
        )

    # Get stats
    stats = monitor.get_operation_stats()
    print(json.dumps(stats, indent=2))

    # System health
    health = monitor.get_system_health()
    print(f"System health: {health}")
