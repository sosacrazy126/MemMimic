"""
Real-time performance monitoring and health tracking for active memory management.

Provides comprehensive performance monitoring, threshold detection, alerting,
and historical data collection for optimal system health tracking.
"""

import logging
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import statistics

from .interfaces import (
    PerformanceMonitor, PerformanceSnapshot, PerformanceError,
    ThreadSafeMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration"""
    # Query performance
    max_avg_query_time_ms: float = 100.0
    max_query_time_ms: float = 1000.0
    min_queries_per_second: float = 10.0
    max_error_rate: float = 0.05  # 5%
    
    # Memory usage
    max_memory_utilization: float = 0.85  # 85%
    max_cache_memory_mb: float = 1024.0
    min_cache_hit_rate: float = 0.8  # 80%
    
    # Database performance
    max_db_response_time_ms: float = 50.0
    max_active_connections: int = 15
    
    # System health
    min_health_score: float = 0.7  # 70%


@dataclass
class PerformanceAlert:
    """Performance alert information"""
    alert_type: str
    severity: str  # 'warning', 'error', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
        }


class RealTimePerformanceMonitor(PerformanceMonitor):
    """
    Real-time performance monitoring with alerting and historical tracking.
    
    Features:
    - Real-time metrics collection and aggregation
    - Configurable performance thresholds
    - Alert generation and notification
    - Historical data retention and analysis
    - Performance trend detection
    - Health score calculation
    """
    
    def __init__(self, 
                 thresholds: Optional[PerformanceThresholds] = None,
                 history_retention_hours: int = 24,
                 snapshot_interval_seconds: int = 60,
                 alert_callback: Optional[Callable[[PerformanceAlert], None]] = None):
        """
        Initialize performance monitor.
        
        Args:
            thresholds: Performance thresholds configuration
            history_retention_hours: How long to retain historical data
            snapshot_interval_seconds: Interval for creating snapshots
            alert_callback: Optional callback for alert notifications
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.history_retention_hours = history_retention_hours
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.alert_callback = alert_callback
        
        # Real-time metrics
        self._metrics = ThreadSafeMetrics()
        self._lock = threading.RLock()
        
        # Historical data
        self._snapshots: deque[PerformanceSnapshot] = deque()
        self._max_snapshots = (history_retention_hours * 3600) // snapshot_interval_seconds
        
        # Query tracking
        self._recent_queries: deque[Tuple[float, bool]] = deque(maxlen=1000)  # (time_ms, success)
        self._query_times_window: deque[float] = deque(maxlen=100)  # Last 100 query times
        
        # Cache tracking
        self._cache_operations: deque[Tuple[float, bool]] = deque(maxlen=1000)  # (time_ms, hit)
        
        # Memory tracking
        self._memory_usage: Dict[str, float] = {}
        
        # Alert management
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: deque[PerformanceAlert] = deque(maxlen=1000)
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._start_background_monitoring()
        
        # Performance calculation cache
        self._last_performance_calculation = 0
        self._cached_performance_data = {}
        
        logger.info("RealTimePerformanceMonitor initialized")
    
    def record_query(self, query_time_ms: float, success: bool) -> None:
        """
        Record query performance metrics.
        
        Args:
            query_time_ms: Query execution time in milliseconds
            success: Whether query was successful
        """
        try:
            with self._lock:
                current_time = time.time()
                
                # Record query
                self._recent_queries.append((query_time_ms, success))
                if success:
                    self._query_times_window.append(query_time_ms)
                
                # Update metrics
                self._metrics.increment_counter('total_queries')
                if success:
                    self._metrics.increment_counter('successful_queries')
                    self._metrics.set_gauge('last_query_time_ms', query_time_ms)
                else:
                    self._metrics.increment_counter('failed_queries')
                
                # Check thresholds
                if query_time_ms > self.thresholds.max_query_time_ms:
                    self._trigger_alert(
                        'slow_query',
                        'warning',
                        f'Slow query detected: {query_time_ms:.2f}ms',
                        'query_time_ms',
                        query_time_ms,
                        self.thresholds.max_query_time_ms
                    )
                
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
    
    def record_cache_operation(self, hit: bool, operation_time_ms: float) -> None:
        """
        Record cache operation metrics.
        
        Args:
            hit: Whether operation was a cache hit
            operation_time_ms: Operation time in milliseconds
        """
        try:
            with self._lock:
                # Record operation
                self._cache_operations.append((operation_time_ms, hit))
                
                # Update metrics
                self._metrics.increment_counter('total_cache_operations')
                if hit:
                    self._metrics.increment_counter('cache_hits')
                else:
                    self._metrics.increment_counter('cache_misses')
                
                self._metrics.set_gauge('last_cache_operation_time_ms', operation_time_ms)
                
        except Exception as e:
            logger.error(f"Failed to record cache metrics: {e}")
    
    def record_memory_usage(self, component: str, memory_mb: float) -> None:
        """
        Record memory usage for component.
        
        Args:
            component: Component name
            memory_mb: Memory usage in MB
        """
        try:
            with self._lock:
                self._memory_usage[component] = memory_mb
                self._metrics.set_gauge(f'memory_usage_{component}_mb', memory_mb)
                
                # Calculate total memory
                total_memory = sum(self._memory_usage.values())
                self._metrics.set_gauge('total_memory_mb', total_memory)
                
                # Check thresholds
                if memory_mb > self.thresholds.max_cache_memory_mb:
                    self._trigger_alert(
                        'high_memory_usage',
                        'warning',
                        f'High memory usage in {component}: {memory_mb:.1f}MB',
                        f'memory_usage_{component}',
                        memory_mb,
                        self.thresholds.max_cache_memory_mb
                    )
                
        except Exception as e:
            logger.error(f"Failed to record memory usage: {e}")
    
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """
        Get current performance snapshot.
        
        Returns:
            Current PerformanceSnapshot
        """
        try:
            with self._lock:
                # Calculate current performance metrics
                current_time = time.time()
                
                # Use cached data if recent enough (within 5 seconds)
                if (current_time - self._last_performance_calculation < 5 and 
                    self._cached_performance_data):
                    return PerformanceSnapshot(**self._cached_performance_data)
                
                # Query performance
                avg_query_time = self._calculate_avg_query_time()
                queries_per_second = self._calculate_queries_per_second()
                query_success_rate = self._calculate_query_success_rate()
                
                # Memory usage
                total_memory_mb = sum(self._memory_usage.values())
                cache_memory_mb = self._memory_usage.get('cache', 0.0)
                index_memory_mb = self._memory_usage.get('indexes', 0.0)
                
                # Cache performance
                cache_hit_rate = self._calculate_cache_hit_rate()
                cache_utilization = self._calculate_cache_utilization()
                
                # Database performance
                db_connections_active = self._metrics.get_metric('db_connections_active')
                db_avg_response_time = self._metrics.get_metric('db_avg_response_time_ms')
                
                # System health
                error_rate = self._calculate_error_rate()
                health_score = self._calculate_health_score(
                    avg_query_time, cache_hit_rate, error_rate, total_memory_mb
                )
                
                # Create snapshot
                snapshot_data = {
                    'avg_query_time_ms': avg_query_time,
                    'queries_per_second': queries_per_second,
                    'query_success_rate': query_success_rate,
                    'total_memory_mb': total_memory_mb,
                    'cache_memory_mb': cache_memory_mb,
                    'index_memory_mb': index_memory_mb,
                    'cache_hit_rate': cache_hit_rate,
                    'cache_utilization': cache_utilization,
                    'db_connections_active': int(db_connections_active),
                    'db_avg_response_time_ms': float(db_avg_response_time),
                    'error_rate': error_rate,
                    'health_score': health_score,
                }
                
                # Cache the calculation
                self._last_performance_calculation = current_time
                self._cached_performance_data = snapshot_data
                
                return PerformanceSnapshot(**snapshot_data)
                
        except Exception as e:
            logger.error(f"Failed to get current snapshot: {e}")
            raise PerformanceError(f"Failed to get performance snapshot: {e}")
    
    def get_historical_data(self, hours: int = 1) -> List[PerformanceSnapshot]:
        """
        Get historical performance data.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of PerformanceSnapshot objects
        """
        try:
            with self._lock:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                historical_data = [
                    snapshot for snapshot in self._snapshots
                    if snapshot.timestamp >= cutoff_time
                ]
                
                return list(historical_data)
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise PerformanceError(f"Failed to get historical data: {e}")
    
    def check_thresholds(self) -> Dict[str, Any]:
        """
        Check if any performance thresholds are exceeded.
        
        Returns:
            Dictionary with threshold check results and active alerts
        """
        try:
            current_snapshot = self.get_current_snapshot()
            threshold_violations = []
            
            # Check query performance
            if current_snapshot.avg_query_time_ms > self.thresholds.max_avg_query_time_ms:
                threshold_violations.append({
                    'metric': 'avg_query_time_ms',
                    'current': current_snapshot.avg_query_time_ms,
                    'threshold': self.thresholds.max_avg_query_time_ms,
                    'severity': 'warning'
                })
            
            if current_snapshot.queries_per_second < self.thresholds.min_queries_per_second:
                threshold_violations.append({
                    'metric': 'queries_per_second',
                    'current': current_snapshot.queries_per_second,
                    'threshold': self.thresholds.min_queries_per_second,
                    'severity': 'warning'
                })
            
            # Check memory usage
            memory_utilization = current_snapshot.total_memory_mb / self.thresholds.max_cache_memory_mb
            if memory_utilization > self.thresholds.max_memory_utilization:
                threshold_violations.append({
                    'metric': 'memory_utilization',
                    'current': memory_utilization,
                    'threshold': self.thresholds.max_memory_utilization,
                    'severity': 'error' if memory_utilization > 0.95 else 'warning'
                })
            
            # Check cache performance
            if current_snapshot.cache_hit_rate < self.thresholds.min_cache_hit_rate:
                threshold_violations.append({
                    'metric': 'cache_hit_rate',
                    'current': current_snapshot.cache_hit_rate,
                    'threshold': self.thresholds.min_cache_hit_rate,
                    'severity': 'warning'
                })
            
            # Check error rate
            if current_snapshot.error_rate > self.thresholds.max_error_rate:
                threshold_violations.append({
                    'metric': 'error_rate',
                    'current': current_snapshot.error_rate,
                    'threshold': self.thresholds.max_error_rate,
                    'severity': 'error'
                })
            
            # Check health score
            if current_snapshot.health_score < self.thresholds.min_health_score:
                threshold_violations.append({
                    'metric': 'health_score',
                    'current': current_snapshot.health_score,
                    'threshold': self.thresholds.min_health_score,
                    'severity': 'error' if current_snapshot.health_score < 0.5 else 'warning'
                })
            
            return {
                'threshold_violations': threshold_violations,
                'active_alerts': [alert.to_dict() for alert in self._active_alerts.values()],
                'total_violations': len(threshold_violations),
                'healthy': len(threshold_violations) == 0,
                'overall_health_score': current_snapshot.health_score,
            }
            
        except Exception as e:
            logger.error(f"Failed to check thresholds: {e}")
            raise PerformanceError(f"Failed to check thresholds: {e}")
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            historical_data = self.get_historical_data(hours)
            
            if len(historical_data) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Extract time series data
            timestamps = [snapshot.timestamp for snapshot in historical_data]
            query_times = [snapshot.avg_query_time_ms for snapshot in historical_data]
            cache_hit_rates = [snapshot.cache_hit_rate for snapshot in historical_data]
            memory_usage = [snapshot.total_memory_mb for snapshot in historical_data]
            health_scores = [snapshot.health_score for snapshot in historical_data]
            
            # Calculate trends
            def calculate_trend(values):
                if len(values) < 2:
                    return 0.0
                return (values[-1] - values[0]) / len(values)
            
            trends = {
                'query_time_trend': calculate_trend(query_times),
                'cache_hit_rate_trend': calculate_trend(cache_hit_rates),
                'memory_usage_trend': calculate_trend(memory_usage),
                'health_score_trend': calculate_trend(health_scores),
            }
            
            # Calculate statistics
            statistics_data = {
                'query_time_stats': {
                    'min': min(query_times),
                    'max': max(query_times),
                    'avg': statistics.mean(query_times),
                    'median': statistics.median(query_times),
                    'std_dev': statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
                },
                'cache_hit_rate_stats': {
                    'min': min(cache_hit_rates),
                    'max': max(cache_hit_rates),
                    'avg': statistics.mean(cache_hit_rates),
                },
                'memory_usage_stats': {
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'avg': statistics.mean(memory_usage),
                },
            }
            
            return {
                'data_points': len(historical_data),
                'time_range_hours': hours,
                'trends': trends,
                'statistics': statistics_data,
                'first_timestamp': timestamps[0].isoformat(),
                'last_timestamp': timestamps[-1].isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            raise PerformanceError(f"Failed to analyze trends: {e}")
    
    def _calculate_avg_query_time(self) -> float:
        """Calculate average query time from recent queries"""
        if not self._query_times_window:
            return 0.0
        return statistics.mean(self._query_times_window)
    
    def _calculate_queries_per_second(self) -> float:
        """Calculate queries per second from recent activity"""
        if not self._recent_queries:
            return 0.0
        
        # Count queries in last 60 seconds
        current_time = time.time() * 1000  # Convert to ms
        one_minute_ago = current_time - 60000
        
        recent_count = sum(
            1 for query_time, _ in self._recent_queries
            if query_time > one_minute_ago
        )
        
        return recent_count / 60.0
    
    def _calculate_query_success_rate(self) -> float:
        """Calculate query success rate from recent queries"""
        if not self._recent_queries:
            return 1.0
        
        successful = sum(1 for _, success in self._recent_queries if success)
        return successful / len(self._recent_queries)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent operations"""
        if not self._cache_operations:
            return 0.0
        
        hits = sum(1 for _, hit in self._cache_operations if hit)
        return hits / len(self._cache_operations)
    
    def _calculate_cache_utilization(self) -> float:
        """Calculate cache utilization (simplified)"""
        # This would need integration with actual cache to get real utilization
        return self._metrics.get_metric('cache_utilization') / 100.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        total_queries = self._metrics.get_metric('total_queries')
        failed_queries = self._metrics.get_metric('failed_queries')
        
        if total_queries == 0:
            return 0.0
        
        return failed_queries / total_queries
    
    def _calculate_health_score(self, avg_query_time: float, cache_hit_rate: float,
                               error_rate: float, memory_mb: float) -> float:
        """Calculate overall system health score (0.0-1.0)"""
        # Weighted health calculation
        query_health = max(0.0, 1.0 - (avg_query_time / self.thresholds.max_avg_query_time_ms))
        cache_health = cache_hit_rate
        error_health = max(0.0, 1.0 - (error_rate / self.thresholds.max_error_rate))
        memory_health = max(0.0, 1.0 - (memory_mb / self.thresholds.max_cache_memory_mb))
        
        # Weighted average
        health_score = (
            query_health * 0.3 +
            cache_health * 0.25 +
            error_health * 0.25 +
            memory_health * 0.2
        )
        
        return min(1.0, max(0.0, health_score))
    
    def _trigger_alert(self, alert_type: str, severity: str, message: str,
                      metric_name: str, current_value: float, threshold_value: float) -> None:
        """Trigger performance alert"""
        alert_key = f"{alert_type}_{metric_name}"
        
        # Avoid duplicate alerts
        if alert_key in self._active_alerts:
            return
        
        alert = PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)
        
        # Call alert callback if configured
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Performance alert: {message}")
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread"""
        def monitor_worker():
            while not self._stop_monitoring.wait(self.snapshot_interval_seconds):
                try:
                    # Create snapshot
                    snapshot = self.get_current_snapshot()
                    
                    with self._lock:
                        self._snapshots.append(snapshot)
                        
                        # Trim old snapshots
                        while len(self._snapshots) > self._max_snapshots:
                            self._snapshots.popleft()
                    
                    # Check thresholds and generate alerts
                    self.check_thresholds()
                    
                    # Clear resolved alerts
                    self._clear_resolved_alerts()
                    
                except Exception as e:
                    logger.error(f"Background monitoring failed: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_worker, daemon=True)
        self._monitoring_thread.start()
        logger.debug("Background monitoring thread started")
    
    def _clear_resolved_alerts(self) -> None:
        """Clear alerts that are no longer active"""
        # This is a simplified implementation
        # In production, would check if conditions are still met
        current_time = datetime.now()
        
        resolved_alerts = []
        for alert_key, alert in self._active_alerts.items():
            # Auto-resolve alerts older than 5 minutes if not acknowledged
            if (current_time - alert.timestamp).total_seconds() > 300:
                resolved_alerts.append(alert_key)
        
        for alert_key in resolved_alerts:
            del self._active_alerts[alert_key]
    
    def shutdown(self) -> None:
        """Shutdown monitoring and cleanup resources"""
        try:
            self._stop_monitoring.set()
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            logger.info("Performance monitor shutdown completed")
            
        except Exception as e:
            logger.error(f"Performance monitor shutdown failed: {e}")


def create_performance_monitor(
    thresholds: Optional[PerformanceThresholds] = None,
    **config
) -> PerformanceMonitor:
    """
    Factory function to create performance monitor.
    
    Args:
        thresholds: Performance threshold configuration
        **config: Additional configuration parameters
        
    Returns:
        PerformanceMonitor instance
    """
    return RealTimePerformanceMonitor(
        thresholds=thresholds,
        history_retention_hours=config.get('history_retention_hours', 24),
        snapshot_interval_seconds=config.get('snapshot_interval_seconds', 60),
        alert_callback=config.get('alert_callback')
    )