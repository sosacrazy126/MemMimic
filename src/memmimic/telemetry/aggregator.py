"""
Real-Time Metrics Aggregator for MemMimic v2.0

High-performance aggregation and analysis engine for telemetry data.
Provides real-time statistics, trending, and intelligent insights.
"""

import time
import threading
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json

from ..errors import get_error_logger
from .collector import TelemetryCollector, get_telemetry_collector

logger = get_error_logger("telemetry.aggregator")


@dataclass
class TimeSeriesPoint:
    """Time series data point for trending analysis"""
    timestamp: float
    value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AggregationWindow:
    """Aggregation window with statistics"""
    start_time: float
    end_time: float
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    success_rate: float
    error_count: int


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    operation: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0.0 to 1.0
    rate_of_change: float  # per minute
    confidence: float  # 0.0 to 1.0
    analysis_period_minutes: int
    recommendation: Optional[str] = None


class MetricsAggregator:
    """
    Real-Time Metrics Aggregator with intelligent analysis.
    
    Features:
    - Real-time statistical aggregation
    - Time series analysis and trending
    - Performance threshold monitoring
    - Intelligent anomaly detection
    - Multi-dimensional analysis
    - Memory-efficient sliding windows
    """
    
    def __init__(self, collector: Optional[TelemetryCollector] = None, config: Optional[Dict[str, Any]] = None):
        self.collector = collector or get_telemetry_collector()
        self.config = config or {}
        
        # Configuration
        self.aggregation_intervals = self.config.get('aggregation_intervals', [60, 300, 900, 3600])  # 1m, 5m, 15m, 1h
        self.max_time_series_points = self.config.get('max_time_series_points', 1000)
        self.trend_analysis_window = self.config.get('trend_analysis_window_minutes', 15)
        self.anomaly_detection_enabled = self.config.get('anomaly_detection_enabled', True)
        
        # Time series storage - sliding windows for each interval
        self._time_series: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.max_time_series_points)))
        
        # Aggregation windows - most recent windows for each interval
        self._aggregation_windows: Dict[str, Dict[int, List[AggregationWindow]]] = defaultdict(lambda: defaultdict(list))
        
        # Performance thresholds
        self._performance_thresholds = {
            'summary_retrieval': {'p95_ms': 5.0, 'p99_ms': 10.0},
            'full_context_retrieval': {'p95_ms': 50.0, 'p99_ms': 100.0},
            'enhanced_remember': {'p95_ms': 15.0, 'p99_ms': 30.0},
            'governance_validation': {'p95_ms': 10.0, 'p99_ms': 20.0},
            'telemetry_overhead': {'p95_ms': 1.0, 'p99_ms': 2.0}
        }
        
        # Trend analysis cache
        self._trend_cache: Dict[str, TrendAnalysis] = {}
        self._trend_cache_ttl = 60  # 1 minute TTL
        self._last_trend_update = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._background_thread = None
        self._stop_background = threading.Event()
        self._aggregation_thread = None
        
        self._start_background_aggregation()
        logger.info("MetricsAggregator initialized")
    
    def aggregate_operation_metrics(self, operation: str, window_seconds: int = 60) -> Optional[AggregationWindow]:
        """Aggregate metrics for specific operation over time window"""
        op_stats = self.collector.get_operation_stats(operation)
        if not op_stats:
            return None
        
        current_time = time.perf_counter()
        window_start = current_time - window_seconds
        
        # Get recent events from collector
        if operation in self.collector._operation_buffers:
            recent_events = self.collector._operation_buffers[operation].get_recent(10000)
            
            # Filter events within time window
            window_events = [
                event for event in recent_events
                if event.timestamp >= window_start
            ]
            
            if not window_events:
                return None
            
            # Calculate aggregation statistics
            durations = [event.duration_ms for event in window_events]
            success_events = [event for event in window_events if event.success]
            error_count = len(window_events) - len(success_events)
            
            durations.sort()
            n = len(durations)
            
            if n > 0:
                aggregation = AggregationWindow(
                    start_time=window_start,
                    end_time=current_time,
                    count=n,
                    sum_value=sum(durations),
                    min_value=min(durations),
                    max_value=max(durations),
                    mean_value=sum(durations) / n,
                    p50_value=durations[int(n * 0.5)],
                    p95_value=durations[int(n * 0.95)],
                    p99_value=durations[int(n * 0.99)],
                    success_rate=len(success_events) / n,
                    error_count=error_count
                )
                
                # Store in aggregation windows
                with self._lock:
                    self._aggregation_windows[operation][window_seconds].append(aggregation)
                    # Keep only recent windows (max 100 windows per operation)
                    if len(self._aggregation_windows[operation][window_seconds]) > 100:
                        self._aggregation_windows[operation][window_seconds] = \
                            self._aggregation_windows[operation][window_seconds][-100:]
                
                return aggregation
        
        return None
    
    def get_time_series_data(self, operation: str, interval_seconds: int = 60, points: int = 100) -> List[TimeSeriesPoint]:
        """Get time series data for operation"""
        with self._lock:
            if operation in self._time_series and interval_seconds in self._time_series[operation]:
                time_series = list(self._time_series[operation][interval_seconds])
                return time_series[-points:]  # Return most recent points
        return []
    
    def analyze_performance_trends(self, operation: str, analysis_window_minutes: int = None) -> TrendAnalysis:
        """Analyze performance trends for operation"""
        if analysis_window_minutes is None:
            analysis_window_minutes = self.trend_analysis_window
        
        current_time = time.time()
        
        # Check cache first
        cache_key = f"{operation}_{analysis_window_minutes}"
        if (cache_key in self._trend_cache and 
            current_time - self._last_trend_update < self._trend_cache_ttl):
            return self._trend_cache[cache_key]
        
        # Get time series data for trend analysis
        time_series = self.get_time_series_data(operation, interval_seconds=60, points=analysis_window_minutes)
        
        if len(time_series) < 5:  # Need minimum data points
            trend = TrendAnalysis(
                operation=operation,
                trend_direction='stable',
                trend_strength=0.0,
                rate_of_change=0.0,
                confidence=0.0,
                analysis_period_minutes=analysis_window_minutes,
                recommendation="Insufficient data for trend analysis"
            )
        else:
            trend = self._calculate_trend_analysis(operation, time_series, analysis_window_minutes)
        
        # Cache result
        with self._lock:
            self._trend_cache[cache_key] = trend
            self._last_trend_update = current_time
        
        return trend
    
    def _calculate_trend_analysis(self, operation: str, time_series: List[TimeSeriesPoint], window_minutes: int) -> TrendAnalysis:
        """Calculate detailed trend analysis"""
        if len(time_series) < 2:
            return TrendAnalysis(
                operation=operation,
                trend_direction='stable',
                trend_strength=0.0,
                rate_of_change=0.0,
                confidence=0.0,
                analysis_period_minutes=window_minutes
            )
        
        # Extract values and calculate linear regression
        values = [point.value for point in time_series]
        n = len(values)
        
        # Simple linear regression to detect trend
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate trend metrics
        rate_of_change = slope * 60  # Convert to per-minute rate
        
        # Determine trend direction and strength
        abs_slope = abs(slope)
        if abs_slope < 0.01:  # Very small change
            trend_direction = 'stable'
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = 'degrading'  # Increasing response time is bad
            trend_strength = min(abs_slope * 100, 1.0)  # Scale to 0-1
        else:
            trend_direction = 'improving'  # Decreasing response time is good
            trend_strength = min(abs_slope * 100, 1.0)
        
        # Calculate confidence based on data consistency
        if n >= 10:
            # Calculate R-squared for confidence
            y_pred = [x_mean + slope * (x - x_mean) for x in x_values]
            ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence = max(0.0, r_squared)
        else:
            confidence = max(0.0, (n - 2) / 8.0)  # Increase confidence with more data points
        
        # Generate recommendation
        recommendation = self._generate_trend_recommendation(operation, trend_direction, trend_strength, rate_of_change)
        
        return TrendAnalysis(
            operation=operation,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            rate_of_change=rate_of_change,
            confidence=confidence,
            analysis_period_minutes=window_minutes,
            recommendation=recommendation
        )
    
    def _generate_trend_recommendation(self, operation: str, direction: str, strength: float, rate: float) -> str:
        """Generate intelligent recommendations based on trend analysis"""
        if direction == 'stable':
            return f"Performance for {operation} is stable - continue monitoring"
        elif direction == 'improving':
            if strength > 0.5:
                return f"Performance for {operation} is significantly improving at {rate:.2f}ms/min - monitor for sustained improvement"
            else:
                return f"Performance for {operation} is slightly improving - continue current optimizations"
        else:  # degrading
            if strength > 0.7:
                return f"ALERT: Performance for {operation} is degrading rapidly at {rate:.2f}ms/min - immediate investigation required"
            elif strength > 0.3:
                return f"WARNING: Performance for {operation} is degrading at {rate:.2f}ms/min - consider optimization"
            else:
                return f"Performance for {operation} is slightly degrading - monitor closely"
    
    def detect_performance_anomalies(self, operation: str, window_minutes: int = 15) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis"""
        if not self.anomaly_detection_enabled:
            return []
        
        anomalies = []
        
        # Get recent performance data
        current_stats = self.collector.get_operation_stats(operation)
        if not current_stats:
            return anomalies
        
        # Get baseline performance (last hour)
        baseline_windows = []
        with self._lock:
            if operation in self._aggregation_windows:
                for window_size in [60, 300]:  # 1min and 5min windows
                    if window_size in self._aggregation_windows[operation]:
                        baseline_windows.extend(self._aggregation_windows[operation][window_size][-12:])  # Last 12 windows
        
        if len(baseline_windows) < 5:  # Need minimum baseline data
            return anomalies
        
        # Calculate baseline statistics
        baseline_p95s = [w.p95_value for w in baseline_windows]
        baseline_mean_p95 = statistics.mean(baseline_p95s)
        baseline_stdev_p95 = statistics.stdev(baseline_p95s) if len(baseline_p95s) > 1 else 0
        
        baseline_success_rates = [w.success_rate for w in baseline_windows]
        baseline_mean_success = statistics.mean(baseline_success_rates)
        
        # Check for anomalies
        current_p95 = current_stats.get('p95_duration_ms', 0)
        current_success_rate = current_stats.get('success_rate', 1.0)
        
        # P95 latency anomaly detection (using 2-sigma rule)
        if baseline_stdev_p95 > 0:
            threshold = baseline_mean_p95 + 2 * baseline_stdev_p95
            if current_p95 > threshold:
                anomalies.append({
                    'type': 'latency_spike',
                    'severity': 'high' if current_p95 > threshold * 1.5 else 'medium',
                    'message': f"P95 latency spike detected: {current_p95:.2f}ms vs baseline {baseline_mean_p95:.2f}ms",
                    'current_value': current_p95,
                    'baseline_value': baseline_mean_p95,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Success rate anomaly detection
        success_rate_threshold = baseline_mean_success - 0.1  # 10% drop
        if current_success_rate < success_rate_threshold:
            anomalies.append({
                'type': 'success_rate_drop',
                'severity': 'critical' if current_success_rate < baseline_mean_success - 0.2 else 'high',
                'message': f"Success rate drop detected: {current_success_rate*100:.1f}% vs baseline {baseline_mean_success*100:.1f}%",
                'current_value': current_success_rate,
                'baseline_value': baseline_mean_success,
                'threshold': success_rate_threshold,
                'timestamp': datetime.now().isoformat()
            })
        
        return anomalies
    
    def get_performance_health_score(self) -> Dict[str, Any]:
        """Calculate overall performance health score"""
        scores = {}
        overall_score = 100.0
        issues = []
        
        # Check each operation against thresholds
        for operation, thresholds in self._performance_thresholds.items():
            op_stats = self.collector.get_operation_stats(operation)
            if not op_stats:
                continue
            
            operation_score = 100.0
            
            # Check P95 threshold
            if 'p95_ms' in thresholds:
                current_p95 = op_stats.get('p95_duration_ms', 0)
                threshold_p95 = thresholds['p95_ms']
                
                if current_p95 > threshold_p95:
                    # Penalty based on how far over threshold
                    penalty = min(50, (current_p95 / threshold_p95 - 1) * 100)
                    operation_score -= penalty
                    issues.append(f"{operation} P95 latency {current_p95:.2f}ms exceeds {threshold_p95}ms threshold")
            
            # Check P99 threshold
            if 'p99_ms' in thresholds:
                current_p99 = op_stats.get('p99_duration_ms', 0)
                threshold_p99 = thresholds['p99_ms']
                
                if current_p99 > threshold_p99:
                    penalty = min(30, (current_p99 / threshold_p99 - 1) * 50)
                    operation_score -= penalty
                    issues.append(f"{operation} P99 latency {current_p99:.2f}ms exceeds {threshold_p99}ms threshold")
            
            # Check success rate
            success_rate = op_stats.get('success_rate', 1.0)
            if success_rate < 0.95:  # 95% success rate threshold
                penalty = (0.95 - success_rate) * 200  # Heavy penalty for low success rate
                operation_score -= penalty
                issues.append(f"{operation} success rate {success_rate*100:.1f}% below 95% threshold")
            
            scores[operation] = max(0, operation_score)
        
        # Calculate overall score as weighted average
        if scores:
            overall_score = sum(scores.values()) / len(scores)
        
        # Apply global penalties
        system_stats = self.collector.get_system_stats()
        if system_stats.get('total_errors', 0) > system_stats.get('total_events', 1) * 0.05:  # >5% error rate
            overall_score -= 20
            issues.append("System error rate above 5%")
        
        health_status = 'healthy'
        if overall_score < 60:
            health_status = 'critical'
        elif overall_score < 80:
            health_status = 'degraded'
        elif overall_score < 95:
            health_status = 'warning'
        
        return {
            'overall_score': max(0, overall_score),
            'health_status': health_status,
            'operation_scores': scores,
            'issues': issues,
            'thresholds_met': {
                op: score >= 95 for op, score in scores.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get comprehensive aggregation summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'aggregation_intervals': self.aggregation_intervals,
                'trend_analysis_window': self.trend_analysis_window,
                'anomaly_detection_enabled': self.anomaly_detection_enabled
            },
            'operations': {},
            'health_score': self.get_performance_health_score(),
            'system_trends': {}
        }
        
        # Aggregate data for each operation
        operations = list(self.collector._operation_buffers.keys())
        for operation in operations:
            op_data = {
                'recent_stats': self.collector.get_operation_stats(operation),
                'trend_analysis': self.analyze_performance_trends(operation),
                'anomalies': self.detect_performance_anomalies(operation),
                'aggregation_windows': {}
            }
            
            # Get recent aggregation windows
            for interval in [60, 300, 900]:  # 1m, 5m, 15m
                recent_agg = self.aggregate_operation_metrics(operation, interval)
                if recent_agg:
                    op_data['aggregation_windows'][f"{interval}s"] = {
                        'count': recent_agg.count,
                        'mean_ms': recent_agg.mean_value,
                        'p95_ms': recent_agg.p95_value,
                        'p99_ms': recent_agg.p99_value,
                        'success_rate': recent_agg.success_rate,
                        'error_count': recent_agg.error_count
                    }
            
            summary['operations'][operation] = op_data
        
        return summary
    
    def _start_background_aggregation(self):
        """Start background aggregation processing"""
        def aggregation_worker():
            while not self._stop_background.wait(30):  # Aggregate every 30 seconds
                try:
                    current_time = time.perf_counter()
                    
                    # Aggregate metrics for all operations
                    operations = list(self.collector._operation_buffers.keys())
                    for operation in operations:
                        for interval in self.aggregation_intervals:
                            # Create time series points
                            agg_window = self.aggregate_operation_metrics(operation, interval)
                            if agg_window:
                                time_point = TimeSeriesPoint(
                                    timestamp=current_time,
                                    value=agg_window.p95_value,  # Use P95 as primary metric
                                    metadata={
                                        'mean': agg_window.mean_value,
                                        'count': agg_window.count,
                                        'success_rate': agg_window.success_rate
                                    }
                                )
                                
                                with self._lock:
                                    self._time_series[operation][interval].append(time_point)
                
                except Exception as e:
                    logger.error(f"Background aggregation failed: {e}")
        
        self._aggregation_thread = threading.Thread(target=aggregation_worker, daemon=True)
        self._aggregation_thread.start()
        logger.info("Background aggregation started")
    
    def shutdown(self):
        """Shutdown metrics aggregator"""
        try:
            self._stop_background.set()
            if self._aggregation_thread and self._aggregation_thread.is_alive():
                self._aggregation_thread.join(timeout=5.0)
            
            logger.info("MetricsAggregator shutdown completed")
            
        except Exception as e:
            logger.error(f"MetricsAggregator shutdown failed: {e}")


# Convenience function for getting global aggregator
_global_aggregator: Optional[MetricsAggregator] = None
_aggregator_lock = threading.Lock()


def get_metrics_aggregator() -> MetricsAggregator:
    """Get global metrics aggregator instance"""
    global _global_aggregator
    
    if _global_aggregator is None:
        with _aggregator_lock:
            if _global_aggregator is None:
                _global_aggregator = MetricsAggregator()
    
    return _global_aggregator