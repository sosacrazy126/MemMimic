"""
DSPy Performance Monitor

Real-time performance monitoring and metrics collection for DSPy consciousness
optimization with alerting and reporting capabilities.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics
import json
from pathlib import Path

from .config import DSPyConfig
from ..errors import MemMimicError, get_error_logger

logger = get_error_logger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""
    timestamp: float
    metric_type: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are exceeded"""
    timestamp: float
    alert_type: str
    message: str
    severity: str  # "warning", "error", "critical"
    metric_value: float
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DSPyPerformanceMonitor:
    """
    Real-time performance monitoring for DSPy consciousness optimization.
    
    Tracks response times, success rates, resource usage, and generates
    alerts when performance thresholds are exceeded.
    """
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.is_monitoring = False
        
        # Performance metrics storage
        self.metrics: Dict[str, deque] = {
            "response_times": deque(maxlen=1000),
            "confidence_scores": deque(maxlen=1000),
            "success_rates": deque(maxlen=100),
            "error_rates": deque(maxlen=100),
            "token_usage": deque(maxlen=1000),
            "circuit_breaker_events": deque(maxlen=100)
        }
        
        # Real-time tracking
        self.current_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "average_response_time": 0.0,
            "average_confidence": 0.0,
            "current_error_rate": 0.0,
            "last_update": time.time()
        }
        
        # Alerting
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance windows
        self.window_sizes = {
            "response_time": 100,    # Last 100 requests
            "success_rate": 50,      # Last 50 requests
            "error_rate": 50,        # Last 50 requests
            "confidence": 100        # Last 100 requests
        }
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already active")
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("DSPy performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        logger.info("DSPy performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.monitoring.metrics_collection_interval)
                self._update_derived_metrics()
                self._check_alert_conditions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def record_request(
        self,
        response_time_ms: float,
        success: bool,
        confidence_score: float,
        tokens_used: int = 0,
        operation_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a request performance metric"""
        timestamp = time.time()
        
        # Update counters
        self.current_metrics["total_requests"] += 1
        if success:
            self.current_metrics["successful_requests"] += 1
        else:
            self.current_metrics["failed_requests"] += 1
        
        self.current_metrics["total_tokens_used"] += tokens_used
        self.current_metrics["last_update"] = timestamp
        
        # Store individual metrics
        self.metrics["response_times"].append(PerformanceMetric(
            timestamp=timestamp,
            metric_type="response_time",
            value=response_time_ms,
            metadata={"operation_type": operation_type, "success": success}
        ))
        
        self.metrics["confidence_scores"].append(PerformanceMetric(
            timestamp=timestamp,
            metric_type="confidence",
            value=confidence_score,
            metadata={"operation_type": operation_type}
        ))
        
        if tokens_used > 0:
            self.metrics["token_usage"].append(PerformanceMetric(
                timestamp=timestamp,
                metric_type="token_usage",
                value=tokens_used,
                metadata={"operation_type": operation_type}
            ))
        
        # Check immediate alert conditions
        self._check_immediate_alerts(response_time_ms, success, confidence_score, operation_type)
    
    def record_circuit_breaker_event(
        self,
        event_type: str,
        circuit_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record circuit breaker event"""
        self.metrics["circuit_breaker_events"].append(PerformanceMetric(
            timestamp=time.time(),
            metric_type="circuit_breaker",
            value=1.0 if event_type == "open" else 0.0,
            metadata={"event_type": event_type, "circuit_name": circuit_name, **(metadata or {})}
        ))
    
    def _update_derived_metrics(self) -> None:
        """Update derived metrics from raw data"""
        # Calculate average response time
        if self.metrics["response_times"]:
            recent_times = list(self.metrics["response_times"])[-self.window_sizes["response_time"]:]
            self.current_metrics["average_response_time"] = statistics.mean(
                metric.value for metric in recent_times
            )
        
        # Calculate average confidence
        if self.metrics["confidence_scores"]:
            recent_confidence = list(self.metrics["confidence_scores"])[-self.window_sizes["confidence"]:]
            self.current_metrics["average_confidence"] = statistics.mean(
                metric.value for metric in recent_confidence
            )
        
        # Calculate error rate
        total_requests = self.current_metrics["total_requests"]
        if total_requests > 0:
            self.current_metrics["current_error_rate"] = (
                self.current_metrics["failed_requests"] / total_requests
            )
    
    def _check_immediate_alerts(
        self,
        response_time_ms: float,
        success: bool,
        confidence_score: float,
        operation_type: str
    ) -> None:
        """Check for immediate alert conditions"""
        # Response time alerts
        if operation_type in ["biological_reflex", "immediate"] and response_time_ms > 5.0:
            self._create_alert(
                "biological_reflex_slow",
                f"Biological reflex exceeded 5ms: {response_time_ms:.2f}ms",
                "critical",
                response_time_ms,
                5.0,
                {"operation_type": operation_type}
            )
        elif response_time_ms > self.config.monitoring.response_time_alert_threshold:
            self._create_alert(
                "response_time_high",
                f"Response time exceeded threshold: {response_time_ms:.2f}ms",
                "warning",
                response_time_ms,
                self.config.monitoring.response_time_alert_threshold,
                {"operation_type": operation_type}
            )
        
        # Confidence alerts
        if confidence_score < 0.5:
            self._create_alert(
                "confidence_low",
                f"Low confidence score: {confidence_score:.3f}",
                "warning",
                confidence_score,
                0.5,
                {"operation_type": operation_type}
            )
    
    def _check_alert_conditions(self) -> None:
        """Check for alert conditions based on trends"""
        # Error rate alert
        if self.current_metrics["current_error_rate"] > self.config.monitoring.error_rate_alert_threshold:
            self._create_alert(
                "error_rate_high",
                f"Error rate exceeded threshold: {self.current_metrics['current_error_rate']:.2%}",
                "error",
                self.current_metrics["current_error_rate"],
                self.config.monitoring.error_rate_alert_threshold
            )
        
        # Token usage alert
        max_tokens = self.config.performance.max_token_budget_per_hour
        if max_tokens > 0:
            # Calculate tokens used in last hour
            one_hour_ago = time.time() - 3600
            recent_token_usage = sum(
                metric.value for metric in self.metrics["token_usage"]
                if metric.timestamp > one_hour_ago
            )
            
            usage_rate = recent_token_usage / max_tokens
            if usage_rate > self.config.monitoring.token_usage_alert_threshold:
                self._create_alert(
                    "token_usage_high",
                    f"Token usage at {usage_rate:.1%} of hourly budget",
                    "warning",
                    recent_token_usage,
                    max_tokens * self.config.monitoring.token_usage_alert_threshold
                )
    
    def _create_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        metric_value: float,
        threshold_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create and process performance alert"""
        alert = PerformanceAlert(
            timestamp=time.time(),
            alert_type=alert_type,
            message=message,
            severity=severity,
            metric_value=metric_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Keep alert history manageable
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]  # Keep last 500 alerts
        
        # Log alert
        log_level = {
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical
        }.get(severity, logger.info)
        
        log_level(f"DSPy Performance Alert [{severity.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.current_metrics,
            "monitoring_active": self.is_monitoring,
            "metrics_collected": {
                name: len(metrics) for name, metrics in self.metrics.items()
            }
        }
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        
        # Filter metrics to time window
        recent_response_times = [
            metric.value for metric in self.metrics["response_times"]
            if metric.timestamp > cutoff_time
        ]
        
        recent_confidence = [
            metric.value for metric in self.metrics["confidence_scores"]
            if metric.timestamp > cutoff_time
        ]
        
        recent_tokens = [
            metric.value for metric in self.metrics["token_usage"]
            if metric.timestamp > cutoff_time
        ]
        
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
        
        # Calculate statistics
        summary = {
            "time_window_minutes": window_minutes,
            "total_requests": len(recent_response_times),
            "response_times": {},
            "confidence_scores": {},
            "token_usage": {},
            "alerts": {
                "total": len(recent_alerts),
                "by_severity": {},
                "by_type": {}
            }
        }
        
        # Response time statistics
        if recent_response_times:
            summary["response_times"] = {
                "average": statistics.mean(recent_response_times),
                "median": statistics.median(recent_response_times),
                "min": min(recent_response_times),
                "max": max(recent_response_times),
                "p95": statistics.quantiles(recent_response_times, n=20)[18] if len(recent_response_times) > 20 else max(recent_response_times),
                "p99": statistics.quantiles(recent_response_times, n=100)[98] if len(recent_response_times) > 100 else max(recent_response_times)
            }
        
        # Confidence statistics
        if recent_confidence:
            summary["confidence_scores"] = {
                "average": statistics.mean(recent_confidence),
                "median": statistics.median(recent_confidence),
                "min": min(recent_confidence),
                "max": max(recent_confidence)
            }
        
        # Token usage
        if recent_tokens:
            summary["token_usage"] = {
                "total": sum(recent_tokens),
                "average_per_request": statistics.mean(recent_tokens),
                "max_per_request": max(recent_tokens)
            }
        
        # Alert statistics
        for alert in recent_alerts:
            summary["alerts"]["by_severity"][alert.severity] = summary["alerts"]["by_severity"].get(alert.severity, 0) + 1
            summary["alerts"]["by_type"][alert.alert_type] = summary["alerts"]["by_type"].get(alert.alert_type, 0) + 1
        
        return summary
    
    def export_metrics(self, output_path: Path, window_minutes: Optional[int] = None) -> None:
        """Export metrics to file"""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "current_metrics": self.get_current_metrics(),
                "performance_summary": self.get_performance_summary(window_minutes or 60),
                "raw_metrics": {}
            }
            
            # Export raw metrics if requested
            if window_minutes is None:
                cutoff_time = 0
            else:
                cutoff_time = time.time() - (window_minutes * 60)
            
            for metric_type, metrics in self.metrics.items():
                export_data["raw_metrics"][metric_type] = [
                    {
                        "timestamp": metric.timestamp,
                        "value": metric.value,
                        "metadata": metric.metadata
                    }
                    for metric in metrics
                    if metric.timestamp > cutoff_time
                ]
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        for metrics in self.metrics.values():
            metrics.clear()
        
        self.current_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "average_response_time": 0.0,
            "average_confidence": 0.0,
            "current_error_rate": 0.0,
            "last_update": time.time()
        }
        
        self.alerts.clear()
        
        logger.info("Performance metrics reset")

# Global performance monitor instance
performance_monitor: Optional[DSPyPerformanceMonitor] = None

def get_performance_monitor() -> Optional[DSPyPerformanceMonitor]:
    """Get global performance monitor instance"""
    return performance_monitor

def initialize_performance_monitor(config: DSPyConfig) -> DSPyPerformanceMonitor:
    """Initialize global performance monitor"""
    global performance_monitor
    performance_monitor = DSPyPerformanceMonitor(config)
    return performance_monitor