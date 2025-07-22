"""
Ultra-Fast Telemetry Collector for MemMimic v2.0

High-performance metrics collection engine with <1ms overhead target.
Designed for minimal performance impact while capturing comprehensive telemetry.
"""

import time
import threading
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import sys
import traceback

from ..errors import get_error_logger

# Performance optimization imports
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

logger = get_error_logger("telemetry")


@dataclass
class TelemetryEvent:
    """Ultra-lightweight telemetry event structure"""
    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.perf_counter)
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    
    def __post_init__(self):
        # Optimize memory usage
        if not self.metadata:
            self.metadata = {}


@dataclass
class PerformanceSnapshot:
    """Performance snapshot for aggregation"""
    timestamp: float
    operation: str
    count: int
    mean_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    error_count: int
    success_rate: float


class RingBuffer:
    """Ultra-fast ring buffer for telemetry events"""
    
    def __init__(self, size: int = 10000):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.count = 0
        self._lock = threading.RLock()
    
    def append(self, item: TelemetryEvent):
        """Add item to ring buffer with minimal overhead"""
        with self._lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.size
            if self.count < self.size:
                self.count += 1
    
    def get_recent(self, n: int = None) -> List[TelemetryEvent]:
        """Get most recent n items"""
        with self._lock:
            if n is None or n > self.count:
                n = self.count
            
            items = []
            for i in range(n):
                idx = (self.head - 1 - i) % self.size
                if self.buffer[idx] is not None:
                    items.append(self.buffer[idx])
            
            return items
    
    def clear(self):
        """Clear buffer contents"""
        with self._lock:
            self.buffer = [None] * self.size
            self.head = 0
            self.count = 0


class TelemetryCollector:
    """
    Ultra-Fast Telemetry Collector with <1ms overhead target.
    
    Features:
    - Sub-millisecond event recording
    - Lock-free operations where possible  
    - Memory-efficient ring buffers
    - Minimal allocation overhead
    - Background aggregation
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Performance configuration
        self.buffer_size = self.config.get('buffer_size', 10000)
        self.aggregation_interval = self.config.get('aggregation_interval', 1.0)  # 1 second
        self.max_operation_types = self.config.get('max_operation_types', 100)
        
        # Ultra-fast storage using ring buffers
        self._operation_buffers: Dict[str, RingBuffer] = {}
        self._operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'error_count': 0,
            'last_update': time.perf_counter()
        })
        
        # Lock-free counters for critical metrics
        self._total_events = 0
        self._total_errors = 0
        self._overhead_samples = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # Background processing
        self._background_thread = None
        self._stop_background = threading.Event()
        
        # Performance tracking
        self._self_timing = deque(maxlen=1000)  # Track our own overhead
        
        if self.enabled:
            self._start_background_processing()
        
        logger.info(f"TelemetryCollector initialized - enabled: {self.enabled}")
    
    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """
        Record operation with minimal overhead (<1ms target).
        
        Optimizations:
        - Early return if disabled
        - Minimal object creation
        - Lock-free updates where possible
        - Deferred expensive operations
        """
        if not self.enabled:
            return
        
        # Start overhead timing
        start_time = time.perf_counter()
        
        try:
            # Create minimal event object
            event = TelemetryEvent(
                operation=operation,
                duration_ms=duration_ms,
                metadata=metadata,
                success=success
            )
            
            # Get or create ring buffer for operation (minimal locking)
            if operation not in self._operation_buffers:
                with self._lock:
                    if operation not in self._operation_buffers:
                        # Limit number of operation types to prevent memory exhaustion
                        if len(self._operation_buffers) >= self.max_operation_types:
                            logger.warning(f"Max operation types reached: {self.max_operation_types}")
                            return
                        self._operation_buffers[operation] = RingBuffer(self.buffer_size)
            
            # Ultra-fast ring buffer append
            self._operation_buffers[operation].append(event)
            
            # Lock-free counter updates
            self._total_events += 1
            if not success:
                self._total_errors += 1
            
            # Update operation stats (minimal locking)
            with self._stats_lock:
                stats = self._operation_stats[operation]
                stats['count'] += 1
                stats['total_duration'] += duration_ms
                if not success:
                    stats['error_count'] += 1
                stats['last_update'] = time.perf_counter()
        
        except Exception as e:
            # Avoid recursion in error handling
            logger.error(f"TelemetryCollector.record_operation failed: {e}")
        
        finally:
            # Track our own overhead
            overhead_ms = (time.perf_counter() - start_time) * 1000
            self._overhead_samples.append(overhead_ms)
    
    def record_governance_metrics(
        self,
        operation: str,
        governance_time_ms: float,
        violations: int = 0,
        warnings: int = 0,
        status: str = "approved"
    ) -> None:
        """Record governance-specific telemetry with minimal overhead"""
        if not self.enabled:
            return
        
        metadata = {
            'type': 'governance',
            'violations': violations,
            'warnings': warnings,
            'status': status
        }
        
        self.record_operation(
            operation=f"governance_{operation}",
            duration_ms=governance_time_ms,
            metadata=metadata,
            success=(violations == 0)
        )
    
    def record_storage_metrics(
        self,
        operation: str,
        duration_ms: float,
        context_size: int = 0,
        cache_hit: bool = False,
        success: bool = True
    ) -> None:
        """Record storage-specific telemetry"""
        if not self.enabled:
            return
        
        metadata = {
            'type': 'storage',
            'context_size': context_size,
            'cache_hit': cache_hit
        }
        
        self.record_operation(
            operation=f"storage_{operation}",
            duration_ms=duration_ms,
            metadata=metadata,
            success=success
        )
    
    def record_memory_metrics(
        self,
        operation: str,
        duration_ms: float,
        memory_type: str = "enhanced",
        tag_count: int = 0,
        success: bool = True
    ) -> None:
        """Record memory operation telemetry"""
        if not self.enabled:
            return
        
        metadata = {
            'type': 'memory',
            'memory_type': memory_type,
            'tag_count': tag_count
        }
        
        self.record_operation(
            operation=f"memory_{operation}",
            duration_ms=duration_ms,
            metadata=metadata,
            success=success
        )
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get real-time statistics for specific operation"""
        if operation not in self._operation_buffers:
            return None
        
        with self._stats_lock:
            stats = self._operation_stats[operation]
            
            # Get recent events for percentile calculation
            recent_events = self._operation_buffers[operation].get_recent(1000)
            if not recent_events:
                return None
            
            # Calculate statistics
            durations = [event.duration_ms for event in recent_events]
            success_count = sum(1 for event in recent_events if event.success)
            
            if _NUMPY_AVAILABLE:
                # Use numpy for faster percentile calculation
                durations_array = np.array(durations)
                p50 = float(np.percentile(durations_array, 50))
                p95 = float(np.percentile(durations_array, 95))
                p99 = float(np.percentile(durations_array, 99))
                mean_duration = float(np.mean(durations_array))
            else:
                # Fallback to pure Python
                durations.sort()
                n = len(durations)
                p50 = durations[int(n * 0.5)]
                p95 = durations[int(n * 0.95)]
                p99 = durations[int(n * 0.99)]
                mean_duration = sum(durations) / n
            
            return {
                'operation': operation,
                'total_count': stats['count'],
                'recent_count': len(recent_events),
                'error_count': stats['error_count'],
                'success_rate': success_count / len(recent_events),
                'mean_duration_ms': mean_duration,
                'p50_duration_ms': p50,
                'p95_duration_ms': p95,
                'p99_duration_ms': p99,
                'last_update': stats['last_update']
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system telemetry statistics"""
        # Calculate self-overhead statistics
        overhead_samples = list(self._overhead_samples)
        if overhead_samples:
            if _NUMPY_AVAILABLE:
                overhead_array = np.array(overhead_samples)
                mean_overhead = float(np.mean(overhead_array))
                p95_overhead = float(np.percentile(overhead_array, 95))
                p99_overhead = float(np.percentile(overhead_array, 99))
            else:
                overhead_samples.sort()
                n = len(overhead_samples)
                mean_overhead = sum(overhead_samples) / n
                p95_overhead = overhead_samples[int(n * 0.95)]
                p99_overhead = overhead_samples[int(n * 0.99)]
        else:
            mean_overhead = p95_overhead = p99_overhead = 0.0
        
        # Calculate success rate
        success_rate = ((self._total_events - self._total_errors) / self._total_events * 100) if self._total_events > 0 else 100.0
        
        return {
            'enabled': self.enabled,
            'total_events': self._total_events,
            'total_errors': self._total_errors,
            'success_rate_percent': success_rate,
            'operation_types': len(self._operation_buffers),
            'buffer_utilization': {
                op: buf.count / buf.size * 100
                for op, buf in self._operation_buffers.items()
            },
            'overhead_stats': {
                'mean_ms': mean_overhead,
                'p95_ms': p95_overhead,
                'p99_ms': p99_overhead,
                'target_ms': 1.0,
                'target_met': p95_overhead < 1.0
            },
            'memory_usage': {
                'ring_buffers': len(self._operation_buffers) * self.buffer_size * sys.getsizeof(TelemetryEvent),
                'overhead_samples': len(self._overhead_samples) * sys.getsizeof(0.0)
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_stats': self.get_system_stats(),
            'operations': {}
        }
        
        for operation in self._operation_buffers.keys():
            op_stats = self.get_operation_stats(operation)
            if op_stats:
                summary['operations'][operation] = op_stats
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        timestamp_ms = int(time.time() * 1000)
        
        # System-level metrics
        system_stats = self.get_system_stats()
        lines.extend([
            f'memmimic_telemetry_events_total {system_stats["total_events"]} {timestamp_ms}',
            f'memmimic_telemetry_errors_total {system_stats["total_errors"]} {timestamp_ms}',
            f'memmimic_telemetry_success_rate {system_stats["success_rate_percent"]} {timestamp_ms}',
            f'memmimic_telemetry_overhead_p95_ms {system_stats["overhead_stats"]["p95_ms"]} {timestamp_ms}',
            f'memmimic_telemetry_overhead_p99_ms {system_stats["overhead_stats"]["p99_ms"]} {timestamp_ms}',
            f'memmimic_telemetry_target_met {1 if system_stats["overhead_stats"]["target_met"] else 0} {timestamp_ms}'
        ])
        
        # Operation-specific metrics
        for operation in self._operation_buffers.keys():
            op_stats = self.get_operation_stats(operation)
            if op_stats:
                op_name = operation.replace('-', '_').replace('.', '_')
                lines.extend([
                    f'memmimic_operation_count_total{{operation="{op_name}"}} {op_stats["total_count"]} {timestamp_ms}',
                    f'memmimic_operation_duration_p50_ms{{operation="{op_name}"}} {op_stats["p50_duration_ms"]} {timestamp_ms}',
                    f'memmimic_operation_duration_p95_ms{{operation="{op_name}"}} {op_stats["p95_duration_ms"]} {timestamp_ms}',
                    f'memmimic_operation_duration_p99_ms{{operation="{op_name}"}} {op_stats["p99_duration_ms"]} {timestamp_ms}',
                    f'memmimic_operation_success_rate{{operation="{op_name}"}} {op_stats["success_rate"]} {timestamp_ms}',
                    f'memmimic_operation_error_count{{operation="{op_name}"}} {op_stats["error_count"]} {timestamp_ms}'
                ])
        
        return '\n'.join(lines)
    
    def _start_background_processing(self):
        """Start background aggregation thread"""
        def background_worker():
            while not self._stop_background.wait(self.aggregation_interval):
                try:
                    # Periodic cleanup and optimization
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Background telemetry processing failed: {e}")
        
        self._background_thread = threading.Thread(target=background_worker, daemon=True)
        self._background_thread.start()
        logger.info("Background telemetry processing started")
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        current_time = time.perf_counter()
        cleanup_threshold = current_time - 3600  # 1 hour
        
        with self._stats_lock:
            # Remove stale operation stats
            stale_operations = [
                op for op, stats in self._operation_stats.items()
                if stats['last_update'] < cleanup_threshold
            ]
            
            for op in stale_operations:
                if op in self._operation_stats:
                    del self._operation_stats[op]
                if op in self._operation_buffers:
                    del self._operation_buffers[op]
                logger.debug(f"Cleaned up stale telemetry for operation: {op}")
    
    def reset_stats(self):
        """Reset all statistics (for testing)"""
        with self._lock, self._stats_lock:
            self._total_events = 0
            self._total_errors = 0
            self._operation_buffers.clear()
            self._operation_stats.clear()
            self._overhead_samples.clear()
            self._self_timing.clear()
        
        logger.info("Telemetry statistics reset")
    
    def shutdown(self):
        """Shutdown telemetry collector"""
        try:
            self._stop_background.set()
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)
            
            self.enabled = False
            logger.info("TelemetryCollector shutdown completed")
            
        except Exception as e:
            logger.error(f"TelemetryCollector shutdown failed: {e}")


# Global instance for ultra-fast access
_global_collector: Optional[TelemetryCollector] = None
_collector_lock = threading.Lock()


def get_telemetry_collector() -> TelemetryCollector:
    """Get global telemetry collector instance (thread-safe singleton)"""
    global _global_collector
    
    if _global_collector is None:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = TelemetryCollector()
    
    return _global_collector


def record_operation(
    operation: str,
    duration_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True
) -> None:
    """Ultra-fast operation recording - module-level function for minimal overhead"""
    get_telemetry_collector().record_operation(operation, duration_ms, metadata, success)


def telemetry_timer(operation: str):
    """Decorator for automatic operation timing with minimal overhead"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                record_operation(operation, duration_ms, success=success)
        return wrapper
    return decorator