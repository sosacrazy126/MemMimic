"""
Enterprise Metrics Collection System for MemMimic

Collects comprehensive metrics compatible with Prometheus and other monitoring systems.
Provides real-time performance, resource, and operational metrics.
"""

import time
import threading
import psutil
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "gauge"  # gauge, counter, histogram, summary


@dataclass 
class Counter:
    """Thread-safe counter metric"""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def increment(self, amount: float = 1.0):
        with self._lock:
            self.value += amount
    
    def get(self) -> float:
        with self._lock:
            return self.value
    
    def reset(self):
        with self._lock:
            self.value = 0.0


@dataclass
class Gauge:
    """Thread-safe gauge metric"""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def set(self, value: float):
        with self._lock:
            self.value = value
    
    def get(self) -> float:
        with self._lock:
            return self.value
    
    def increment(self, amount: float = 1.0):
        with self._lock:
            self.value += amount
    
    def decrement(self, amount: float = 1.0):
        with self._lock:
            self.value -= amount


@dataclass
class Histogram:
    """Thread-safe histogram metric"""
    buckets: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    count: int = 0
    sum: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        if not self.buckets:
            # Default Prometheus buckets for duration metrics
            default_buckets = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')]
            self.buckets = {bucket: 0 for bucket in default_buckets}
    
    def observe(self, value: float):
        with self._lock:
            self.count += 1
            self.sum += value
            
            for bucket in sorted(self.buckets.keys()):
                if value <= bucket:
                    self.buckets[bucket] += 1
    
    def get_buckets(self) -> Dict[float, int]:
        with self._lock:
            return dict(self.buckets)
    
    def get_count(self) -> int:
        with self._lock:
            return self.count
    
    def get_sum(self) -> float:
        with self._lock:
            return self.sum


class MetricsCollector:
    """
    Enterprise-grade metrics collection system for MemMimic.
    
    Features:
    - Thread-safe metric collection
    - Prometheus-compatible exports
    - Real-time performance monitoring
    - Resource usage tracking
    - Custom metric registration
    - Historical data retention
    - Automatic system metrics collection
    """
    
    def __init__(self, collection_interval: float = 10.0, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # Metric storage
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        
        # Historical metrics storage
        self._historical_metrics: deque = deque(maxlen=int(retention_hours * 3600 / collection_interval))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background collection
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        # Initialize standard MemMimic metrics
        self._initialize_memmimic_metrics()
        
        # Start background collection
        self._start_background_collection()
        
        logger.info("Enterprise Metrics Collector initialized")
    
    def _initialize_memmimic_metrics(self):
        """Initialize standard MemMimic metrics"""
        # Memory operations
        self.register_counter("memmimic_memories_stored_total", "Total memories stored")
        self.register_counter("memmimic_memories_retrieved_total", "Total memories retrieved") 
        self.register_counter("memmimic_searches_total", "Total searches performed")
        self.register_counter("memmimic_cxd_classifications_total", "Total CXD classifications")
        
        # Performance metrics
        self.register_histogram("memmimic_operation_duration_seconds", "Operation duration")
        self.register_histogram("memmimic_search_duration_seconds", "Search operation duration")
        self.register_histogram("memmimic_storage_duration_seconds", "Storage operation duration")
        self.register_histogram("memmimic_cxd_classification_duration_seconds", "CXD classification duration")
        
        # Cache metrics
        self.register_counter("memmimic_cache_hits_total", "Cache hits")
        self.register_counter("memmimic_cache_misses_total", "Cache misses")
        self.register_gauge("memmimic_cache_size_bytes", "Cache size in bytes")
        self.register_gauge("memmimic_cache_entries_count", "Number of cache entries")
        
        # Database metrics
        self.register_counter("memmimic_db_queries_total", "Database queries")
        self.register_histogram("memmimic_db_query_duration_seconds", "Database query duration")
        self.register_gauge("memmimic_db_connections_active", "Active database connections")
        self.register_gauge("memmimic_db_size_bytes", "Database size in bytes")
        
        # Quality metrics
        self.register_histogram("memmimic_quality_score", "Quality scores")
        self.register_counter("memmimic_quality_gate_pass_total", "Quality gate passes")
        self.register_counter("memmimic_quality_gate_fail_total", "Quality gate failures")
        
        # System health
        self.register_gauge("memmimic_health_score", "Overall system health score")
        self.register_counter("memmimic_errors_total", "Total errors by type")
        self.register_counter("memmimic_alerts_total", "Total alerts by severity")
        
        # Resource usage
        self.register_gauge("memmimic_memory_usage_bytes", "Memory usage in bytes")
        self.register_gauge("memmimic_cpu_usage_percent", "CPU usage percentage")
        self.register_gauge("memmimic_disk_usage_bytes", "Disk usage in bytes")
    
    def register_counter(self, name: str, description: str) -> Counter:
        """Register a new counter metric"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter()
                logger.debug(f"Registered counter: {name}")
            return self._counters[name]
    
    def register_gauge(self, name: str, description: str) -> Gauge:
        """Register a new gauge metric"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge()
                logger.debug(f"Registered gauge: {name}")
            return self._gauges[name]
    
    def register_histogram(self, name: str, description: str, buckets: Optional[List[float]] = None) -> Histogram:
        """Register a new histogram metric"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram()
                if buckets:
                    self._histograms[name].buckets = {bucket: 0 for bucket in buckets}
                logger.debug(f"Registered histogram: {name}")
            return self._histograms[name]
    
    def get_counter(self, name: str) -> Optional[Counter]:
        """Get counter by name"""
        return self._counters.get(name)
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get gauge by name"""
        return self._gauges.get(name)
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get histogram by name"""
        return self._histograms.get(name)
    
    def increment_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter metric"""
        counter = self.get_counter(name)
        if counter:
            counter.increment(amount)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric value"""
        gauge = self.get_gauge(name)
        if gauge:
            gauge.set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe value in histogram"""
        histogram = self.get_histogram(name)
        if histogram:
            histogram.observe(value)
    
    def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.set_gauge("memmimic_memory_usage_bytes", memory_info.used)
            self.set_gauge("memmimic_memory_available_bytes", memory_info.available)
            self.set_gauge("memmimic_memory_usage_percent", memory_info.percent)
            
            # CPU usage  
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("memmimic_cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            self.set_gauge("memmimic_disk_usage_bytes", disk_usage.used)
            self.set_gauge("memmimic_disk_available_bytes", disk_usage.free)
            self.set_gauge("memmimic_disk_usage_percent", (disk_usage.used / disk_usage.total) * 100)
            
            # Load average (Unix/Linux only)
            try:
                load_avg = psutil.getloadavg()
                self.set_gauge("memmimic_load_average_1m", load_avg[0])
                self.set_gauge("memmimic_load_average_5m", load_avg[1]) 
                self.set_gauge("memmimic_load_average_15m", load_avg[2])
            except AttributeError:
                pass  # Windows doesn't have load average
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def export_prometheus_format(self) -> str:
        """Export all metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            # Export counters
            for name, counter in self._counters.items():
                lines.append(f'# TYPE {name} counter')
                lines.append(f'{name} {counter.get()}')
            
            # Export gauges
            for name, gauge in self._gauges.items():
                lines.append(f'# TYPE {name} gauge')
                lines.append(f'{name} {gauge.get()}')
            
            # Export histograms
            for name, histogram in self._histograms.items():
                lines.append(f'# TYPE {name} histogram')
                
                # Export buckets
                buckets = histogram.get_buckets()
                for bucket, count in buckets.items():
                    if bucket == float('inf'):
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
                
                # Export count and sum
                lines.append(f'{name}_count {histogram.get_count()}')
                lines.append(f'{name}_sum {histogram.get_sum()}')
        
        return '\n'.join(lines) + '\n'
    
    def export_json_format(self) -> str:
        """Export all metrics in JSON format"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'counters': {},
                'gauges': {},
                'histograms': {}
            }
        }
        
        with self._lock:
            # Export counters
            for name, counter in self._counters.items():
                data['metrics']['counters'][name] = counter.get()
            
            # Export gauges  
            for name, gauge in self._gauges.items():
                data['metrics']['gauges'][name] = gauge.get()
            
            # Export histograms
            for name, histogram in self._histograms.items():
                data['metrics']['histograms'][name] = {
                    'buckets': histogram.get_buckets(),
                    'count': histogram.get_count(),
                    'sum': histogram.get_sum()
                }
        
        return json.dumps(data, indent=2)
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            return {
                'counters': len(self._counters),
                'gauges': len(self._gauges),
                'histograms': len(self._histograms),
                'collection_interval': self.collection_interval,
                'retention_hours': self.retention_hours,
                'historical_points': len(self._historical_metrics),
                'last_collection': datetime.now().isoformat()
            }
    
    def _start_background_collection(self):
        """Start background metrics collection thread"""
        def collection_worker():
            while not self._stop_collection.wait(self.collection_interval):
                try:
                    # Collect system metrics
                    self.collect_system_metrics()
                    
                    # Store historical snapshot
                    snapshot = {
                        'timestamp': time.time(),
                        'metrics': {
                            'counters': {name: counter.get() for name, counter in self._counters.items()},
                            'gauges': {name: gauge.get() for name, gauge in self._gauges.items()}
                        }
                    }
                    self._historical_metrics.append(snapshot)
                    
                except Exception as e:
                    logger.error(f"Background metrics collection failed: {e}")
        
        self._collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self._collection_thread.start()
        logger.info("Background metrics collection started")
    
    def get_historical_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            snapshot for snapshot in self._historical_metrics
            if snapshot['timestamp'] >= cutoff_time
        ]
    
    def shutdown(self):
        """Shutdown metrics collector"""
        try:
            self._stop_collection.set()
            if self._collection_thread and self._collection_thread.is_alive():
                self._collection_thread.join(timeout=5.0)
            
            logger.info("Metrics collector shutdown completed")
            
        except Exception as e:
            logger.error(f"Metrics collector shutdown failed: {e}")


# Global metrics collector instance
_metrics_collector_instance = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance (thread-safe singleton)"""
    global _metrics_collector_instance
    
    if _metrics_collector_instance is None:
        with _collector_lock:
            if _metrics_collector_instance is None:
                _metrics_collector_instance = MetricsCollector()
    
    return _metrics_collector_instance


def metrics_timer(metric_name: str):
    """Decorator to time function execution and record in histogram"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics_collector().observe_histogram(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics_collector().observe_histogram(f"{metric_name}_error", duration)
                get_metrics_collector().increment_counter("memmimic_errors_total")
                raise
        return wrapper
    return decorator


def metrics_counter(metric_name: str):
    """Decorator to count function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            get_metrics_collector().increment_counter(metric_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator