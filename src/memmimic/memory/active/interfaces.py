"""
Interfaces and abstractions for high-performance active memory management.

Defines the contracts for indexing engines, cache managers, database pools,
and performance monitoring components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Protocol
import threading


class IndexType(Enum):
    """Types of indexes supported by the indexing engine"""
    BTREE = "btree"
    HASH = "hash"
    FULLTEXT = "fulltext"
    TEMPORAL = "temporal"


class MemoryStatus(Enum):
    """Memory lifecycle status"""
    ACTIVE = "active"
    ARCHIVED = "archived" 
    PRUNE_CANDIDATE = "prune_candidate"
    DELETED = "deleted"


@dataclass
class MemoryQuery:
    """Query specification for memory search operations"""
    memory_ids: Optional[List[str]] = None
    content_search: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    status_filter: Optional[MemoryStatus] = None
    importance_threshold: Optional[float] = None
    limit: int = 100
    offset: int = 0


@dataclass
class IndexingConfig:
    """Configuration for memory indexing engine"""
    enable_btree_index: bool = True
    enable_hash_index: bool = True
    enable_fulltext_index: bool = True
    enable_temporal_index: bool = True
    
    # Index-specific settings
    btree_cache_size: int = 1000
    hash_bucket_count: int = 4096
    fulltext_min_word_length: int = 3
    temporal_resolution_minutes: int = 60
    
    # Performance settings
    index_update_batch_size: int = 100
    background_optimization: bool = True
    auto_defragmentation: bool = True
    
    # Monitoring
    enable_performance_tracking: bool = True
    metrics_retention_hours: int = 24


@dataclass
class IndexingMetrics:
    """Performance metrics for indexing operations"""
    total_memories_indexed: int = 0
    index_build_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    index_hit_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Index-specific metrics
    btree_depth: int = 0
    hash_collision_rate: float = 0.0
    fulltext_terms_indexed: int = 0
    temporal_buckets: int = 0
    
    # Performance tracking
    queries_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    last_optimization: Optional[datetime] = None
    
    def reset(self):
        """Reset all metrics to default values"""
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_def.type in (int, float):
                setattr(self, field_name, 0 if field_def.type == int else 0.0)
            elif field_def.type == Optional[datetime]:
                setattr(self, field_name, None)


class MemoryIndexingEngine(Protocol):
    """High-performance memory indexing with multiple access patterns"""
    
    @abstractmethod
    def index_memory(self, memory_id: str, content: str, 
                    metadata: Dict[str, Any], created_at: datetime) -> None:
        """Add memory to all relevant indexes"""
        pass
    
    @abstractmethod
    def remove_memory(self, memory_id: str) -> None:
        """Remove memory from all indexes"""
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, content: str,
                     metadata: Dict[str, Any]) -> None:
        """Update memory in all indexes"""
        pass
    
    @abstractmethod
    def search_memories(self, query: MemoryQuery) -> List[str]:
        """Search with O(log n) performance using appropriate indexes"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> IndexingMetrics:
        """Get current indexing performance metrics"""
        pass
    
    @abstractmethod
    def optimize_indexes(self) -> Dict[str, Any]:
        """Optimize all indexes for performance"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check index health and integrity"""
        pass


@dataclass
class CacheEntry:
    """Cache entry with metadata and TTL support"""
    value: Any
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


class CacheManager(Protocol):
    """Memory-aware LRU cache with automatic eviction"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU update"""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put item in cache with automatic eviction"""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove specific item from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        pass
    
    @abstractmethod
    def evict_expired(self) -> int:
        """Evict all expired entries"""
        pass


class DatabasePool(Protocol):
    """Database connection pooling with transaction management"""
    
    @abstractmethod
    def get_connection(self) -> Any:
        """Get database connection from pool"""
        pass
    
    @abstractmethod
    def return_connection(self, connection: Any) -> None:
        """Return connection to pool"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: tuple = ()) -> Any:
        """Execute query with automatic connection management"""
        pass
    
    @abstractmethod
    def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute multiple queries in a transaction"""
        pass
    
    @abstractmethod
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check pool health and connectivity"""
        pass


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Query performance
    avg_query_time_ms: float = 0.0
    queries_per_second: float = 0.0
    query_success_rate: float = 0.0
    
    # Memory usage
    total_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    index_memory_mb: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_utilization: float = 0.0
    
    # Database performance
    db_connections_active: int = 0
    db_avg_response_time_ms: float = 0.0
    
    # System health
    error_rate: float = 0.0
    health_score: float = 0.0


class PerformanceMonitor(Protocol):
    """Real-time performance tracking"""
    
    @abstractmethod
    def record_query(self, query_time_ms: float, success: bool) -> None:
        """Record query performance metrics"""
        pass
    
    @abstractmethod
    def record_cache_operation(self, hit: bool, operation_time_ms: float) -> None:
        """Record cache operation metrics"""
        pass
    
    @abstractmethod
    def record_memory_usage(self, component: str, memory_mb: float) -> None:
        """Record memory usage for component"""
        pass
    
    @abstractmethod
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        pass
    
    @abstractmethod
    def get_historical_data(self, hours: int = 1) -> List[PerformanceSnapshot]:
        """Get historical performance data"""
        pass
    
    @abstractmethod
    def check_thresholds(self) -> Dict[str, Any]:
        """Check if any performance thresholds are exceeded"""
        pass


class LifecycleCoordinator(Protocol):
    """Memory lifecycle management and optimization"""
    
    @abstractmethod
    def evaluate_memory_importance(self, memory_id: str) -> float:
        """Evaluate current importance score for memory"""
        pass
    
    @abstractmethod
    def suggest_archival_candidates(self, max_candidates: int = 100) -> List[str]:
        """Suggest memories for archival based on usage patterns"""
        pass
    
    @abstractmethod
    def suggest_deletion_candidates(self, max_candidates: int = 50) -> List[str]:
        """Suggest memories for deletion based on age and importance"""
        pass
    
    @abstractmethod
    def optimize_memory_distribution(self) -> Dict[str, Any]:
        """Optimize memory distribution across tiers"""
        pass
    
    @abstractmethod
    def predict_memory_usage(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future memory usage patterns"""
        pass


# Exception hierarchy for active memory management
class ActiveMemoryError(Exception):
    """Base exception for active memory management operations"""
    
    def __init__(self, message: str, error_code: str = None, 
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now()


class IndexingError(ActiveMemoryError):
    """Errors in memory indexing operations"""
    pass


class CacheError(ActiveMemoryError):
    """Errors in cache operations"""
    pass


class DatabasePoolError(ActiveMemoryError):
    """Errors in database pool operations"""
    pass


class PerformanceError(ActiveMemoryError):
    """Performance-related errors and threshold violations"""
    pass


class LifecycleError(ActiveMemoryError):
    """Memory lifecycle management errors"""
    pass


# Threading and concurrency support
class ThreadSafeCounter:
    """Thread-safe counter for metrics collection"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def set(self, value: int) -> None:
        with self._lock:
            self._value = value


class ThreadSafeMetrics:
    """Thread-safe metrics collection for concurrent environments"""
    
    def __init__(self):
        self._metrics: Dict[str, Union[ThreadSafeCounter, float]] = {}
        self._lock = threading.RLock()
    
    def increment_counter(self, name: str, amount: int = 1) -> int:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = ThreadSafeCounter()
            counter = self._metrics[name]
            if isinstance(counter, ThreadSafeCounter):
                return counter.increment(amount)
            return 0
    
    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._metrics[name] = value
    
    def get_metric(self, name: str) -> Union[int, float]:
        with self._lock:
            metric = self._metrics.get(name, 0)
            if isinstance(metric, ThreadSafeCounter):
                return metric.get()
            return metric
    
    def get_all_metrics(self) -> Dict[str, Union[int, float]]:
        with self._lock:
            result = {}
            for name, metric in self._metrics.items():
                if isinstance(metric, ThreadSafeCounter):
                    result[name] = metric.get()
                else:
                    result[name] = metric
            return result