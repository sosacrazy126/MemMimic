"""
Active Memory Management System - High-Performance Architecture

Optimized active memory management with O(log n) indexing, connection pooling,
and automated lifecycle management for sub-100ms query performance.

Components:
- IndexingEngine: B-tree/hash indexing for fast lookup
- CacheManager: LRU cache with memory limits  
- DatabasePool: Connection pooling and transaction management
- PerformanceMonitor: Real-time performance tracking
- LifecycleCoordinator: Memory lifecycle management
- OptimizationEngine: Automatic performance tuning
"""

from .interfaces import (
    MemoryIndexingEngine,
    MemoryQuery, 
    IndexingConfig,
    IndexingMetrics,
    CacheManager,
    DatabasePool,
    PerformanceMonitor,
    LifecycleCoordinator
)

from .indexing_engine import (
    BTreeIndexingEngine,
    create_indexing_engine
)

from .cache_manager import (
    LRUMemoryCache,
    create_cache_manager
)

from .database_pool import (
    ConnectionPool,
    create_database_pool
)

from .performance_monitor import (
    RealTimePerformanceMonitor,
    create_performance_monitor
)

from .optimization_engine import (
    AutomaticOptimizationEngine,
    create_optimization_engine
)

__all__ = [
    # Interfaces
    'MemoryIndexingEngine',
    'MemoryQuery',
    'IndexingConfig', 
    'IndexingMetrics',
    'CacheManager',
    'DatabasePool',
    'PerformanceMonitor',
    'LifecycleCoordinator',
    
    # Implementations
    'BTreeIndexingEngine',
    'LRUMemoryCache', 
    'ConnectionPool',
    'RealTimePerformanceMonitor',
    'AutomaticOptimizationEngine',
    
    # Factory functions
    'create_indexing_engine',
    'create_cache_manager', 
    'create_database_pool',
    'create_performance_monitor',
    'create_optimization_engine'
]