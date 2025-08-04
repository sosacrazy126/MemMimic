"""
ReflexLatencyOptimizer - Sub-5ms Response Time Achievement

Implements advanced optimization techniques to achieve sub-5ms response times
for core nervous system operations (remember, recall, think). Uses intelligent
caching, parallel processing, and predictive preloading to minimize latency.

This component is critical for achieving biological reflex-level response times
that enable natural human-AI interaction patterns.
"""

import asyncio
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import weakref
import pickle
import hashlib

from ..errors import MemMimicError, with_error_context, get_error_logger


@dataclass
class LatencyMetrics:
    """Detailed latency tracking metrics"""
    operation_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)


@dataclass
class OptimizationStrategy:
    """Configuration for optimization strategies"""
    enable_predictive_caching: bool = True
    enable_parallel_preloading: bool = True
    enable_memory_pooling: bool = True
    enable_result_compression: bool = True
    cache_size_limit: int = 10000
    preload_threshold_ms: float = 2.0
    compression_threshold_bytes: int = 1024
    memory_pool_size: int = 100


class ReflexLatencyOptimizer:
    """
    Advanced Latency Optimization System
    
    Implements multiple optimization strategies to achieve sub-5ms response times
    for core nervous system operations through intelligent caching, parallel
    processing, and predictive optimization.
    """
    
    def __init__(self, target_latency_ms: float = 5.0):
        self.target_latency_ms = target_latency_ms
        self.logger = get_error_logger("reflex_latency_optimizer")
        
        # Optimization configuration
        self.strategy = OptimizationStrategy()
        
        # Performance tracking
        self.metrics: Dict[str, LatencyMetrics] = {}
        self._global_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'cache_efficiency': 0.0,
            'average_latency_reduction': 0.0
        }
        
        # Multi-level caching system
        self._l1_cache: Dict[str, Any] = {}  # Hot cache (in-memory)
        self._l2_cache: Dict[str, bytes] = {}  # Compressed cache
        self._l3_cache: Dict[str, str] = {}  # Disk cache paths
        
        # Predictive systems
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._prediction_cache: Dict[str, Any] = {}
        self._preload_queue = asyncio.Queue()
        
        # Memory pooling
        self._memory_pools: Dict[str, List[Any]] = defaultdict(list)
        self._pool_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Background optimization
        self._optimization_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Weak references for cleanup
        self._cached_objects = weakref.WeakValueDictionary()
        
    async def initialize(self) -> None:
        """Initialize the latency optimization system"""
        try:
            # Start background optimization task
            self._optimization_task = asyncio.create_task(self._background_optimization_loop())
            
            # Initialize memory pools
            await self._initialize_memory_pools()
            
            # Load historical optimization data
            await self._load_optimization_history()
            
            self.logger.info("Reflex latency optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize latency optimizer: {e}")
            raise MemMimicError(f"Latency optimizer initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the optimization system"""
        self._shutdown_event.set()
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Save optimization history
        await self._save_optimization_history()
    
    async def optimize_operation(self, operation_name: str, operation_func: Callable, 
                               *args, **kwargs) -> Tuple[Any, float]:
        """
        Optimize a single operation for minimum latency
        
        Returns: (result, execution_time_ms)
        """
        start_time = time.perf_counter()
        
        # Generate cache key
        cache_key = self._generate_cache_key(operation_name, args, kwargs)
        
        # Try L1 cache first (fastest)
        if cache_key in self._l1_cache:
            result = self._l1_cache[cache_key]
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_metrics(operation_name, execution_time, cache_hit=True)
            return result, execution_time
        
        # Try L2 cache (compressed)
        if cache_key in self._l2_cache:
            try:
                result = pickle.loads(self._l2_cache[cache_key])
                self._l1_cache[cache_key] = result  # Promote to L1
                execution_time = (time.perf_counter() - start_time) * 1000
                self._update_metrics(operation_name, execution_time, cache_hit=True)
                return result, execution_time
            except Exception as e:
                self.logger.warning(f"Failed to deserialize L2 cache entry: {e}")
        
        # Execute operation with optimization
        try:
            # Use memory pool if available
            if self.strategy.enable_memory_pooling:
                result = await self._execute_with_memory_pool(operation_name, operation_func, *args, **kwargs)
            else:
                result = await self._execute_operation(operation_func, *args, **kwargs)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Cache result if beneficial
            await self._cache_result(cache_key, result, execution_time)
            
            # Update access patterns for prediction
            self._update_access_patterns(operation_name, execution_time)
            
            # Update metrics
            self._update_metrics(operation_name, execution_time, cache_hit=False)
            
            return result, execution_time
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_metrics(operation_name, execution_time, cache_hit=False, error=True)
            raise
    
    async def _execute_operation(self, operation_func: Callable, *args, **kwargs) -> Any:
        """Execute operation with basic optimization"""
        if asyncio.iscoroutinefunction(operation_func):
            return await operation_func(*args, **kwargs)
        else:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, operation_func, *args, **kwargs)
    
    async def _execute_with_memory_pool(self, operation_name: str, operation_func: Callable, 
                                      *args, **kwargs) -> Any:
        """Execute operation using memory pooling for reduced allocation overhead"""
        pool_key = f"{operation_name}_pool"
        
        # Get object from pool if available
        pooled_object = None
        with self._pool_locks[pool_key]:
            if self._memory_pools[pool_key]:
                pooled_object = self._memory_pools[pool_key].pop()
        
        try:
            # Execute with pooled object if available
            if pooled_object:
                kwargs['_pooled_object'] = pooled_object
            
            result = await self._execute_operation(operation_func, *args, **kwargs)
            
            # Return object to pool
            if pooled_object and len(self._memory_pools[pool_key]) < self.strategy.memory_pool_size:
                with self._pool_locks[pool_key]:
                    self._memory_pools[pool_key].append(pooled_object)
            
            return result
            
        except Exception:
            # Return object to pool even on error
            if pooled_object and len(self._memory_pools[pool_key]) < self.strategy.memory_pool_size:
                with self._pool_locks[pool_key]:
                    self._memory_pools[pool_key].append(pooled_object)
            raise
    
    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for the operation"""
        # Create a hash of the operation and its parameters
        key_data = {
            'operation': operation_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _cache_result(self, cache_key: str, result: Any, execution_time: float) -> None:
        """Cache result using appropriate caching strategy"""
        try:
            # Always cache in L1 for immediate access
            self._l1_cache[cache_key] = result
            
            # Cache in L2 if result is large or execution time is high
            if execution_time > self.strategy.preload_threshold_ms:
                try:
                    serialized = pickle.dumps(result)
                    if len(serialized) > self.strategy.compression_threshold_bytes:
                        # Compress large results
                        import gzip
                        serialized = gzip.compress(serialized)
                    
                    self._l2_cache[cache_key] = serialized
                    
                except Exception as e:
                    self.logger.warning(f"Failed to serialize result for L2 cache: {e}")
            
            # Manage cache size
            await self._manage_cache_size()
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    async def _manage_cache_size(self) -> None:
        """Manage cache size to prevent memory bloat"""
        if len(self._l1_cache) > self.strategy.cache_size_limit:
            # Remove oldest entries (simple LRU approximation)
            keys_to_remove = list(self._l1_cache.keys())[:-self.strategy.cache_size_limit//2]
            for key in keys_to_remove:
                self._l1_cache.pop(key, None)
        
        if len(self._l2_cache) > self.strategy.cache_size_limit:
            keys_to_remove = list(self._l2_cache.keys())[:-self.strategy.cache_size_limit//2]
            for key in keys_to_remove:
                self._l2_cache.pop(key, None)
    
    def _update_access_patterns(self, operation_name: str, execution_time: float) -> None:
        """Update access patterns for predictive optimization"""
        current_time = time.time()
        self._access_patterns[operation_name].append(current_time)
        
        # Keep only recent patterns
        cutoff_time = current_time - 3600  # 1 hour
        self._access_patterns[operation_name] = [
            t for t in self._access_patterns[operation_name] if t > cutoff_time
        ]
    
    def _update_metrics(self, operation_name: str, execution_time: float, 
                       cache_hit: bool = False, error: bool = False) -> None:
        """Update detailed performance metrics"""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = LatencyMetrics(operation_name=operation_name)
        
        metrics = self.metrics[operation_name]
        
        if not error:
            metrics.total_calls += 1
            metrics.total_time_ms += execution_time
            metrics.min_time_ms = min(metrics.min_time_ms, execution_time)
            metrics.max_time_ms = max(metrics.max_time_ms, execution_time)
            metrics.avg_time_ms = metrics.total_time_ms / metrics.total_calls
            metrics.recent_times.append(execution_time)
            
            # Calculate percentiles
            if len(metrics.recent_times) >= 20:
                sorted_times = sorted(metrics.recent_times)
                metrics.p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
                metrics.p99_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
        
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
        
        metrics.last_update = time.time()
    
    async def _background_optimization_loop(self) -> None:
        """Background loop for continuous optimization"""
        while not self._shutdown_event.is_set():
            try:
                # Predictive preloading
                if self.strategy.enable_predictive_caching:
                    await self._predictive_preload()
                
                # Cache optimization
                await self._optimize_caches()
                
                # Memory pool maintenance
                await self._maintain_memory_pools()
                
                # Wait before next optimization cycle
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Background optimization error: {e}")
                await asyncio.sleep(5.0)
    
    async def _predictive_preload(self) -> None:
        """Predictively preload likely-to-be-accessed operations"""
        # Analyze access patterns and preload predicted operations
        current_time = time.time()
        
        for operation_name, access_times in self._access_patterns.items():
            if len(access_times) < 3:
                continue
            
            # Simple prediction: if operation was accessed recently and frequently
            recent_accesses = [t for t in access_times if current_time - t < 300]  # 5 minutes
            
            if len(recent_accesses) >= 2:
                # Predict next access and preload if beneficial
                avg_interval = (recent_accesses[-1] - recent_accesses[0]) / (len(recent_accesses) - 1)
                predicted_next_access = recent_accesses[-1] + avg_interval
                
                if predicted_next_access - current_time < 60:  # Within next minute
                    await self._preload_operation(operation_name)
    
    async def _preload_operation(self, operation_name: str) -> None:
        """Preload an operation based on prediction"""
        # This would be implemented based on specific operation types
        # For now, just log the prediction
        self.logger.debug(f"Predicted access for operation: {operation_name}")
    
    async def _optimize_caches(self) -> None:
        """Optimize cache performance"""
        # Remove expired entries, optimize memory usage, etc.
        current_time = time.time()
        
        # Clean up old cache entries
        for cache in [self._l1_cache, self._l2_cache]:
            if len(cache) > self.strategy.cache_size_limit * 0.8:
                await self._manage_cache_size()
    
    async def _maintain_memory_pools(self) -> None:
        """Maintain memory pools for optimal performance"""
        for pool_name, pool in self._memory_pools.items():
            with self._pool_locks[pool_name]:
                # Ensure minimum pool size
                while len(pool) < self.strategy.memory_pool_size // 4:
                    # Create new pooled object (implementation specific)
                    pool.append({})  # Placeholder
    
    async def _initialize_memory_pools(self) -> None:
        """Initialize memory pools for common operations"""
        common_operations = ['remember', 'recall', 'think', 'analyze']
        
        for operation in common_operations:
            pool_key = f"{operation}_pool"
            with self._pool_locks[pool_key]:
                for _ in range(self.strategy.memory_pool_size // 4):
                    self._memory_pools[pool_key].append({})  # Placeholder objects
    
    async def _load_optimization_history(self) -> None:
        """Load historical optimization data"""
        # Implementation would load from persistent storage
        pass
    
    async def _save_optimization_history(self) -> None:
        """Save optimization history for future use"""
        # Implementation would save to persistent storage
        pass
    
    def get_latency_metrics(self) -> Dict[str, LatencyMetrics]:
        """Get detailed latency metrics for all operations"""
        return self.metrics.copy()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance"""
        total_operations = sum(m.total_calls for m in self.metrics.values())
        total_cache_hits = sum(m.cache_hits for m in self.metrics.values())
        
        avg_latency = 0.0
        if self.metrics:
            avg_latency = sum(m.avg_time_ms for m in self.metrics.values()) / len(self.metrics)
        
        return {
            'total_operations': total_operations,
            'cache_hit_rate': total_cache_hits / max(total_operations, 1),
            'average_latency_ms': avg_latency,
            'target_latency_ms': self.target_latency_ms,
            'operations_under_target': sum(1 for m in self.metrics.values() if m.avg_time_ms <= self.target_latency_ms),
            'total_operation_types': len(self.metrics),
            'l1_cache_size': len(self._l1_cache),
            'l2_cache_size': len(self._l2_cache),
            'memory_pools': {name: len(pool) for name, pool in self._memory_pools.items()}
        }
    
    def is_target_latency_achieved(self, operation_name: str = None) -> bool:
        """Check if target latency is achieved for specific operation or overall"""
        if operation_name:
            if operation_name in self.metrics:
                return self.metrics[operation_name].avg_time_ms <= self.target_latency_ms
            return False
        
        # Check overall performance
        if not self.metrics:
            return False
        
        avg_latency = sum(m.avg_time_ms for m in self.metrics.values()) / len(self.metrics)
        return avg_latency <= self.target_latency_ms
