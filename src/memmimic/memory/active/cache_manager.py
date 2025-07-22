"""
High-performance LRU cache with memory management and automatic eviction.

Provides memory-aware caching with TTL support, automatic eviction policies,
and comprehensive performance monitoring for optimal memory utilization.
"""

import logging
import time
import threading
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import weakref

from .interfaces import (
    CacheManager, CacheEntry, CacheError, PerformanceSnapshot,
    ThreadSafeMetrics
)

logger = logging.getLogger(__name__)


class MemoryEstimator:
    """Utility class for estimating memory usage of objects"""
    
    @staticmethod
    def estimate_size(obj: Any) -> int:
        """
        Estimate memory size of an object in bytes.
        
        Args:
            obj: Object to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Use sys.getsizeof as base estimate
            size = sys.getsizeof(obj)
            
            # Add estimates for container contents
            if isinstance(obj, dict):
                for key, value in obj.items():
                    size += sys.getsizeof(key) + sys.getsizeof(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    size += sys.getsizeof(item)
            elif isinstance(obj, str):
                # String size is already accurate from sys.getsizeof
                pass
            elif hasattr(obj, '__dict__'):
                # Object with attributes
                size += sys.getsizeof(obj.__dict__)
                for attr_value in obj.__dict__.values():
                    size += sys.getsizeof(attr_value)
            
            return size
            
        except Exception:
            # Fallback estimate
            return len(str(obj)) * 2  # Rough estimate: 2 bytes per character


class LRUMemoryCache(CacheManager):
    """
    High-performance LRU cache with memory management and TTL support.
    
    Features:
    - LRU eviction when memory or size limits reached
    - TTL (time-to-live) expiration for entries
    - Memory usage tracking and limits
    - Performance metrics and monitoring
    - Thread-safe operations
    - Automatic background cleanup
    """
    
    def __init__(self, max_memory_mb: int = 512, max_items: int = 10000,
                 default_ttl_seconds: int = 3600, cleanup_interval_seconds: int = 300):
        """
        Initialize LRU memory cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_items: Maximum number of cache entries
            default_ttl_seconds: Default TTL for entries (0 = no expiration)
            cleanup_interval_seconds: Background cleanup interval
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_items = max_items
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Cache storage: OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_memory_usage = 0
        self._lock = threading.RLock()
        
        # Performance metrics
        self._metrics = ThreadSafeMetrics()
        self._start_time = time.time()
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_background_cleanup()
        
        # Memory pressure detection
        self._memory_pressure_threshold = 0.8  # 80% of max memory
        self._emergency_eviction_threshold = 0.95  # 95% of max memory
        
        logger.info(f"LRUMemoryCache initialized: {max_memory_mb}MB, {max_items} items")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache with LRU update.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                self._metrics.increment_counter('total_gets')
                
                # Check if key exists
                if key not in self._cache:
                    self._metrics.increment_counter('cache_misses')
                    return None
                
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self._metrics.increment_counter('cache_misses')
                    self._metrics.increment_counter('expired_entries')
                    return None
                
                # Update access tracking
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._metrics.increment_counter('cache_hits')
                return entry.value
                
        except Exception as e:
            logger.error(f"Cache get failed for key '{key}': {e}")
            raise CacheError(f"Cache get failed: {e}", context={'key': key})
        
        finally:
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics.set_gauge('last_get_time_ms', operation_time)
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Put item in cache with automatic eviction.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        start_time = time.perf_counter()
        
        try:
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds if self.default_ttl_seconds > 0 else None
            
            # Estimate entry size
            entry_size = self._estimate_entry_size(key, value)
            
            with self._lock:
                self._metrics.increment_counter('total_puts')
                
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                
                # Check if single entry exceeds memory limit
                if entry_size > self.max_memory_bytes:
                    logger.warning(f"Entry size ({entry_size} bytes) exceeds cache limit")
                    raise CacheError(
                        f"Entry too large: {entry_size} bytes",
                        error_code="ENTRY_TOO_LARGE",
                        context={'key': key, 'size': entry_size}
                    )
                
                # Evict entries if necessary
                self._ensure_capacity(entry_size)
                
                # Create and add new entry
                entry = CacheEntry(
                    value=value,
                    size_bytes=entry_size,
                    ttl_seconds=ttl_seconds
                )
                
                self._cache[key] = entry
                self._current_memory_usage += entry_size
                
                self._metrics.increment_counter('cache_puts')
                self._metrics.set_gauge('current_memory_mb', self._current_memory_usage / (1024 * 1024))
                self._metrics.set_gauge('current_items', len(self._cache))
                
        except Exception as e:
            logger.error(f"Cache put failed for key '{key}': {e}")
            if not isinstance(e, CacheError):
                raise CacheError(f"Cache put failed: {e}", context={'key': key})
            raise
        
        finally:
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics.set_gauge('last_put_time_ms', operation_time)
    
    def remove(self, key: str) -> bool:
        """
        Remove specific item from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if item was removed, False if not found
        """
        try:
            with self._lock:
                if key in self._cache:
                    self._remove_entry(key)
                    self._metrics.increment_counter('manual_removals')
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache remove failed for key '{key}': {e}")
            raise CacheError(f"Cache remove failed: {e}", context={'key': key})
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._lock:
                self._cache.clear()
                self._current_memory_usage = 0
                self._metrics.increment_counter('cache_clears')
                self._metrics.set_gauge('current_memory_mb', 0)
                self._metrics.set_gauge('current_items', 0)
                
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            raise CacheError(f"Cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache metrics and performance data
        """
        try:
            with self._lock:
                metrics = self._metrics.get_all_metrics()
                
                # Calculate derived metrics
                total_requests = metrics.get('total_gets', 0)
                cache_hits = metrics.get('cache_hits', 0)
                cache_misses = metrics.get('cache_misses', 0)
                
                hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
                miss_rate = cache_misses / total_requests if total_requests > 0 else 0.0
                
                memory_utilization = self._current_memory_usage / self.max_memory_bytes
                item_utilization = len(self._cache) / self.max_items
                
                uptime_seconds = time.time() - self._start_time
                
                return {
                    # Basic stats
                    'current_items': len(self._cache),
                    'max_items': self.max_items,
                    'current_memory_mb': self._current_memory_usage / (1024 * 1024),
                    'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                    
                    # Utilization
                    'memory_utilization': memory_utilization,
                    'item_utilization': item_utilization,
                    
                    # Performance
                    'hit_rate': hit_rate,
                    'miss_rate': miss_rate,
                    'total_requests': total_requests,
                    'cache_hits': cache_hits,
                    'cache_misses': cache_misses,
                    
                    # Operations
                    'total_puts': metrics.get('total_puts', 0),
                    'evictions': metrics.get('evictions', 0),
                    'expired_entries': metrics.get('expired_entries', 0),
                    'manual_removals': metrics.get('manual_removals', 0),
                    
                    # Timing
                    'last_get_time_ms': metrics.get('last_get_time_ms', 0),
                    'last_put_time_ms': metrics.get('last_put_time_ms', 0),
                    'uptime_seconds': uptime_seconds,
                    
                    # Configuration
                    'default_ttl_seconds': self.default_ttl_seconds,
                    'cleanup_interval_seconds': self.cleanup_interval_seconds,
                    
                    # Health indicators
                    'memory_pressure': memory_utilization > self._memory_pressure_threshold,
                    'emergency_mode': memory_utilization > self._emergency_eviction_threshold,
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            raise CacheError(f"Failed to get stats: {e}")
    
    def evict_expired(self) -> int:
        """
        Evict all expired entries.
        
        Returns:
            Number of entries evicted
        """
        try:
            with self._lock:
                expired_keys = []
                current_time = datetime.now()
                
                for key, entry in self._cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_entry(key)
                
                if expired_keys:
                    self._metrics.increment_counter('expired_entries', len(expired_keys))
                
                return len(expired_keys)
                
        except Exception as e:
            logger.error(f"Failed to evict expired entries: {e}")
            raise CacheError(f"Failed to evict expired: {e}")
    
    def force_eviction(self, target_memory_ratio: float = 0.7) -> int:
        """
        Force eviction to reduce memory usage to target ratio.
        
        Args:
            target_memory_ratio: Target memory usage as ratio of max (0.0-1.0)
            
        Returns:
            Number of entries evicted
        """
        try:
            with self._lock:
                target_memory = self.max_memory_bytes * target_memory_ratio
                evicted_count = 0
                
                # Evict LRU entries until target reached
                while (self._current_memory_usage > target_memory and 
                       len(self._cache) > 0):
                    # Remove least recently used (first in OrderedDict)
                    oldest_key = next(iter(self._cache))
                    self._remove_entry(oldest_key)
                    evicted_count += 1
                
                if evicted_count > 0:
                    self._metrics.increment_counter('forced_evictions', evicted_count)
                    logger.info(f"Force evicted {evicted_count} entries to reach target memory")
                
                return evicted_count
                
        except Exception as e:
            logger.error(f"Failed to force eviction: {e}")
            raise CacheError(f"Failed to force eviction: {e}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update memory usage"""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory_usage -= entry.size_bytes
            del self._cache[key]
    
    def _estimate_entry_size(self, key: str, value: Any) -> int:
        """Estimate total size of cache entry"""
        key_size = sys.getsizeof(key)
        value_size = MemoryEstimator.estimate_size(value)
        entry_overhead = sys.getsizeof(CacheEntry.__new__(CacheEntry))
        
        return key_size + value_size + entry_overhead
    
    def _ensure_capacity(self, entry_size: int) -> None:
        """Ensure cache has capacity for new entry"""
        # Check memory capacity
        while (self._current_memory_usage + entry_size > self.max_memory_bytes and 
               len(self._cache) > 0):
            self._evict_lru()
        
        # Check item count capacity
        while len(self._cache) >= self.max_items:
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self._cache:
            # Remove least recently used (first in OrderedDict)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._metrics.increment_counter('evictions')
    
    def _start_background_cleanup(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_seconds):
                try:
                    expired_count = self.evict_expired()
                    if expired_count > 0:
                        logger.debug(f"Background cleanup: evicted {expired_count} expired entries")
                        
                    # Check for memory pressure and force eviction if needed
                    stats = self.get_stats()
                    if stats['memory_pressure']:
                        forced_count = self.force_eviction(0.7)  # Reduce to 70%
                        if forced_count > 0:
                            logger.info(f"Memory pressure cleanup: evicted {forced_count} entries")
                            
                except Exception as e:
                    logger.error(f"Background cleanup failed: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Background cleanup thread started")
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        try:
            self._stop_cleanup.set()
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
            
            with self._lock:
                self._cache.clear()
                self._current_memory_usage = 0
            
            logger.info("LRUMemoryCache shutdown completed")
            
        except Exception as e:
            logger.error(f"Cache shutdown failed: {e}")
    
    def __del__(self):
        """Cleanup on garbage collection"""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup


class CachePool:
    """Pool of cache instances for different data types"""
    
    def __init__(self, pool_config: Dict[str, Dict[str, Any]]):
        """
        Initialize cache pool with configuration for different cache types.
        
        Args:
            pool_config: Dictionary mapping cache names to configuration
        """
        self.caches: Dict[str, LRUMemoryCache] = {}
        self._lock = threading.Lock()
        
        for cache_name, config in pool_config.items():
            self.caches[cache_name] = LRUMemoryCache(**config)
        
        logger.info(f"CachePool initialized with {len(self.caches)} caches")
    
    def get_cache(self, cache_name: str) -> Optional[LRUMemoryCache]:
        """Get cache instance by name"""
        return self.caches.get(cache_name)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches in pool"""
        pool_stats = {}
        total_memory_mb = 0
        total_items = 0
        
        for cache_name, cache in self.caches.items():
            cache_stats = cache.get_stats()
            pool_stats[cache_name] = cache_stats
            total_memory_mb += cache_stats['current_memory_mb']
            total_items += cache_stats['current_items']
        
        pool_stats['_pool_summary'] = {
            'total_caches': len(self.caches),
            'total_memory_mb': total_memory_mb,
            'total_items': total_items,
        }
        
        return pool_stats
    
    def shutdown_all(self) -> None:
        """Shutdown all caches in pool"""
        for cache in self.caches.values():
            cache.shutdown()
        logger.info("All caches in pool shutdown")


def create_cache_manager(cache_type: str = "lru", **config) -> CacheManager:
    """
    Factory function to create cache manager instances.
    
    Args:
        cache_type: Type of cache to create ("lru", "pool")
        **config: Configuration parameters for the cache
        
    Returns:
        CacheManager instance
    """
    if cache_type == "lru":
        return LRUMemoryCache(
            max_memory_mb=config.get('max_memory_mb', 512),
            max_items=config.get('max_items', 10000),
            default_ttl_seconds=config.get('default_ttl_seconds', 3600),
            cleanup_interval_seconds=config.get('cleanup_interval_seconds', 300)
        )
    elif cache_type == "pool":
        pool_config = config.get('pool_config', {})
        if not pool_config:
            # Default pool configuration
            pool_config = {
                'search_results': {'max_memory_mb': 256, 'max_items': 5000},
                'embeddings': {'max_memory_mb': 128, 'max_items': 2000}, 
                'classifications': {'max_memory_mb': 64, 'max_items': 1000},
            }
        return CachePool(pool_config)
    else:
        raise CacheError(
            f"Unknown cache type: {cache_type}",
            error_code="INVALID_CACHE_TYPE",
            context={"cache_type": cache_type, "available_types": ["lru", "pool"]}
        )