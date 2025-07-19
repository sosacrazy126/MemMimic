"""
High-performance caching layer for memory search operations.

Provides multi-level caching with TTL support, LRU eviction, and performance monitoring.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .interfaces import (
    PerformanceCache, 
    SearchResults, 
    SearchQuery,
    CacheError
)


class LRUMemoryCache(PerformanceCache):
    """
    High-performance in-memory cache with LRU eviction and TTL support.
    
    Features:
    - LRU eviction when cache size limit reached
    - TTL (time-to-live) expiration for entries
    - Performance metrics and monitoring
    - Cache warming and invalidation capabilities
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU memory cache.
        
        Args:
            max_size: Maximum number of entries to cache
            default_ttl: Default TTL in seconds for cached entries
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Cache storage: OrderedDict for LRU behavior
        self._cache = OrderedDict()
        self._expiry_times = OrderedDict()
        
        # Performance metrics
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expiries': 0,
            'total_requests': 0,
            'cache_size': 0,
            'last_cleanup': time.time(),
        }
    
    def get(self, cache_key: str) -> Optional[SearchResults]:
        """
        Retrieve cached search results.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached SearchResults if found and not expired, None otherwise
        """
        try:
            self._metrics['total_requests'] += 1
            
            # Check if key exists
            if cache_key not in self._cache:
                self._metrics['misses'] += 1
                return None
            
            # Check if expired
            if self._is_expired(cache_key):
                self._remove_expired_entry(cache_key)
                self._metrics['misses'] += 1
                self._metrics['expiries'] += 1
                return None
            
            # Move to end (most recently used)
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value
            
            # Also update expiry order
            expiry_time = self._expiry_times.pop(cache_key)
            self._expiry_times[cache_key] = expiry_time
            
            self._metrics['hits'] += 1
            return value
            
        except Exception as e:
            raise CacheError(
                f"Failed to retrieve from cache: {e}",
                error_code="CACHE_GET_ERROR",
                context={"cache_key": cache_key}
            )
    
    def set(self, cache_key: str, results: SearchResults, ttl: int = None) -> None:
        """
        Store search results in cache.
        
        Args:
            cache_key: Key to store under
            results: SearchResults to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Calculate expiry time
            expiry_time = time.time() + ttl
            
            # Remove existing entry if present
            if cache_key in self._cache:
                self._cache.pop(cache_key)
                self._expiry_times.pop(cache_key)
            
            # Evict oldest entries if cache is full
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[cache_key] = results
            self._expiry_times[cache_key] = expiry_time
            
            # Update metrics
            self._metrics['cache_size'] = len(self._cache)
            
            # Periodic cleanup of expired entries
            if time.time() - self._metrics['last_cleanup'] > 300:  # 5 minutes
                self._cleanup_expired()
                
        except Exception as e:
            raise CacheError(
                f"Failed to store in cache: {e}",
                error_code="CACHE_SET_ERROR",
                context={"cache_key": cache_key, "ttl": ttl}
            )
    
    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Number of entries invalidated
        """
        try:
            invalidated_count = 0
            keys_to_remove = []
            
            for cache_key in self._cache.keys():
                if pattern in cache_key:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._expiry_times.pop(key, None)
                invalidated_count += 1
            
            self._metrics['cache_size'] = len(self._cache)
            return invalidated_count
            
        except Exception as e:
            raise CacheError(
                f"Failed to invalidate cache entries: {e}",
                error_code="CACHE_INVALIDATE_ERROR",
                context={"pattern": pattern}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache metrics and performance data
        """
        try:
            total_requests = self._metrics['total_requests']
            hit_rate = (self._metrics['hits'] / total_requests) if total_requests > 0 else 0.0
            miss_rate = (self._metrics['misses'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size,
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'total_hits': self._metrics['hits'],
                'total_misses': self._metrics['misses'],
                'total_requests': total_requests,
                'evictions': self._metrics['evictions'],
                'expiries': self._metrics['expiries'],
                'default_ttl': self.default_ttl,
                'last_cleanup': datetime.fromtimestamp(self._metrics['last_cleanup']).isoformat(),
            }
            
        except Exception as e:
            raise CacheError(
                f"Failed to get cache stats: {e}",
                error_code="CACHE_STATS_ERROR"
            )
    
    def warm_cache(self, queries: list[str], results: list[SearchResults]) -> int:
        """
        Warm cache with predefined query-result pairs.
        
        Args:
            queries: List of query strings
            results: List of corresponding SearchResults
            
        Returns:
            Number of entries successfully cached
        """
        try:
            warmed_count = 0
            
            for query, result in zip(queries, results):
                cache_key = self._generate_cache_key(query)
                self.set(cache_key, result)
                warmed_count += 1
            
            return warmed_count
            
        except Exception as e:
            raise CacheError(
                f"Failed to warm cache: {e}",
                error_code="CACHE_WARM_ERROR",
                context={"query_count": len(queries)}
            )
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._cache.clear()
            self._expiry_times.clear()
            self._metrics['cache_size'] = 0
            
        except Exception as e:
            raise CacheError(
                f"Failed to clear cache: {e}",
                error_code="CACHE_CLEAR_ERROR"
            )
    
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired."""
        expiry_time = self._expiry_times.get(cache_key)
        if expiry_time is None:
            return True
        return time.time() > expiry_time
    
    def _remove_expired_entry(self, cache_key: str) -> None:
        """Remove expired entry from cache."""
        self._cache.pop(cache_key, None)
        self._expiry_times.pop(cache_key, None)
        self._metrics['cache_size'] = len(self._cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # Remove oldest entry (first in OrderedDict)
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self._expiry_times.pop(oldest_key, None)
            self._metrics['evictions'] += 1
            self._metrics['cache_size'] = len(self._cache)
    
    def _cleanup_expired(self) -> None:
        """Clean up all expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, expiry_time in self._expiry_times.items():
            if current_time > expiry_time:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_expired_entry(key)
            self._metrics['expiries'] += 1
        
        self._metrics['last_cleanup'] = current_time
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key from query string."""
        # Create deterministic hash from query
        return hashlib.md5(query.encode('utf-8')).hexdigest()


def generate_search_cache_key(query: SearchQuery) -> str:
    """
    Generate a cache key from a SearchQuery object.
    
    Args:
        query: SearchQuery to generate key for
        
    Returns:
        Deterministic cache key string
    """
    try:
        # Create normalized query representation
        query_dict = {
            'text': query.text.lower().strip(),
            'limit': query.limit,
            'filters': sorted(query.filters.items()) if query.filters else [],
            'include_metadata': query.include_metadata,
            'min_confidence': query.min_confidence,
            'search_type': query.search_type.value,
        }
        
        # Generate deterministic hash
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode('utf-8')).hexdigest()
        
    except Exception as e:
        raise CacheError(
            f"Failed to generate cache key: {e}",
            error_code="CACHE_KEY_ERROR",
            context={"query_text": query.text[:100]}  # Truncate for privacy
        )


def create_performance_cache(cache_type: str = "memory", **config) -> PerformanceCache:
    """
    Factory function to create performance cache instances.
    
    Args:
        cache_type: Type of cache to create ("memory", "redis", "memcached")
        **config: Configuration parameters for the cache
        
    Returns:
        PerformanceCache instance
    """
    if cache_type == "memory":
        return LRUMemoryCache(
            max_size=config.get('max_size', 1000),
            default_ttl=config.get('default_ttl', 3600)
        )
    elif cache_type == "redis":
        # Future: Redis cache implementation
        raise NotImplementedError("Redis cache not yet implemented")
    elif cache_type == "memcached":
        # Future: Memcached implementation
        raise NotImplementedError("Memcached cache not yet implemented")
    else:
        raise CacheError(
            f"Unknown cache type: {cache_type}",
            error_code="INVALID_CACHE_TYPE",
            context={"cache_type": cache_type, "available_types": ["memory", "redis", "memcached"]}
        )