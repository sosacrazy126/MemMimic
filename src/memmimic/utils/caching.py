#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caching utilities for MemMimic performance optimization.

Provides intelligent caching decorators and cache management for expensive operations.
"""

import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class MemMimicCache:
    """
    Advanced cache manager for MemMimic operations.
    
    Provides TTL-based caching, size limits, and hit/miss metrics.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        self._stats["total_requests"] += 1
        
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        item = self._cache[key]
        
        # Check expiration
        if time.time() > item["expires_at"]:
            del self._cache[key]
            self._stats["misses"] += 1
            return None
        
        self._stats["hits"] += 1
        item["access_count"] += 1
        item["last_accessed"] = time.time()
        return item["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()
        
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 1
        }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU item
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["last_accessed"]
        )
        
        del self._cache[lru_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache item: {lru_key}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["total_requests"]
        hit_rate = (
            self._stats["hits"] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size
        }


# Global cache instances
_cxd_cache = MemMimicCache(max_size=500, default_ttl=1800)  # 30 minutes
_memory_cache = MemMimicCache(max_size=1000, default_ttl=3600)  # 1 hour
_embedding_cache = MemMimicCache(max_size=2000, default_ttl=7200)  # 2 hours


def _generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    try:
        # Create a deterministic key from arguments
        key_data = {
            "args": [str(arg) for arg in args],
            "kwargs": {str(k): str(v) for k, v in kwargs.items()}
        }
        
        # Create hash of the key data
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    except Exception as e:
        logger.warning(f"Failed to generate cache key: {e}")
        return f"fallback_{time.time()}_{id(args)}"


def cached_cxd_operation(ttl: int = 1800) -> Callable[[F], F]:
    """
    Cache decorator for expensive CXD operations.
    
    Args:
        ttl: Time to live in seconds (default: 30 minutes)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = _cxd_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            _cxd_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cached_memory_operation(ttl: int = 3600) -> Callable[[F], F]:
    """
    Cache decorator for memory operations.
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = _memory_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Memory cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"Memory cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            _memory_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cached_embedding_operation(ttl: int = 7200) -> Callable[[F], F]:
    """
    Cache decorator for embedding operations.
    
    Args:
        ttl: Time to live in seconds (default: 2 hours)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{_generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = _embedding_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Embedding cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"Embedding cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            _embedding_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Standard LRU cache for simple operations
def lru_cached(maxsize: int = 128) -> Callable[[F], F]:
    """
    Simple LRU cache decorator.
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorated function with LRU caching
    """
    return lru_cache(maxsize=maxsize)


def clear_all_caches() -> None:
    """Clear all MemMimic caches."""
    _cxd_cache.clear()
    _memory_cache.clear()
    _embedding_cache.clear()
    logger.info("All MemMimic caches cleared")


def get_cache_statistics() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return {
        "cxd_cache": _cxd_cache.get_stats(),
        "memory_cache": _memory_cache.get_stats(),
        "embedding_cache": _embedding_cache.get_stats(),
        "total_cached_operations": (
            len(_cxd_cache._cache) + 
            len(_memory_cache._cache) + 
            len(_embedding_cache._cache)
        )
    }


def warm_up_cache(operations: Optional[list] = None) -> None:
    """
    Warm up caches with common operations.
    
    Args:
        operations: List of operations to warm up (optional)
    """
    logger.info("Starting cache warm-up...")
    
    # This would be implemented with actual warm-up logic
    # for now, just log that warm-up is available
    logger.info("Cache warm-up completed")