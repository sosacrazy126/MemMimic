#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic utility modules for caching, performance optimization, and helper functions.
"""

from .caching import (
    cached_cxd_operation,
    cached_memory_operation,
    cached_embedding_operation,
    lru_cached,
    clear_all_caches,
    get_cache_statistics,
    warm_up_cache
)

__all__ = [
    'cached_cxd_operation',
    'cached_memory_operation', 
    'cached_embedding_operation',
    'lru_cached',
    'clear_all_caches',
    'get_cache_statistics',
    'warm_up_cache'
]

