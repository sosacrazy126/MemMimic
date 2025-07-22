# MemMimic Caching System Documentation

## Overview

MemMimic's Phase 2 caching system provides a multi-layered, intelligent caching architecture designed to optimize performance across all system operations. The system combines specialized cache managers, TTL-based expiration, memory pressure management, and operation-specific caching strategies.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   MemMimic Caching Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                      Application Layer                          │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   Search    │ │   Memory    │ │     CXD     │ │  Embedding  │ │
│ │ Operations  │ │ Operations  │ │ Operations  │ │ Operations  │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│        │               │               │               │        │
│        ▼               ▼               ▼               ▼        │
├─────────────────────────────────────────────────────────────────┤
│                     Decorator Layer                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │@lru_cached  │ │@cached_     │ │@cached_cxd_ │ │@cached_     │ │
│ │             │ │memory_op    │ │operation    │ │embedding_op │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│        │               │               │               │        │
│        ▼               ▼               ▼               ▼        │
├─────────────────────────────────────────────────────────────────┤
│                      Cache Layer                                │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   Python    │ │MemMimicCache│ │MemMimicCache│ │MemMimicCache│ │
│ │   LRUCache  │ │Memory: 1000 │ │CXD: 500     │ │Embed: 2000  │ │
│ │   128 items │ │TTL: 1hr     │ │TTL: 30min   │ │TTL: 2hr     │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│        │               │               │               │        │
│        ▼               ▼               ▼               ▼        │
├─────────────────────────────────────────────────────────────────┤
│                   Active Memory Layer                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                LRUMemoryCache System                        │ │
│ │  • Multi-pool architecture with specialized configurations  │ │
│ │  • Memory pressure detection and automatic eviction        │ │
│ │  • TTL-based expiration with background cleanup            │ │
│ │  • Thread-safe concurrent access                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Caching Decorators

#### 1.1 LRU Cache (`@lru_cached`)

**Purpose**: Simple, fast caching for lightweight operations with size-based eviction.

**Usage**:
```python
from memmimic.utils.caching import lru_cached

@lru_cached(maxsize=512)
def get_synonyms(word: str) -> List[str]:
    """Cached synonym lookup with LRU eviction."""
    return expensive_synonym_lookup(word)

# Clear specific cache
get_synonyms.cache_clear()

# Get cache info
info = get_synonyms.cache_info()
print(f"Hits: {info.hits}, Misses: {info.misses}")
```

**Characteristics**:
- **Maximum Size**: Configurable (default: 128)
- **Eviction Policy**: Least Recently Used (LRU)
- **Thread Safety**: Yes (built-in Python LRU)
- **TTL Support**: No
- **Memory Overhead**: Minimal
- **Use Cases**: Quick lookups, small data sets, simple operations

#### 1.2 Memory Operation Cache (`@cached_memory_operation`)

**Purpose**: TTL-based caching for memory search and retrieval operations.

**Usage**:
```python
from memmimic.utils.caching import cached_memory_operation

@cached_memory_operation(ttl=1800)  # 30 minutes
def search_memories(query: str, limit: int = 10) -> List[Dict]:
    """Cached memory search with TTL expiration."""
    return expensive_memory_search(query, limit)

# Function automatically uses cache
results = search_memories("project architecture")  # Cache miss - executes
results = search_memories("project architecture")  # Cache hit - instant
```

**Characteristics**:
- **Cache Size**: 1000 items maximum
- **TTL**: 1 hour (configurable per call)
- **Eviction**: LRU when size limit reached
- **Key Generation**: Automatic based on function arguments
- **Use Cases**: Memory searches, content retrieval, query results

#### 1.3 CXD Operation Cache (`@cached_cxd_operation`)

**Purpose**: Caching for expensive CXD classification operations.

**Usage**:
```python
from memmimic.utils.caching import cached_cxd_operation

@cached_cxd_operation(ttl=1800)  # 30 minutes
def classify_content(content: str) -> CXDResult:
    """Cached CXD classification with shorter TTL."""
    return expensive_cxd_classification(content)

# Automatic cache key generation handles complex arguments
result = classify_content("user query about memory patterns")
```

**Characteristics**:
- **Cache Size**: 500 items maximum
- **TTL**: 30 minutes (optimized for changing classifications)
- **Eviction**: LRU with access tracking
- **Key Generation**: Content-aware hashing
- **Use Cases**: CXD classifications, semantic analysis, pattern recognition

#### 1.4 Embedding Operation Cache (`@cached_embedding_operation`)

**Purpose**: Long-term caching for expensive embedding generation operations.

**Usage**:
```python
from memmimic.utils.caching import cached_embedding_operation

@cached_embedding_operation(ttl=7200)  # 2 hours
def get_text_embedding(text: str) -> np.ndarray:
    """Cached embedding generation with extended TTL."""
    return expensive_embedding_model(text)

# Embeddings are expensive to compute, cache for longer
embedding = get_text_embedding("complex technical documentation")
```

**Characteristics**:
- **Cache Size**: 2000 items maximum
- **TTL**: 2 hours (embeddings are stable)
- **Eviction**: LRU with access counting
- **Key Generation**: Text content hashing
- **Use Cases**: Text embeddings, vector similarity, semantic search

### 2. Active Memory Cache Manager

#### 2.1 LRUMemoryCache

**Location**: `src/memmimic/memory/active/cache_manager.py`

**Architecture**:
```python
from memmimic.memory.active.cache_manager import LRUMemoryCache, create_cache_manager

# Single cache instance
cache = LRUMemoryCache(
    max_memory_mb=512,
    max_items=10000,
    default_ttl_seconds=3600,
    cleanup_interval_seconds=300
)

# Basic operations
cache.put("search_results_query123", results, ttl_seconds=1800)
cached_results = cache.get("search_results_query123")
removed = cache.remove("search_results_query123")

# Maintenance operations
expired_count = cache.evict_expired()
forced_count = cache.force_eviction(target_memory_ratio=0.7)
cache.clear()

# Performance monitoring
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Memory usage: {stats['current_memory_mb']:.1f}MB")
print(f"Memory pressure: {stats['memory_pressure']}")
```

**Key Features**:

**Memory Management**:
- **Memory Limit**: Configurable in MB (default: 512MB)
- **Memory Tracking**: Real-time usage monitoring
- **Memory Pressure**: 80% warning, 95% emergency eviction
- **Memory Estimation**: Intelligent size calculation for complex objects

**TTL Management**:
- **Per-item TTL**: Individual expiration times
- **Background Cleanup**: Automatic expired item removal
- **TTL Inheritance**: Default TTL with per-item override

**Performance Monitoring**:
- **Hit/Miss Tracking**: Detailed cache performance metrics
- **Response Time Monitoring**: Operation timing with percentiles
- **Memory Utilization**: Real-time memory and item usage
- **Health Indicators**: Memory pressure and performance alerts

#### 2.2 Cache Pool Management

**Purpose**: Multiple specialized caches for different data types and access patterns.

**Usage**:
```python
from memmimic.memory.active.cache_manager import create_cache_manager

# Create cache pool with specialized configurations
pool_config = {
    'search_results': {
        'max_memory_mb': 256, 
        'max_items': 5000,
        'default_ttl_seconds': 1800,
        'cleanup_interval_seconds': 300
    },
    'embeddings': {
        'max_memory_mb': 128, 
        'max_items': 2000,
        'default_ttl_seconds': 7200,
        'cleanup_interval_seconds': 600
    },
    'classifications': {
        'max_memory_mb': 64, 
        'max_items': 1000,
        'default_ttl_seconds': 3600,
        'cleanup_interval_seconds': 450
    },
    'query_expansions': {
        'max_memory_mb': 32,
        'max_items': 500,
        'default_ttl_seconds': 2700,
        'cleanup_interval_seconds': 300
    }
}

cache_pool = create_cache_manager(cache_type="pool", pool_config=pool_config)

# Access specific caches
search_cache = cache_pool.get_cache("search_results")
embedding_cache = cache_pool.get_cache("embeddings")
cxd_cache = cache_pool.get_cache("classifications")

# Use caches with specialized configurations
search_cache.put("complex_query_hash", search_results, ttl_seconds=1800)
embedding_cache.put("text_content_hash", embedding_vector, ttl_seconds=7200)
cxd_cache.put("content_classification", cxd_result, ttl_seconds=3600)

# Pool-wide statistics and management
pool_stats = cache_pool.get_pool_stats()
print(f"Total memory usage: {pool_stats['_pool_summary']['total_memory_mb']:.1f}MB")
print(f"Total cached items: {pool_stats['_pool_summary']['total_items']}")

# Shutdown all caches
cache_pool.shutdown_all()
```

## Performance Optimization

### 3. Cache Performance Strategies

#### 3.1 Hit Rate Optimization

**Strategy**: Predictive preloading based on access patterns.

```python
# Example of pattern-based cache warming
def warm_cache_for_common_queries():
    """Pre-populate cache with frequently accessed data."""
    common_queries = [
        "project architecture",
        "consciousness integration", 
        "memory patterns",
        "performance optimization"
    ]
    
    for query in common_queries:
        # Pre-compute and cache common searches
        results = search_memories_hybrid(query, limit=10)
        cache.put(f"search_{query}", results, ttl_seconds=3600)
```

#### 3.2 Memory Pressure Management

**Thresholds**:
- **80% Memory Usage**: Background cleanup initiated
- **95% Memory Usage**: Emergency eviction triggered
- **100% Memory Usage**: All new cache operations blocked

**Strategies**:
```python
# Automatic memory pressure handling
def handle_memory_pressure(cache_manager):
    stats = cache_manager.get_stats()
    
    if stats['memory_utilization'] > 0.95:
        # Emergency eviction to 70% capacity
        evicted = cache_manager.force_eviction(target_memory_ratio=0.7)
        logger.warning(f"Emergency eviction: {evicted} items removed")
        
    elif stats['memory_utilization'] > 0.8:
        # Background cleanup of expired items
        expired = cache_manager.evict_expired()
        logger.info(f"Background cleanup: {expired} expired items removed")
```

#### 3.3 Cache Key Optimization

**Key Generation Strategy**:
```python
def generate_optimized_cache_key(*args, **kwargs) -> str:
    """Generate optimized cache key with content awareness."""
    
    # Handle different argument types efficiently
    key_components = []
    
    for arg in args:
        if isinstance(arg, str):
            # Use hash for long strings to avoid key bloat
            if len(arg) > 100:
                key_components.append(hashlib.md5(arg.encode()).hexdigest()[:16])
            else:
                key_components.append(arg)
        elif isinstance(arg, (int, float, bool)):
            key_components.append(str(arg))
        else:
            # Fallback to string representation hash
            key_components.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
    
    # Add kwargs with sorted keys for consistency
    if kwargs:
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_components.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:16])
    
    return "_".join(key_components)
```

## Configuration and Tuning

### 4. Configuration Management

#### 4.1 Cache Configuration

**Configuration File**: `config/performance_config.yaml`

```yaml
caching:
  # Global cache settings
  enabled: true
  global_ttl_multiplier: 1.0
  memory_pressure_threshold: 0.8
  emergency_threshold: 0.95
  
  # Decorator cache settings  
  decorators:
    lru_cache:
      default_maxsize: 128
      
    memory_operations:
      max_size: 1000
      default_ttl: 3600
      
    cxd_operations:
      max_size: 500
      default_ttl: 1800
      
    embedding_operations:
      max_size: 2000
      default_ttl: 7200
  
  # Active memory cache settings
  active_memory:
    search_results:
      max_memory_mb: 256
      max_items: 5000
      default_ttl_seconds: 1800
      cleanup_interval_seconds: 300
      
    embeddings:
      max_memory_mb: 128
      max_items: 2000
      default_ttl_seconds: 7200
      cleanup_interval_seconds: 600
      
    classifications:
      max_memory_mb: 64
      max_items: 1000
      default_ttl_seconds: 3600
      cleanup_interval_seconds: 450
      
    query_expansions:
      max_memory_mb: 32
      max_items: 500
      default_ttl_seconds: 2700
      cleanup_interval_seconds: 300

  # Performance monitoring
  monitoring:
    metrics_collection: true
    performance_alerts: true
    hit_rate_threshold: 0.7
    response_time_threshold_ms: 100
```

#### 4.2 Runtime Tuning

**Dynamic Configuration Adjustment**:
```python
from memmimic.utils.caching import get_cache_statistics, clear_all_caches

# Monitor cache performance
def monitor_and_tune_caches():
    stats = get_cache_statistics()
    
    # Analyze cache effectiveness
    for cache_name, cache_stats in stats.items():
        if cache_name.startswith('_'):
            continue
            
        hit_rate = cache_stats['hit_rate']
        cache_size = cache_stats['cache_size']
        max_size = cache_stats['max_size']
        
        print(f"{cache_name}:")
        print(f"  Hit rate: {hit_rate:.1%}")
        print(f"  Utilization: {cache_size}/{max_size} ({cache_size/max_size:.1%})")
        
        # Suggest tuning adjustments
        if hit_rate < 0.5:
            print(f"  ⚠️  Low hit rate - consider increasing TTL or cache size")
        elif cache_size / max_size > 0.9:
            print(f"  ⚠️  High utilization - consider increasing max_size")
        else:
            print(f"  ✅ Cache performing well")

# Emergency cache management
def emergency_cache_clear():
    """Clear all caches in case of memory pressure."""
    clear_all_caches()
    logger.warning("All caches cleared due to memory pressure")
```

## Performance Metrics and Monitoring

### 5. Cache Performance Analytics

#### 5.1 Comprehensive Metrics

**Available Metrics**:
```python
def get_comprehensive_cache_metrics():
    """Get detailed cache performance metrics."""
    
    stats = get_cache_statistics()
    
    # Aggregate metrics across all caches
    total_hits = sum(cache['hits'] for cache in stats.values() if 'hits' in cache)
    total_misses = sum(cache['misses'] for cache in stats.values() if 'misses' in cache)
    total_requests = total_hits + total_misses
    
    overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
    
    return {
        'overall_performance': {
            'hit_rate': overall_hit_rate,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'total_misses': total_misses
        },
        'cache_breakdown': stats,
        'memory_utilization': {
            cache_name: cache_stats.get('current_memory_mb', 0) 
            for cache_name, cache_stats in stats.items()
            if hasattr(cache_stats, 'get')
        }
    }
```

#### 5.2 Performance Benchmarks

**Expected Performance Characteristics**:

| Cache Type | Hit Rate | Response Time | Memory Usage | TTL |
|------------|----------|---------------|--------------|-----|
| LRU Cache | 85-95% | <1ms | <50MB | N/A |
| Memory Operations | 70-85% | <5ms | <256MB | 1 hour |
| CXD Operations | 60-80% | <10ms | <64MB | 30 min |
| Embedding Operations | 80-95% | <15ms | <128MB | 2 hours |
| Active Memory Cache | 85-95% | <3ms | <512MB | Variable |

#### 5.3 Monitoring and Alerts

**Performance Alert Conditions**:
```python
def check_cache_health():
    """Monitor cache health and trigger alerts."""
    
    metrics = get_comprehensive_cache_metrics()
    alerts = []
    
    # Check overall hit rate
    if metrics['overall_performance']['hit_rate'] < 0.6:
        alerts.append({
            'level': 'WARNING',
            'message': f"Low overall cache hit rate: {metrics['overall_performance']['hit_rate']:.1%}"
        })
    
    # Check individual cache performance
    for cache_name, stats in metrics['cache_breakdown'].items():
        if 'hit_rate' in stats:
            if stats['hit_rate'] < 0.5:
                alerts.append({
                    'level': 'WARNING',
                    'message': f"{cache_name} hit rate below 50%: {stats['hit_rate']:.1%}"
                })
    
    # Check memory pressure
    total_memory = sum(
        usage for usage in metrics['memory_utilization'].values()
    )
    
    if total_memory > 800:  # 800MB threshold
        alerts.append({
            'level': 'ERROR',
            'message': f"High cache memory usage: {total_memory:.1f}MB"
        })
    
    return alerts
```

## Best Practices and Usage Guidelines

### 6. Implementation Best Practices

#### 6.1 Decorator Selection Guide

**Choose the Right Decorator**:

```python
# ✅ Use @lru_cached for simple, frequently accessed functions
@lru_cached(maxsize=256)
def get_stop_words(language: str) -> Set[str]:
    return load_stop_words_from_file(language)

# ✅ Use @cached_memory_operation for database queries
@cached_memory_operation(ttl=1800)
def search_memories_by_type(memory_type: str) -> List[Memory]:
    return database.search_by_type(memory_type)

# ✅ Use @cached_cxd_operation for classification tasks
@cached_cxd_operation(ttl=1800)
def classify_user_query(query: str) -> CXDFunction:
    return expensive_classification_model(query)

# ✅ Use @cached_embedding_operation for ML model operations
@cached_embedding_operation(ttl=7200)
def encode_text_to_vector(text: str) -> np.ndarray:
    return transformer_model.encode(text)
```

#### 6.2 Cache Key Design

**Effective Cache Key Strategies**:

```python
# ✅ Good: Include all relevant parameters
@cached_memory_operation(ttl=3600)
def search_with_filters(query: str, memory_type: str, limit: int, 
                       function_filter: str) -> List[Dict]:
    # Cache key automatically includes all parameters
    return perform_filtered_search(query, memory_type, limit, function_filter)

# ❌ Avoid: Functions with mutable parameters
def search_with_config(query: str, config: Dict) -> List[Dict]:
    # Dict parameter makes cache key generation unreliable
    pass

# ✅ Better: Extract relevant config values
@cached_memory_operation(ttl=3600)
def search_with_config(query: str, limit: int, threshold: float) -> List[Dict]:
    # Explicit parameters create reliable cache keys
    return perform_search(query, {'limit': limit, 'threshold': threshold})
```

#### 6.3 TTL Configuration Guidelines

**TTL Selection Strategy**:

```python
# Short TTL (5-30 minutes): Frequently changing data
@cached_cxd_operation(ttl=900)  # 15 minutes
def classify_user_intent(query: str) -> str:
    # User patterns change quickly
    return intent_classifier(query)

# Medium TTL (30 minutes - 2 hours): Semi-stable data  
@cached_memory_operation(ttl=3600)  # 1 hour
def search_recent_memories(days: int = 7) -> List[Memory]:
    # Memory content changes moderately
    return database.search_recent(days)

# Long TTL (2+ hours): Stable reference data
@cached_embedding_operation(ttl=7200)  # 2 hours
def get_concept_embedding(concept: str) -> np.ndarray:
    # Conceptual embeddings are stable
    return embedding_model.encode(concept)

# Very Long TTL: Static reference data
@lru_cached(maxsize=1000)  # No TTL, size-based only
def get_language_model_vocab() -> Dict[str, int]:
    # Vocabulary doesn't change during runtime
    return load_static_vocabulary()
```

#### 6.4 Error Handling and Cache Failures

**Robust Cache Error Handling**:

```python
@cached_memory_operation(ttl=3600)
def robust_memory_search(query: str) -> List[Dict]:
    """Memory search with cache failure handling."""
    try:
        return perform_memory_search(query)
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        # Cache will not store failed results
        raise

def search_with_cache_fallback(query: str) -> List[Dict]:
    """Search with graceful cache degradation."""
    try:
        # Try cached search first
        return cached_search_function(query)
    except Exception as e:
        logger.warning(f"Cached search failed: {e}, falling back to direct search")
        # Fallback to direct search without caching
        return direct_search_function(query)
```

## Troubleshooting and Debugging

### 7. Common Issues and Solutions

#### 7.1 Low Hit Rates

**Symptoms**: Cache hit rates below 60%

**Causes and Solutions**:

```python
# Issue: Cache keys too specific
# ❌ Problem
@cached_memory_operation(ttl=3600)
def search_with_timestamp(query: str) -> List[Dict]:
    current_time = time.time()  # Changes every call!
    return search_memories(query, timestamp=current_time)

# ✅ Solution: Remove volatile parameters
@cached_memory_operation(ttl=3600)  
def search_memories_cached(query: str) -> List[Dict]:
    return search_memories(query)

# Issue: TTL too short
# ✅ Solution: Increase TTL for stable operations
@cached_embedding_operation(ttl=14400)  # Increased to 4 hours
def generate_stable_embedding(text: str) -> np.ndarray:
    return expensive_embedding_model(text)
```

#### 7.2 Memory Pressure Issues

**Symptoms**: Frequent cache evictions, high memory usage

**Solutions**:

```python
# Monitor and adjust cache sizes
def optimize_cache_memory():
    stats = get_cache_statistics()
    
    for cache_name, cache_stats in stats.items():
        memory_usage = cache_stats.get('current_memory_mb', 0)
        hit_rate = cache_stats.get('hit_rate', 0)
        
        if memory_usage > 200 and hit_rate < 0.7:
            logger.warning(f"{cache_name}: High memory ({memory_usage}MB) with low hit rate ({hit_rate:.1%})")
            # Consider reducing cache size or increasing TTL

# Implement memory-aware caching
def memory_aware_cache_operation():
    current_memory = get_current_memory_usage()
    
    if current_memory > 0.8:  # 80% memory pressure
        # Use shorter TTL to reduce memory footprint
        ttl = 1800  # 30 minutes instead of 1 hour
    else:
        ttl = 3600  # Normal 1 hour TTL
    
    return ttl
```

#### 7.3 Performance Debugging

**Cache Performance Analysis**:

```python
def debug_cache_performance():
    """Comprehensive cache performance debugging."""
    
    stats = get_cache_statistics()
    
    print("=== CACHE PERFORMANCE ANALYSIS ===")
    
    for cache_name, cache_stats in stats.items():
        if cache_name.startswith('_'):
            continue
            
        print(f"\n{cache_name.upper()}:")
        print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"  Total Requests: {cache_stats.get('total_requests', 0)}")
        print(f"  Cache Size: {cache_stats.get('cache_size', 0)}")
        print(f"  Memory Usage: {cache_stats.get('current_memory_mb', 0):.1f}MB")
        
        # Performance recommendations
        hit_rate = cache_stats.get('hit_rate', 0)
        if hit_rate < 0.5:
            print(f"  ⚠️  RECOMMENDATION: Low hit rate - check cache key design")
        elif hit_rate > 0.9:
            print(f"  ✅ EXCELLENT: High hit rate indicates good caching")
        else:
            print(f"  ✅ GOOD: Acceptable hit rate")
            
        size_ratio = cache_stats.get('cache_size', 0) / cache_stats.get('max_size', 1)
        if size_ratio > 0.9:
            print(f"  ⚠️  RECOMMENDATION: Near capacity - consider increasing max_size")
```

This comprehensive caching system documentation provides complete coverage of MemMimic's multi-layered caching architecture, enabling developers to effectively utilize and optimize the caching system for maximum performance.