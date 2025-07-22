# MemMimic Phase 2 Integration Guide

## Overview

This guide provides comprehensive instructions for integrating with MemMimic's Phase 2 modular architecture, including the new caching system, Active Memory Management System (AMMS), and modular search components.

## Quick Start

### Basic Integration

```python
from memmimic import create_memmimic
from memmimic.memory.search.hybrid_search import HybridSearchEngine
from memmimic.memory.active.cache_manager import create_cache_manager

# Initialize MemMimic with enhanced features
api = create_memmimic("memmimic.db")

# Initialize hybrid search engine
search_engine = HybridSearchEngine("memmimic.db")

# Initialize cache manager
cache_manager = create_cache_manager(
    cache_type="lru",
    max_memory_mb=256,
    max_items=5000
)

# Perform cached hybrid search
results = search_engine.search_memories_hybrid(
    query="consciousness integration patterns",
    limit=10,
    function_filter="CONTEXT"
)

print(f"Found {len(results['results'])} results in {results['metadata']['search_time_ms']:.2f}ms")
```

### MCP Integration

```javascript
// Use enhanced MCP tools with Phase 2 features
const results = await client.call('recall_cxd', {
    query: 'project architecture decisions',
    function_filter: 'CONTEXT',
    limit: 10,
    db_name: 'memmimic'
});

// Access performance metadata
console.log(`Search time: ${results.metadata.search_time_ms}ms`);
console.log(`Cache hit rate: ${results.metadata.cache_hit_rate}`);
```

## Component Integration

### 1. Hybrid Search Engine Integration

#### Basic Usage

```python
from memmimic.memory.search.hybrid_search import HybridSearchEngine

# Initialize with custom weights
engine = HybridSearchEngine("memmimic.db")

# Configure search parameters
results = engine.search_memories_hybrid(
    query="memory management strategies",
    limit=15,
    function_filter="ALL",
    semantic_weight=0.6,        # Favor semantic search
    wordnet_weight=0.4,         # Moderate WordNet expansion
    convergence_bonus=0.15      # Bonus for results found by both methods
)

# Access detailed results
for result in results['results']:
    print(f"Rank {result['rank']}: {result['content'][:100]}...")
    print(f"  Combined Score: {result['combined_score']:.3f}")
    print(f"  Method: {result['search_method']}")
    print(f"  Convergence: {result['convergence']}")
```

#### Advanced Configuration

```python
# Custom search engine with performance tuning
class OptimizedSearchEngine(HybridSearchEngine):
    def __init__(self, db_name: str):
        super().__init__(db_name)
        
        # Configure semantic processor
        self.semantic_processor.similarity_threshold = 0.15
        
        # Configure WordNet expander  
        self.wordnet_expander.max_synonyms_per_word = 5
        
        # Configure result combiner
        self.result_combiner.combination_strategies["custom"] = self._custom_strategy
    
    def _custom_strategy(self, group_data, semantic_weight, wordnet_weight, convergence_bonus):
        """Custom scoring strategy for specific use case."""
        semantic_result = group_data["semantic"]
        wordnet_result = group_data["wordnet"]
        
        # Implement custom scoring logic
        # ... custom implementation
        
        return combined_result

# Use optimized engine
optimized_engine = OptimizedSearchEngine("memmimic.db")
```

### 2. Cache Manager Integration

#### Single Cache Integration

```python
from memmimic.memory.active.cache_manager import LRUMemoryCache

# Create cache for specific use case
search_cache = LRUMemoryCache(
    max_memory_mb=128,
    max_items=2000,
    default_ttl_seconds=1800,    # 30 minutes
    cleanup_interval_seconds=300  # 5 minute cleanup
)

# Cache search results
def cached_search(query: str, params: dict) -> list:
    cache_key = f"search_{hash(query)}_{hash(str(params))}"
    
    # Check cache first
    cached_result = search_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Perform search if not cached
    results = perform_expensive_search(query, params)
    
    # Cache results with custom TTL
    search_cache.put(cache_key, results, ttl_seconds=1800)
    
    return results

# Monitor cache performance
stats = search_cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Memory usage: {stats['current_memory_mb']:.1f}MB")
```

#### Cache Pool Integration

```python
from memmimic.memory.active.cache_manager import create_cache_manager

# Define specialized cache configurations
cache_config = {
    'user_queries': {
        'max_memory_mb': 64,
        'max_items': 1000, 
        'default_ttl_seconds': 1200,  # 20 minutes (user sessions)
        'cleanup_interval_seconds': 180
    },
    'system_data': {
        'max_memory_mb': 128,
        'max_items': 2000,
        'default_ttl_seconds': 7200,  # 2 hours (stable data)
        'cleanup_interval_seconds': 600
    },
    'temp_results': {
        'max_memory_mb': 32,
        'max_items': 500,
        'default_ttl_seconds': 600,   # 10 minutes (temporary)
        'cleanup_interval_seconds': 120
    }
}

# Create cache pool
cache_pool = create_cache_manager(cache_type="pool", pool_config=cache_config)

# Use specialized caches
user_cache = cache_pool.get_cache("user_queries")
system_cache = cache_pool.get_cache("system_data") 
temp_cache = cache_pool.get_cache("temp_results")

# Cache with appropriate cache type
user_cache.put(f"user_{user_id}_query", user_query_results, ttl_seconds=1200)
system_cache.put("config_data", system_configuration, ttl_seconds=7200)
temp_cache.put("temp_calculation", intermediate_results, ttl_seconds=600)
```

### 3. Caching Decorator Integration

#### Function-Level Caching

```python
from memmimic.utils.caching import (
    lru_cached, 
    cached_memory_operation, 
    cached_cxd_operation,
    cached_embedding_operation
)

# Simple LRU caching for lightweight operations
@lru_cached(maxsize=256)
def get_stop_words(language: str) -> set:
    """Get stop words for a language (cached by size)."""
    return load_stop_words_from_file(language)

# TTL-based caching for memory operations
@cached_memory_operation(ttl=3600)  # 1 hour
def search_memories_by_type(memory_type: str, limit: int = 10) -> list:
    """Search memories by type (cached with TTL)."""
    return database.search_by_type(memory_type, limit)

# Specialized caching for CXD operations  
@cached_cxd_operation(ttl=1800)  # 30 minutes
def classify_query_intent(query: str) -> dict:
    """Classify query intent (shorter TTL for evolving data)."""
    return cxd_classifier.classify(query)

# Long-term caching for expensive operations
@cached_embedding_operation(ttl=7200)  # 2 hours
def generate_text_embedding(text: str) -> list:
    """Generate text embedding (cached longer due to expense)."""
    return embedding_model.encode(text)
```

#### Class-Method Caching

```python
class MemorySearchService:
    """Service class with integrated caching."""
    
    def __init__(self):
        self.cache_manager = create_cache_manager(
            cache_type="lru",
            max_memory_mb=256,
            max_items=5000
        )
    
    @cached_memory_operation(ttl=1800)
    def search_recent_memories(self, days: int = 7) -> list:
        """Search recent memories with caching."""
        return self._perform_search(days)
    
    @cached_cxd_operation(ttl=1200) 
    def classify_and_search(self, query: str) -> dict:
        """Classify query and perform filtered search."""
        classification = self.classify_query(query)
        results = self.search_by_classification(classification)
        return {
            'classification': classification,
            'results': results
        }
    
    def _perform_search(self, days: int) -> list:
        # Actual search implementation
        pass
```

### 4. Database Pool Integration

```python
from memmimic.memory.active.database_pool import DatabaseConnectionPool

# Initialize connection pool
db_pool = DatabaseConnectionPool(
    database_path="memmimic.db",
    pool_size=5,
    max_overflow=10,
    recycle_time=3600,
    connect_timeout=30
)

# Use pooled connections
def search_with_pool(query: str) -> list:
    """Search using connection pool."""
    with db_pool.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM memories WHERE content LIKE ?", 
            (f"%{query}%",)
        )
        return [dict(row) for row in cursor.fetchall()]

# Monitor pool health
health = db_pool.get_pool_health()
print(f"Active connections: {health['active_connections']}")
print(f"Pool efficiency: {health['efficiency']:.1%}")
print(f"Average query time: {health['avg_query_time_ms']:.2f}ms")

# Optimize pool performance
if health['efficiency'] < 0.8:
    print("Pool efficiency low - consider adjusting pool size")
```

### 5. Performance Optimization Integration

```python
from memmimic.memory.active.optimization_engine import OptimizationEngine

# Initialize optimization engine
optimizer = OptimizationEngine(
    cache_manager=cache_manager,
    database_pool=db_pool,
    optimization_interval=600  # 10 minutes
)

# Run optimization cycle
results = optimizer.optimize()
print(f"Memory freed: {results['memory_freed_mb']:.2f}MB")
print(f"Cache hit rate improved by: {results['hit_rate_improvement']:.1%}")
print(f"Query performance improved by: {results['query_performance_gain']:.1%}")

# Get optimization recommendations
recommendations = optimizer.get_optimization_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec['action']}")
    print(f"Expected benefit: {rec['benefit']}")
    print(f"Implementation: {rec['implementation']}")
```

## Configuration Integration

### 1. YAML Configuration

Create `config/memmimic_integration.yaml`:

```yaml
# MemMimic Integration Configuration
memmimic:
  database:
    path: "memmimic.db"
    pool_size: 5
    max_overflow: 10
    
  search:
    # Hybrid search configuration
    semantic_weight: 0.7
    wordnet_weight: 0.3
    convergence_bonus: 0.1
    default_limit: 10
    
    # Component-specific settings
    semantic_processor:
      similarity_threshold: 0.1
      similarity_metric: "cosine"
      
    wordnet_expander:
      max_synonyms_per_word: 3
      enable_multilingual: true
      
    result_combiner:
      default_strategy: "weighted_sum"
      enable_statistics: true
  
  caching:
    # Global cache settings
    total_memory_limit_mb: 512
    enable_monitoring: true
    
    # Cache pool configuration
    pools:
      search_results:
        max_memory_mb: 128
        max_items: 2000
        default_ttl_seconds: 1800
        
      embeddings:
        max_memory_mb: 64
        max_items: 1000
        default_ttl_seconds: 7200
        
      classifications:
        max_memory_mb: 32
        max_items: 500
        default_ttl_seconds: 3600
  
  performance:
    # Optimization settings
    enable_auto_optimization: true
    optimization_interval_seconds: 600
    memory_pressure_threshold: 0.8
    
    # Monitoring thresholds
    alert_on_low_hit_rate: 0.6
    alert_on_high_memory: 400
    alert_on_slow_queries: 200
```

### 2. Configuration Loading

```python
import yaml
from pathlib import Path

def load_memmimic_config(config_path: str = "config/memmimic_integration.yaml") -> dict:
    """Load MemMimic integration configuration."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def initialize_from_config(config_path: str = None) -> dict:
    """Initialize MemMimic components from configuration."""
    config = load_memmimic_config(config_path) if config_path else {}
    
    # Initialize search engine with config
    search_config = config.get('memmimic', {}).get('search', {})
    search_engine = HybridSearchEngine(
        db_name=config.get('memmimic', {}).get('database', {}).get('path', 'memmimic.db')
    )
    
    # Configure search weights
    search_engine.default_semantic_weight = search_config.get('semantic_weight', 0.7)
    search_engine.default_wordnet_weight = search_config.get('wordnet_weight', 0.3)
    search_engine.default_convergence_bonus = search_config.get('convergence_bonus', 0.1)
    
    # Initialize cache manager with config
    cache_config = config.get('memmimic', {}).get('caching', {})
    cache_manager = create_cache_manager(
        cache_type="pool",
        pool_config=cache_config.get('pools', {})
    )
    
    return {
        'search_engine': search_engine,
        'cache_manager': cache_manager,
        'config': config
    }
```

## Error Handling Integration

### 1. Comprehensive Error Handling

```python
from memmimic.errors.exceptions import (
    MemMimicError, CacheError, SearchError, 
    DatabaseError, ValidationError
)
from memmimic.errors.handlers import handle_errors

class RobustMemMimicClient:
    """MemMimic client with comprehensive error handling."""
    
    def __init__(self, config: dict):
        self.search_engine = None
        self.cache_manager = None
        self.initialize_components(config)
    
    @handle_errors(catch=[MemMimicError], log_level="ERROR", fallback_value=[])
    def search_with_fallback(self, query: str, **kwargs) -> list:
        """Search with automatic fallback on errors."""
        try:
            # Try hybrid search first
            results = self.search_engine.search_memories_hybrid(query, **kwargs)
            return results['results']
            
        except SearchError as e:
            logger.warning(f"Hybrid search failed: {e}, trying simple search")
            # Fallback to simple search
            return self.simple_search(query, **kwargs)
            
        except CacheError as e:
            logger.warning(f"Cache error during search: {e}")
            # Clear cache and retry
            self.cache_manager.clear()
            return self.search_engine.search_memories_hybrid(query, **kwargs)['results']
    
    @handle_errors(catch=[DatabaseError], log_level="ERROR", retry_count=3)
    def resilient_memory_operation(self, operation_func, *args, **kwargs):
        """Execute memory operation with automatic retry on database errors."""
        return operation_func(*args, **kwargs)
    
    def initialize_components(self, config: dict):
        """Initialize components with error handling."""
        try:
            components = initialize_from_config()
            self.search_engine = components['search_engine']
            self.cache_manager = components['cache_manager']
            
        except Exception as e:
            logger.error(f"Failed to initialize MemMimic components: {e}")
            # Initialize with minimal configuration
            self.search_engine = HybridSearchEngine("memmimic.db")
            self.cache_manager = create_cache_manager(cache_type="lru")
```

### 2. Graceful Degradation

```python
class GracefulMemMimicService:
    """Service with graceful degradation capabilities."""
    
    def __init__(self):
        self.cache_available = True
        self.search_engine_available = True
        self.performance_mode = "full"  # full, cached_only, minimal
    
    def adaptive_search(self, query: str, **kwargs) -> dict:
        """Search that adapts to system conditions."""
        
        # Check system health
        system_health = self.check_system_health()
        
        if system_health['memory_pressure'] > 0.9:
            self.performance_mode = "minimal"
            return self.minimal_search(query)
            
        elif system_health['cache_hit_rate'] < 0.3:
            self.performance_mode = "cached_only"
            return self.cached_only_search(query)
            
        else:
            self.performance_mode = "full"
            return self.full_search(query, **kwargs)
    
    def check_system_health(self) -> dict:
        """Check overall system health."""
        try:
            cache_stats = self.cache_manager.get_stats()
            memory_usage = cache_stats.get('current_memory_mb', 0)
            max_memory = cache_stats.get('max_memory_mb', 512)
            
            return {
                'memory_pressure': memory_usage / max_memory,
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'system_responsive': True
            }
        except Exception:
            return {
                'memory_pressure': 1.0,
                'cache_hit_rate': 0.0,
                'system_responsive': False
            }
```

## Performance Monitoring Integration

### 1. Metrics Collection

```python
import time
from contextlib import contextmanager

class PerformanceMonitor:
    """Performance monitoring for MemMimic integration."""
    
    def __init__(self):
        self.metrics = {
            'search_times': [],
            'cache_hit_rates': [],
            'memory_usage': [],
            'error_counts': {}
        }
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        start_time = time.perf_counter()
        
        try:
            yield
            
        except Exception as e:
            # Count errors
            error_type = type(e).__name__
            self.metrics['error_counts'][error_type] = (
                self.metrics['error_counts'].get(error_type, 0) + 1
            )
            raise
            
        finally:
            # Record timing
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics['search_times'].append({
                'operation': operation_name,
                'duration_ms': duration_ms,
                'timestamp': time.time()
            })
    
    def get_performance_summary(self) -> dict:
        """Get performance summary."""
        if not self.metrics['search_times']:
            return {'status': 'no_data'}
        
        recent_times = [
            entry['duration_ms'] 
            for entry in self.metrics['search_times'][-100:]  # Last 100 operations
        ]
        
        return {
            'avg_response_time_ms': sum(recent_times) / len(recent_times),
            'max_response_time_ms': max(recent_times),
            'min_response_time_ms': min(recent_times),
            'total_operations': len(self.metrics['search_times']),
            'error_rate': sum(self.metrics['error_counts'].values()) / len(self.metrics['search_times']),
            'recent_errors': self.metrics['error_counts']
        }

# Usage example
monitor = PerformanceMonitor()

def monitored_search(query: str) -> list:
    """Search with performance monitoring."""
    with monitor.measure_operation("hybrid_search"):
        results = search_engine.search_memories_hybrid(query)
        return results
```

### 2. Health Checks

```python
class HealthChecker:
    """System health monitoring for MemMimic integration."""
    
    def __init__(self, search_engine, cache_manager, db_pool):
        self.search_engine = search_engine
        self.cache_manager = cache_manager
        self.db_pool = db_pool
    
    def comprehensive_health_check(self) -> dict:
        """Perform comprehensive health check."""
        return {
            'search_engine': self.check_search_engine_health(),
            'cache_system': self.check_cache_health(),
            'database': self.check_database_health(),
            'overall': self.calculate_overall_health()
        }
    
    def check_search_engine_health(self) -> dict:
        """Check search engine health."""
        try:
            # Test search with simple query
            start_time = time.time()
            results = self.search_engine.search_memories_hybrid("test", limit=1)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'results_count': len(results.get('results', [])),
                'last_check': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy', 
                'error': str(e),
                'last_check': time.time()
            }
    
    def check_cache_health(self) -> dict:
        """Check cache system health."""
        try:
            stats = self.cache_manager.get_stats()
            
            # Evaluate cache health
            hit_rate = stats.get('hit_rate', 0)
            memory_utilization = stats.get('memory_utilization', 0)
            
            if hit_rate < 0.5:
                status = 'degraded'
                reason = f'Low hit rate: {hit_rate:.1%}'
            elif memory_utilization > 0.9:
                status = 'warning'
                reason = f'High memory usage: {memory_utilization:.1%}'
            else:
                status = 'healthy'
                reason = 'All metrics within normal ranges'
            
            return {
                'status': status,
                'reason': reason,
                'hit_rate': hit_rate,
                'memory_utilization': memory_utilization,
                'last_check': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': time.time()
            }
```

## Best Practices Summary

### 1. Integration Checklist

- [ ] **Configuration Management**: Use YAML configuration files for all settings
- [ ] **Error Handling**: Implement comprehensive error handling with fallbacks
- [ ] **Performance Monitoring**: Add metrics collection and health checks
- [ ] **Cache Strategy**: Choose appropriate cache types and TTL values
- [ ] **Resource Management**: Monitor memory usage and connection pools
- [ ] **Testing**: Include integration tests for all components
- [ ] **Logging**: Configure detailed logging for troubleshooting
- [ ] **Documentation**: Document custom configurations and integrations

### 2. Performance Optimization

- **Cache Hit Rates**: Target >80% for repeated operations
- **Response Times**: Target <50ms for cached operations, <200ms for cold operations
- **Memory Usage**: Stay under configured limits with 20% buffer
- **Connection Efficiency**: Maintain >90% database pool efficiency
- **Error Rates**: Keep error rates <1% under normal conditions

### 3. Operational Considerations

- **Monitoring**: Set up alerts for performance degradation
- **Backup**: Regular backup of configuration and database files
- **Updates**: Test component updates in staging before production
- **Scaling**: Monitor resource usage trends for capacity planning
- **Security**: Validate all inputs and sanitize database queries

This integration guide provides the foundation for successfully implementing MemMimic's Phase 2 enhancements in your applications.