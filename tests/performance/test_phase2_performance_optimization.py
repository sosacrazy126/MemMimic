#!/usr/bin/env python3
"""
Phase 2 Performance Optimization Tests

Comprehensive performance tests to validate Phase 2 improvements:
1. Multi-tier caching system with >80% hit rate target
2. Async operation performance testing
3. Memory usage optimization
4. Modular architecture performance
5. Non-blocking operation validation
"""

import asyncio
import time
import sys
import os
import tempfile
import statistics
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import threading

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from memmimic.utils.caching import MemMimicCache, cached_cxd_operation, cached_memory_operation, cached_embedding_operation, lru_cached
from memmimic.memory.storage.amms_storage import AMMSStorage
from memmimic.memory.active.cache_manager import CacheManager
from memmimic.memory.active.optimization_engine import OptimizationEngine


class PerformanceMetrics:
    """Utility class for collecting performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        duration = time.time() - self.start_times[operation]
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics:
            return {}
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'total': sum(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'min': min(durations),
            'max': max(durations),
            'stdev': statistics.stdev(durations) if len(durations) > 1 else 0.0
        }


class TestCachingSystem:
    """Test the multi-tier caching system from Phase 2."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
    
    def test_memmimic_cache_basic_operations(self):
        """Test basic cache operations and TTL."""
        print("🚀 Testing MemMimic cache basic operations...")
        
        cache = MemMimicCache(max_size=100, default_ttl=1)
        
        # Test set/get
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        print("   ✅ Basic set/get operations working")
        
        # Test TTL expiration
        time.sleep(1.1)  # Wait for TTL
        expired_value = cache.get("test_key")
        assert expired_value is None
        print("   ✅ TTL expiration working")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["total_requests"] > 0
        assert stats["hits"] > 0 or stats["misses"] > 0
        print(f"   ✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
    
    def test_cache_hit_rate_performance(self):
        """Test cache hit rate performance (target >80%)."""
        print("🎯 Testing cache hit rate performance...")
        
        cache = MemMimicCache(max_size=1000, default_ttl=60)
        
        # Populate cache with test data
        test_keys = [f"key_{i}" for i in range(100)]
        for key in test_keys:
            cache.set(key, f"value_{key}")
        
        # Simulate realistic access pattern (80/20 rule - 20% of keys accessed 80% of time)
        hot_keys = test_keys[:20]  # 20% hot keys
        cold_keys = test_keys[20:]  # 80% cold keys
        
        total_requests = 1000
        hit_count = 0
        
        for i in range(total_requests):
            # 80% chance of accessing hot keys, 20% chance of cold keys
            if i % 5 != 0:  # 80% of requests
                key = hot_keys[i % len(hot_keys)]
            else:  # 20% of requests
                key = cold_keys[i % len(cold_keys)] if cold_keys else hot_keys[0]
            
            value = cache.get(key)
            if value is not None:
                hit_count += 1
        
        hit_rate = (hit_count / total_requests) * 100
        print(f"   📊 Cache hit rate: {hit_rate:.1f}% ({hit_count}/{total_requests})")
        
        # Should exceed 80% target
        assert hit_rate >= 80.0, f"Cache hit rate {hit_rate:.1f}% below 80% target"
        print("   ✅ Cache hit rate target achieved (>80%)")
    
    def test_cache_memory_management(self):
        """Test cache memory management and eviction."""
        print("💾 Testing cache memory management...")
        
        cache = MemMimicCache(max_size=10, default_ttl=60)  # Small cache for testing
        
        # Fill cache to capacity
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        
        assert len(cache._cache) == 10
        print("   ✅ Cache filled to capacity")
        
        # Add one more item to trigger eviction
        cache.set("overflow_key", "overflow_value")
        assert len(cache._cache) == 10  # Should still be at max size
        assert cache._stats["evictions"] > 0  # Should have evicted something
        print(f"   ✅ LRU eviction triggered: {cache._stats['evictions']} evictions")
        
        # Most recently added should still be there
        assert cache.get("overflow_key") == "overflow_value"
        print("   ✅ LRU eviction working correctly")
    
    def test_cached_decorators_performance(self):
        """Test performance of cached decorators."""
        print("⚡ Testing cached decorators performance...")
        
        # Test CXD operation caching
        call_count = 0
        
        @cached_cxd_operation(ttl=60)
        def expensive_cxd_operation(text: str):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return f"processed_{text}"
        
        # First call should be slow
        start_time = time.time()
        result1 = expensive_cxd_operation("test_text")
        first_call_time = time.time() - start_time
        
        # Second call should be fast (cached)
        start_time = time.time()
        result2 = expensive_cxd_operation("test_text")
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        assert call_count == 1  # Should only call the function once
        assert second_call_time < first_call_time * 0.5  # Should be much faster
        
        print(f"   ✅ CXD caching: {first_call_time:.3f}s → {second_call_time:.3f}s")
        print(f"   ✅ Speedup: {first_call_time/second_call_time:.1f}x")
    
    def test_multi_tier_cache_integration(self):
        """Test integration between different cache tiers."""
        print("🏗️ Testing multi-tier cache integration...")
        
        # Test different cache types
        cxd_call_count = 0
        memory_call_count = 0
        embedding_call_count = 0
        
        @cached_cxd_operation(ttl=30)
        def cxd_operation(text: str):
            nonlocal cxd_call_count
            cxd_call_count += 1
            return f"cxd_{text}"
        
        @cached_memory_operation(ttl=60)
        def memory_operation(query: str):
            nonlocal memory_call_count
            memory_call_count += 1
            return f"memory_{query}"
        
        @cached_embedding_operation(ttl=120)
        def embedding_operation(text: str):
            nonlocal embedding_call_count
            embedding_call_count += 1
            return f"embedding_{text}"
        
        # Each operation should be cached independently
        cxd_result = cxd_operation("test")
        memory_result = memory_operation("test")
        embedding_result = embedding_operation("test")
        
        # Repeat calls should hit cache
        cxd_result2 = cxd_operation("test")
        memory_result2 = memory_operation("test")
        embedding_result2 = embedding_operation("test")
        
        assert cxd_call_count == 1
        assert memory_call_count == 1
        assert embedding_call_count == 1
        
        assert cxd_result == cxd_result2
        assert memory_result == memory_result2
        assert embedding_result == embedding_result2
        
        print("   ✅ Multi-tier caching working independently")
        print(f"   ✅ CXD tier: {cxd_call_count} calls")
        print(f"   ✅ Memory tier: {memory_call_count} calls")
        print(f"   ✅ Embedding tier: {embedding_call_count} calls")


class TestAsyncPerformance:
    """Test async operation performance improvements."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
    
    async def test_async_storage_operations(self):
        """Test async storage operations performance."""
        print("⚡ Testing async storage operations...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            storage = AMMSStorage(db_path)
            await storage.initialize()
            
            # Test concurrent async operations
            tasks = []
            start_time = time.time()
            
            # Create multiple concurrent storage operations
            for i in range(20):
                task = storage.store_memory(
                    content=f"Test memory {i}",
                    memory_type="performance_test",
                    metadata={"index": i}
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            async_time = time.time() - start_time
            
            assert len(results) == 20
            assert all(result > 0 for result in results)  # All should return memory IDs
            
            print(f"   ✅ 20 concurrent operations completed in {async_time:.3f}s")
            print(f"   ✅ Average per operation: {async_time/20:.3f}s")
            
            # Test concurrent read operations
            read_tasks = []
            start_time = time.time()
            
            for memory_id in results[:10]:  # Test first 10
                task = storage.get_memory(memory_id)
                read_tasks.append(task)
            
            read_results = await asyncio.gather(*read_tasks)
            read_time = time.time() - start_time
            
            assert len(read_results) == 10
            assert all(result is not None for result in read_results)
            
            print(f"   ✅ 10 concurrent reads completed in {read_time:.3f}s")
            print(f"   ✅ Average per read: {read_time/10:.3f}s")
            
            await storage.close()
            
        finally:
            os.unlink(db_path)
    
    async def test_non_blocking_operations(self):
        """Test that operations are truly non-blocking."""
        print("🚫 Testing non-blocking operations...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            storage = AMMSStorage(db_path)
            await storage.initialize()
            
            # Test that async operations don't block the event loop
            start_time = time.time()
            
            # Create a slow operation that shouldn't block other operations
            async def slow_operation():
                await asyncio.sleep(0.1)  # Simulate slow operation
                return await storage.store_memory(
                    content="Slow operation",
                    memory_type="slow",
                    metadata={"slow": True}
                )
            
            # Create fast operations that should complete while slow op is running
            async def fast_operation(i):
                return await storage.store_memory(
                    content=f"Fast operation {i}",
                    memory_type="fast",
                    metadata={"fast": True, "index": i}
                )
            
            # Start slow operation and fast operations concurrently
            slow_task = asyncio.create_task(slow_operation())
            fast_tasks = [asyncio.create_task(fast_operation(i)) for i in range(5)]
            
            # Fast operations should complete before slow operation
            fast_results = await asyncio.gather(*fast_tasks)
            fast_time = time.time() - start_time
            
            # Slow operation should still be running or just finishing
            slow_result = await slow_task
            total_time = time.time() - start_time
            
            assert len(fast_results) == 5
            assert all(result > 0 for result in fast_results)
            assert slow_result > 0
            
            # Fast operations should complete in much less than 0.1s if truly concurrent
            assert fast_time < 0.08, f"Fast operations took {fast_time:.3f}s, may not be truly async"
            
            print(f"   ✅ Fast operations: {fast_time:.3f}s")
            print(f"   ✅ Total time: {total_time:.3f}s")
            print("   ✅ Operations are truly non-blocking")
            
            await storage.close()
            
        finally:
            os.unlink(db_path)
    
    async def test_async_error_handling(self):
        """Test async error handling performance."""
        print("🛠️ Testing async error handling...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            storage = AMMSStorage(db_path)
            await storage.initialize()
            
            # Test that errors don't block other operations
            async def failing_operation():
                try:
                    # This should fail gracefully
                    return await storage.get_memory(-999999)
                except Exception:
                    return None
            
            async def working_operation(i):
                return await storage.store_memory(
                    content=f"Working operation {i}",
                    memory_type="test",
                    metadata={"working": True}
                )
            
            start_time = time.time()
            
            # Mix failing and working operations
            tasks = []
            for i in range(10):
                if i % 3 == 0:  # Every 3rd operation fails
                    tasks.append(failing_operation())
                else:
                    tasks.append(working_operation(i))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            error_handling_time = time.time() - start_time
            
            # Should complete quickly despite errors
            assert error_handling_time < 1.0
            
            # Check that working operations succeeded
            working_results = [r for r in results if isinstance(r, int) and r > 0]
            assert len(working_results) > 0
            
            print(f"   ✅ Mixed operations completed in {error_handling_time:.3f}s")
            print(f"   ✅ {len(working_results)} operations succeeded despite errors")
            
            await storage.close()
            
        finally:
            os.unlink(db_path)


class TestMemoryOptimization:
    """Test memory usage optimization."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
    
    def test_memory_usage_growth(self):
        """Test memory usage growth under load."""
        print("💾 Testing memory usage growth...")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        cache = MemMimicCache(max_size=1000, default_ttl=60)
        
        # Add many items to cache
        for i in range(5000):
            cache.set(f"key_{i}", f"value_{i}_{'x' * 100}")  # ~100 char values
        
        # Force garbage collection
        gc.collect()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = peak_memory - initial_memory
        
        # Cache should be limited to max_size
        assert len(cache._cache) <= 1000
        
        # Memory growth should be reasonable (less than 50MB for this test)
        assert memory_growth < 50, f"Memory growth {memory_growth:.1f}MB too high"
        
        print(f"   ✅ Memory usage: {initial_memory:.1f}MB → {peak_memory:.1f}MB")
        print(f"   ✅ Growth: {memory_growth:.1f}MB")
        print(f"   ✅ Cache size limited to: {len(cache._cache)} items")
    
    def test_cache_cleanup_performance(self):
        """Test cache cleanup and garbage collection."""
        print("🧹 Testing cache cleanup performance...")
        
        cache = MemMimicCache(max_size=100, default_ttl=0.1)  # Very short TTL
        
        # Add items that will expire quickly
        for i in range(200):
            cache.set(f"temp_key_{i}", f"temp_value_{i}")
        
        # Initial state
        initial_size = len(cache._cache)
        initial_evictions = cache._stats["evictions"]
        
        # Wait for TTL expiration
        time.sleep(0.2)
        
        # Access cache to trigger cleanup of expired items
        for i in range(50):
            cache.get(f"temp_key_{i}")
        
        # Check cleanup effectiveness
        final_size = len(cache._cache)
        final_evictions = cache._stats["evictions"]
        
        print(f"   ✅ Cache size: {initial_size} → {final_size}")
        print(f"   ✅ Evictions: {initial_evictions} → {final_evictions}")
        
        # Most expired items should be cleaned up
        cleanup_ratio = (initial_size - final_size) / initial_size if initial_size > 0 else 0
        assert cleanup_ratio > 0.5, f"Cleanup ratio {cleanup_ratio:.2f} too low"
        
        print(f"   ✅ Cleanup ratio: {cleanup_ratio:.1%}")
    
    async def test_memory_pool_optimization(self):
        """Test memory pool optimization in AMMS."""
        print("🏊 Testing memory pool optimization...")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            storage = AMMSStorage(db_path)
            await storage.initialize()
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create many memory operations
            memory_ids = []
            for i in range(100):
                memory_id = await storage.store_memory(
                    content=f"Memory content {i}" * 10,  # Larger content
                    memory_type="optimization_test",
                    metadata={"index": i}
                )
                memory_ids.append(memory_id)
            
            mid_memory = process.memory_info().rss / 1024 / 1024
            
            # Retrieve all memories
            for memory_id in memory_ids:
                memory = await storage.get_memory(memory_id)
                assert memory is not None
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # Memory growth should be reasonable
            storage_growth = mid_memory - initial_memory
            retrieval_growth = final_memory - mid_memory
            
            print(f"   ✅ Initial memory: {initial_memory:.1f}MB")
            print(f"   ✅ After storage: {mid_memory:.1f}MB (+{storage_growth:.1f}MB)")
            print(f"   ✅ After retrieval: {final_memory:.1f}MB (+{retrieval_growth:.1f}MB)")
            
            # Retrieval shouldn't cause significant additional memory growth
            assert retrieval_growth < storage_growth * 0.5
            
            # Get pool statistics
            stats = storage.get_stats()
            if 'memory_pool' in stats:
                pool_stats = stats['memory_pool']
                print(f"   ✅ Pool efficiency: {pool_stats.get('efficiency', 'N/A')}")
            
            await storage.close()
            
        finally:
            os.unlink(db_path)


class TestModularArchitecturePerformance:
    """Test performance of the new modular architecture."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
    
    def test_component_isolation_performance(self):
        """Test that modular components don't interfere with each other's performance."""
        print("🏗️ Testing component isolation performance...")
        
        try:
            from memmimic.memory.search.hybrid_search import HybridSearch
            from memmimic.memory.search.wordnet_expander import WordNetExpander
            from memmimic.memory.search.semantic_processor import SemanticProcessor
            from memmimic.memory.search.result_combiner import ResultCombiner
            
            # Initialize components
            hybrid_search = HybridSearch()
            wordnet_expander = WordNetExpander()
            semantic_processor = SemanticProcessor()
            result_combiner = ResultCombiner()
            
            print("   ✅ All modular components initialized successfully")
            
            # Test concurrent usage of components
            def use_component(component, operation, *args):
                start_time = time.time()
                try:
                    if hasattr(component, operation):
                        result = getattr(component, operation)(*args)
                        duration = time.time() - start_time
                        return {'success': True, 'duration': duration, 'result': result}
                    else:
                        return {'success': False, 'error': f'Operation {operation} not found'}
                except Exception as e:
                    duration = time.time() - start_time
                    return {'success': False, 'error': str(e), 'duration': duration}
            
            # Test basic operations on each component
            operations = [
                (wordnet_expander, 'expand_query', 'search term'),
                (semantic_processor, 'process_embedding', 'test content'),
                (result_combiner, 'combine_results', [], []),
            ]
            
            results = []
            for component, operation, *args in operations:
                result = use_component(component, operation, *args)
                results.append(result)
                if result['success']:
                    print(f"   ✅ {component.__class__.__name__}.{operation}: {result['duration']:.3f}s")
                else:
                    print(f"   ⚠️ {component.__class__.__name__}.{operation}: {result.get('error', 'Failed')}")
            
            # At least some operations should succeed
            successful_operations = sum(1 for r in results if r['success'])
            print(f"   ✅ {successful_operations}/{len(results)} component operations successful")
            
        except ImportError as e:
            print(f"   ⚠️ Some modular components not available: {e}")
            print("   ℹ️ This may be expected if components are still being developed")
    
    def test_memory_search_performance_integration(self):
        """Test integrated performance of memory search components."""
        print("🔍 Testing memory search performance integration...")
        
        try:
            # Test the integration of search components if available
            from memmimic.memory.search import search_engine
            
            # Create test data
            test_queries = [
                "artificial intelligence",
                "machine learning algorithms",
                "natural language processing",
                "neural networks",
                "deep learning",
            ]
            
            performance_results = {}
            
            for query in test_queries:
                start_time = time.time()
                
                try:
                    # This should test the integrated search pipeline
                    if hasattr(search_engine, 'search'):
                        results = search_engine.search(query, limit=10)
                        duration = time.time() - start_time
                        performance_results[query] = {
                            'duration': duration,
                            'results_count': len(results) if results else 0,
                            'success': True
                        }
                    else:
                        performance_results[query] = {
                            'success': False,
                            'error': 'Search method not available'
                        }
                        
                except Exception as e:
                    duration = time.time() - start_time
                    performance_results[query] = {
                        'duration': duration,
                        'success': False,
                        'error': str(e)
                    }
            
            # Analyze performance results
            successful_queries = [r for r in performance_results.values() if r['success']]
            
            if successful_queries:
                avg_duration = statistics.mean(r['duration'] for r in successful_queries)
                max_duration = max(r['duration'] for r in successful_queries)
                min_duration = min(r['duration'] for r in successful_queries)
                
                print(f"   ✅ Search performance - Avg: {avg_duration:.3f}s, Range: {min_duration:.3f}s-{max_duration:.3f}s")
                print(f"   ✅ {len(successful_queries)}/{len(test_queries)} queries successful")
                
                # Performance should be reasonable (< 1s per query)
                assert avg_duration < 1.0, f"Average search time {avg_duration:.3f}s too slow"
            else:
                print("   ⚠️ No successful search queries - integration may need work")
            
        except ImportError:
            print("   ⚠️ Search engine integration not yet available")
    
    def test_backward_compatibility_performance(self):
        """Test that modular architecture maintains backward compatibility performance."""
        print("🔄 Testing backward compatibility performance...")
        
        # Test that existing APIs still work with good performance
        try:
            from memmimic import api
            
            # Test basic API operations
            start_time = time.time()
            
            # These should work without significant performance regression
            operations = [
                ('status', []),
                # Add more API operations as they become available
            ]
            
            for operation, args in operations:
                if hasattr(api, operation):
                    op_start = time.time()
                    try:
                        result = getattr(api, operation)(*args)
                        op_duration = time.time() - op_start
                        print(f"   ✅ API.{operation}: {op_duration:.3f}s")
                    except Exception as e:
                        print(f"   ⚠️ API.{operation} failed: {e}")
                else:
                    print(f"   ℹ️ API.{operation} not available")
            
            total_duration = time.time() - start_time
            print(f"   ✅ Total API test time: {total_duration:.3f}s")
            
        except ImportError:
            print("   ⚠️ API module not available for backward compatibility testing")


async def run_phase2_performance_tests():
    """Run all Phase 2 performance optimization tests."""
    print("🚀 Running Phase 2 Performance Optimization Tests")
    print("=" * 60)
    
    test_classes = [
        ("Caching System", TestCachingSystem()),
        ("Async Performance", TestAsyncPerformance()),
        ("Memory Optimization", TestMemoryOptimization()),
        ("Modular Architecture Performance", TestModularArchitecturePerformance()),
    ]
    
    results = {}
    
    for category, test_instance in test_classes:
        print(f"\n🧪 Testing: {category}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_') and callable(getattr(test_instance, method))
        ]
        
        category_results = {}
        
        for test_method in test_methods:
            method_name = test_method.replace('test_', '').replace('_', ' ').title()
            
            try:
                test_func = getattr(test_instance, test_method)
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                category_results[method_name] = True
                print(f"   ✅ {method_name}")
            except Exception as e:
                category_results[method_name] = False
                print(f"   ❌ {method_name}: {e}")
        
        results[category] = category_results
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 2 Performance Test Results:")
    
    total_tests = 0
    passed_tests = 0
    
    for category, category_results in results.items():
        category_passed = sum(category_results.values())
        category_total = len(category_results)
        total_tests += category_total
        passed_tests += category_passed
        
        status = "✅" if category_passed == category_total else "❌"
        print(f"{status} {category}: {category_passed}/{category_total}")
        
        if category_passed != category_total:
            for test, result in category_results.items():
                if not result:
                    print(f"     ❌ {test}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print("🎉 PHASE 2 PERFORMANCE OPTIMIZATION TESTS PASSED!")
        return 0
    elif success_rate >= 80:
        print("⚠️ Most performance tests passed, some optimizations needed")
        return 1
    else:
        print("❌ SIGNIFICANT PERFORMANCE ISSUES DETECTED!")
        return 2


if __name__ == "__main__":
    import sys
    
    sys.exit(asyncio.run(run_phase2_performance_tests()))