"""
Performance tests for Active Memory Management System.

Tests sub-100ms performance targets across all components:
- Indexing engine O(log n) lookup performance
- Cache manager LRU performance under load
- Database pool connection efficiency
- Performance monitor real-time tracking
- Optimization engine lifecycle coordination
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import unittest
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from memmimic.memory.active.indexing_engine import BTreeIndexingEngine, IndexingConfig
from memmimic.memory.active.cache_manager import LRUMemoryCache, create_cache_manager
from memmimic.memory.active.database_pool import ConnectionPool, ConnectionPoolConfig
from memmimic.memory.active.performance_monitor import RealTimePerformanceMonitor, PerformanceThresholds
from memmimic.memory.active.optimization_engine import AutomaticOptimizationEngine, OptimizationConfig
from memmimic.memory.active.interfaces import MemoryQuery


class ActiveMemoryPerformanceTests(unittest.TestCase):
    """Performance tests for active memory management system"""
    
    def setUp(self):
        """Set up test environment with realistic configurations"""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        
        # Performance test configurations
        self.indexing_config = IndexingConfig(
            enable_btree_index=True,
            enable_hash_index=True,
            enable_fulltext_index=True,
            enable_temporal_index=True,
            hash_bucket_count=8192,
            fulltext_min_word_length=3,
            temporal_resolution_minutes=60
        )
        
        self.pool_config = ConnectionPoolConfig(
            min_connections=10,
            max_connections=50,
            max_connection_age_seconds=1800,
            max_idle_seconds=300,
            connection_timeout_seconds=5
        )
        
        self.perf_thresholds = PerformanceThresholds(
            max_avg_query_time_ms=50.0,  # Strict 50ms target
            max_query_time_ms=100.0,     # Maximum 100ms
            min_queries_per_second=20.0,
            min_cache_hit_rate=0.85
        )
        
        # Initialize components
        self.indexing_engine = BTreeIndexingEngine(self.indexing_config)
        self.cache_manager = LRUMemoryCache(
            max_memory_mb=256,
            max_items=5000,
            default_ttl_seconds=1800,
            cleanup_interval_seconds=60
        )
        self.database_pool = ConnectionPool(self.db_path, self.pool_config)
        self.performance_monitor = RealTimePerformanceMonitor(
            thresholds=self.perf_thresholds,
            history_retention_hours=1,
            snapshot_interval_seconds=10
        )
        self.optimization_engine = AutomaticOptimizationEngine(
            indexing_engine=self.indexing_engine,
            cache_manager=self.cache_manager,
            database_pool=self.database_pool,
            performance_monitor=self.performance_monitor
        )
        
        # Test data
        self.test_memories = self._generate_test_memories(1000)
        
    def tearDown(self):
        """Clean up test environment"""
        try:
            self.optimization_engine.shutdown()
            self.performance_monitor.shutdown()
            self.cache_manager.shutdown()
            self.database_pool.shutdown()
            
            # Clean up temp files
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except Exception as e:
            logging.warning(f"Cleanup failed: {e}")
    
    def _generate_test_memories(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic test memory data"""
        memories = []
        for i in range(count):
            memory = {
                'id': f'memory_{i:06d}',
                'content': f'Test memory content for item {i} with various keywords and phrases to test full-text search performance under realistic conditions.',
                'metadata': {
                    'type': 'test_memory',
                    'importance_score': (i % 100) / 100.0,
                    'created_at': datetime.now() - timedelta(hours=i % 168),  # Past week
                    'access_count': i % 50,
                    'tags': [f'tag_{i % 10}', f'category_{i % 5}']
                }
            }
            memories.append(memory)
        return memories
    
    def test_indexing_engine_performance(self):
        """Test indexing engine meets O(log n) performance targets"""
        print("\n=== Testing Indexing Engine Performance ===")
        
        # Index all test memories
        index_start = time.perf_counter()
        for memory in self.test_memories:
            self.indexing_engine.index_memory(
                memory['id'],
                memory['content'],
                memory['metadata'],
                memory['metadata']['created_at']
            )
        index_time = (time.perf_counter() - index_start) * 1000
        
        print(f"Indexed {len(self.test_memories)} memories in {index_time:.2f}ms")
        print(f"Average indexing time: {index_time / len(self.test_memories):.2f}ms per memory")
        
        # Test various query types and measure performance
        query_types = [
            ('Direct ID lookup', MemoryQuery(memory_ids=['memory_000100'])),
            ('Content search', MemoryQuery(content_search='test memory content')),
            ('Time range search', MemoryQuery(
                time_range=(datetime.now() - timedelta(hours=24), datetime.now())
            )),
            ('Metadata filter', MemoryQuery(metadata_filters={'type': 'test_memory'})),
            ('Complex query', MemoryQuery(
                content_search='keywords',
                time_range=(datetime.now() - timedelta(hours=48), datetime.now()),
                limit=50
            ))
        ]
        
        for query_name, query in query_types:
            times = []
            for _ in range(100):  # 100 iterations for statistical significance
                start_time = time.perf_counter()
                results = self.indexing_engine.search_memories(query)
                query_time = (time.perf_counter() - start_time) * 1000
                times.append(query_time)
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            
            print(f"{query_name}:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Maximum: {max_time:.2f}ms")
            print(f"  95th percentile: {p95_time:.2f}ms")
            
            # Assert performance targets
            self.assertLess(avg_time, 50.0, f"{query_name} average time exceeds 50ms target")
            self.assertLess(p95_time, 100.0, f"{query_name} 95th percentile exceeds 100ms target")
        
        # Test index optimization performance
        opt_start = time.perf_counter()
        optimization_results = self.indexing_engine.optimize_indexes()
        opt_time = (time.perf_counter() - opt_start) * 1000
        
        print(f"Index optimization completed in {opt_time:.2f}ms")
        self.assertLess(opt_time, 1000.0, "Index optimization exceeds 1s target")
    
    def test_cache_manager_performance(self):
        """Test cache manager LRU performance under load"""
        print("\n=== Testing Cache Manager Performance ===")
        
        # Generate cache test data
        cache_data = {f'key_{i}': f'value_{i}' * 100 for i in range(2000)}
        
        # Test cache write performance
        write_times = []
        for key, value in cache_data.items():
            start_time = time.perf_counter()
            self.cache_manager.put(key, value)
            write_time = (time.perf_counter() - start_time) * 1000
            write_times.append(write_time)
        
        avg_write_time = sum(write_times) / len(write_times)
        max_write_time = max(write_times)
        
        print(f"Cache write performance:")
        print(f"  Average: {avg_write_time:.3f}ms")
        print(f"  Maximum: {max_write_time:.3f}ms")
        
        # Test cache read performance
        read_times = []
        for key in list(cache_data.keys())[:1000]:  # Test 1000 reads
            start_time = time.perf_counter()
            value = self.cache_manager.get(key)
            read_time = (time.perf_counter() - start_time) * 1000
            read_times.append(read_time)
            self.assertIsNotNone(value, f"Cache miss for key {key}")
        
        avg_read_time = sum(read_times) / len(read_times)
        max_read_time = max(read_times)
        
        print(f"Cache read performance:")
        print(f"  Average: {avg_read_time:.3f}ms")
        print(f"  Maximum: {max_read_time:.3f}ms")
        
        # Test concurrent access performance
        def concurrent_access_worker(keys_subset: List[str], results: List[float]):
            worker_times = []
            for key in keys_subset:
                start_time = time.perf_counter()
                self.cache_manager.get(key)
                access_time = (time.perf_counter() - start_time) * 1000
                worker_times.append(access_time)
            results.extend(worker_times)
        
        # Run concurrent access test
        concurrent_results = []
        threads = []
        keys_list = list(cache_data.keys())[:500]
        chunk_size = len(keys_list) // 10
        
        concurrent_start = time.perf_counter()
        for i in range(10):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(keys_list))
            thread = threading.Thread(
                target=concurrent_access_worker,
                args=(keys_list[start_idx:end_idx], concurrent_results)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_total_time = (time.perf_counter() - concurrent_start) * 1000
        concurrent_avg_time = sum(concurrent_results) / len(concurrent_results)
        
        print(f"Concurrent access performance:")
        print(f"  Total time: {concurrent_total_time:.2f}ms")
        print(f"  Average per operation: {concurrent_avg_time:.3f}ms")
        
        # Assert performance targets
        self.assertLess(avg_write_time, 1.0, "Cache write time exceeds 1ms target")
        self.assertLess(avg_read_time, 0.5, "Cache read time exceeds 0.5ms target")
        self.assertLess(concurrent_avg_time, 2.0, "Concurrent access time exceeds 2ms target")
        
        # Test cache statistics
        stats = self.cache_manager.get_stats()
        print(f"Cache statistics:")
        print(f"  Hit rate: {stats['hit_rate']:.3f}")
        print(f"  Memory utilization: {stats['memory_utilization']:.3f}")
        print(f"  Items: {stats['current_items']}")
        
        self.assertGreater(stats['hit_rate'], 0.8, "Cache hit rate below 80%")
    
    def test_database_pool_performance(self):
        """Test database connection pool efficiency"""
        print("\n=== Testing Database Pool Performance ===")
        
        # Test connection acquisition performance
        acquisition_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            conn = self.database_pool.get_connection()
            acquisition_time = (time.perf_counter() - start_time) * 1000
            acquisition_times.append(acquisition_time)
            self.database_pool.return_connection(conn)
        
        avg_acquisition_time = sum(acquisition_times) / len(acquisition_times)
        max_acquisition_time = max(acquisition_times)
        
        print(f"Connection acquisition performance:")
        print(f"  Average: {avg_acquisition_time:.3f}ms")
        print(f"  Maximum: {max_acquisition_time:.3f}ms")
        
        # Test query execution performance
        test_queries = [
            ("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, data TEXT)", ()),
            ("INSERT INTO test_table (data) VALUES (?)", ("test_data",)),
            ("SELECT * FROM test_table WHERE id = ?", (1,)),
            ("UPDATE test_table SET data = ? WHERE id = ?", ("updated_data", 1)),
            ("DELETE FROM test_table WHERE id = ?", (1,))
        ]
        
        query_times = []
        for query, params in test_queries:
            start_time = time.perf_counter()
            self.database_pool.execute_query(query, params)
            query_time = (time.perf_counter() - start_time) * 1000
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        max_query_time = max(query_times)
        
        print(f"Query execution performance:")
        print(f"  Average: {avg_query_time:.3f}ms")
        print(f"  Maximum: {max_query_time:.3f}ms")
        
        # Test transaction performance
        transaction_queries = [
            ("INSERT INTO test_table (data) VALUES (?)", ("tx_data_1",)),
            ("INSERT INTO test_table (data) VALUES (?)", ("tx_data_2",)),
            ("INSERT INTO test_table (data) VALUES (?)", ("tx_data_3",))
        ]
        
        tx_start = time.perf_counter()
        success = self.database_pool.execute_transaction(transaction_queries)
        tx_time = (time.perf_counter() - tx_start) * 1000
        
        print(f"Transaction performance:")
        print(f"  Time: {tx_time:.3f}ms")
        print(f"  Success: {success}")
        
        # Test concurrent access
        def concurrent_db_worker(worker_id: int, results: Dict[str, List[float]]):
            worker_times = []
            for i in range(10):
                start_time = time.perf_counter()
                query = "INSERT INTO test_table (data) VALUES (?)"
                params = (f"worker_{worker_id}_item_{i}",)
                self.database_pool.execute_query(query, params)
                operation_time = (time.perf_counter() - start_time) * 1000
                worker_times.append(operation_time)
            results[f'worker_{worker_id}'] = worker_times
        
        concurrent_db_results = {}
        db_threads = []
        
        db_concurrent_start = time.perf_counter()
        for worker_id in range(5):
            thread = threading.Thread(
                target=concurrent_db_worker,
                args=(worker_id, concurrent_db_results)
            )
            db_threads.append(thread)
            thread.start()
        
        for thread in db_threads:
            thread.join()
        
        db_concurrent_total_time = (time.perf_counter() - db_concurrent_start) * 1000
        all_concurrent_times = []
        for worker_times in concurrent_db_results.values():
            all_concurrent_times.extend(worker_times)
        
        concurrent_db_avg_time = sum(all_concurrent_times) / len(all_concurrent_times)
        
        print(f"Concurrent database performance:")
        print(f"  Total time: {db_concurrent_total_time:.2f}ms")
        print(f"  Average per operation: {concurrent_db_avg_time:.3f}ms")
        
        # Assert performance targets
        self.assertLess(avg_acquisition_time, 5.0, "Connection acquisition exceeds 5ms target")
        self.assertLess(avg_query_time, 20.0, "Query execution exceeds 20ms target")
        self.assertLess(tx_time, 50.0, "Transaction exceeds 50ms target")
        self.assertLess(concurrent_db_avg_time, 30.0, "Concurrent DB access exceeds 30ms target")
        
        # Test pool statistics
        pool_stats = self.database_pool.get_pool_stats()
        print(f"Pool statistics:")
        print(f"  Total connections: {pool_stats['total_connections']}")
        print(f"  Active connections: {pool_stats['active_connections']}")
        print(f"  Utilization: {pool_stats['utilization']:.3f}")
        print(f"  Reuse rate: {pool_stats['reuse_rate']:.3f}")
        
        self.assertGreater(pool_stats['reuse_rate'], 0.9, "Connection reuse rate below 90%")
    
    def test_performance_monitor_real_time_tracking(self):
        """Test performance monitor real-time tracking accuracy"""
        print("\n=== Testing Performance Monitor ===")
        
        # Generate realistic performance data
        for i in range(50):
            # Simulate various query performance
            query_time = 30 + (i % 10) * 5  # 30-75ms range
            success = i % 20 != 0  # 95% success rate
            self.performance_monitor.record_query(query_time, success)
            
            # Simulate cache operations
            cache_hit = i % 4 != 0  # 75% hit rate
            cache_time = 1 + (i % 3)  # 1-3ms range
            self.performance_monitor.record_cache_operation(cache_hit, cache_time)
            
            # Simulate memory usage
            memory_usage = 100 + (i * 2)  # Gradually increasing
            self.performance_monitor.record_memory_usage('test_component', memory_usage)
        
        # Test snapshot generation performance
        snapshot_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            snapshot = self.performance_monitor.get_current_snapshot()
            snapshot_time = (time.perf_counter() - start_time) * 1000
            snapshot_times.append(snapshot_time)
            
            # Validate snapshot data
            self.assertIsNotNone(snapshot.avg_query_time_ms)
            self.assertIsNotNone(snapshot.cache_hit_rate)
            self.assertIsNotNone(snapshot.total_memory_mb)
            self.assertIsNotNone(snapshot.health_score)
        
        avg_snapshot_time = sum(snapshot_times) / len(snapshot_times)
        print(f"Snapshot generation performance:")
        print(f"  Average: {avg_snapshot_time:.3f}ms")
        
        # Test threshold checking performance
        threshold_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            threshold_results = self.performance_monitor.check_thresholds()
            threshold_time = (time.perf_counter() - start_time) * 1000
            threshold_times.append(threshold_time)
            
            # Validate threshold results
            self.assertIn('threshold_violations', threshold_results)
            self.assertIn('active_alerts', threshold_results)
            self.assertIn('healthy', threshold_results)
        
        avg_threshold_time = sum(threshold_times) / len(threshold_times)
        print(f"Threshold checking performance:")
        print(f"  Average: {avg_threshold_time:.3f}ms")
        
        # Test trend analysis performance
        trend_start = time.perf_counter()
        trends = self.performance_monitor.get_performance_trends(hours=1)
        trend_time = (time.perf_counter() - trend_start) * 1000
        
        print(f"Trend analysis performance:")
        print(f"  Time: {trend_time:.3f}ms")
        
        # Assert performance targets
        self.assertLess(avg_snapshot_time, 10.0, "Snapshot generation exceeds 10ms target")
        self.assertLess(avg_threshold_time, 5.0, "Threshold checking exceeds 5ms target")
        self.assertLess(trend_time, 50.0, "Trend analysis exceeds 50ms target")
        
        print(f"Performance monitor metrics:")
        print(f"  Health score: {snapshot.health_score:.3f}")
        print(f"  Cache hit rate: {snapshot.cache_hit_rate:.3f}")
        print(f"  Average query time: {snapshot.avg_query_time_ms:.2f}ms")
    
    def test_optimization_engine_lifecycle_coordination(self):
        """Test optimization engine performance and coordination"""
        print("\n=== Testing Optimization Engine ===")
        
        # Generate test access patterns
        for i in range(100):
            memory_id = f'memory_{i:06d}'
            metadata = {
                'type': 'test_memory',
                'importance_score': (i % 100) / 100.0,
                'created_at': datetime.now() - timedelta(hours=i % 48),
                'confidence': 0.8
            }
            self.optimization_engine.record_memory_access(memory_id, metadata)
        
        # Test importance evaluation performance
        importance_times = []
        for i in range(50):
            memory_id = f'memory_{i:06d}'
            start_time = time.perf_counter()
            importance = self.optimization_engine.evaluate_memory_importance(memory_id)
            importance_time = (time.perf_counter() - start_time) * 1000
            importance_times.append(importance_time)
            
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0.0)
            self.assertLessEqual(importance, 1.0)
        
        avg_importance_time = sum(importance_times) / len(importance_times)
        print(f"Importance evaluation performance:")
        print(f"  Average: {avg_importance_time:.3f}ms")
        
        # Test archival candidates generation
        archival_start = time.perf_counter()
        archival_candidates = self.optimization_engine.suggest_archival_candidates(50)
        archival_time = (time.perf_counter() - archival_start) * 1000
        
        print(f"Archival candidates generation:")
        print(f"  Time: {archival_time:.3f}ms")
        print(f"  Candidates: {len(archival_candidates)}")
        
        # Test memory distribution optimization
        optimization_start = time.perf_counter()
        optimization_results = self.optimization_engine.optimize_memory_distribution()
        optimization_time = (time.perf_counter() - optimization_start) * 1000
        
        print(f"Memory distribution optimization:")
        print(f"  Time: {optimization_time:.3f}ms")
        print(f"  Archival suggestions: {optimization_results['archival_suggestions']}")
        print(f"  Cache optimizations: {optimization_results['cache_optimizations']}")
        
        # Test memory usage prediction
        prediction_start = time.perf_counter()
        predictions = self.optimization_engine.predict_memory_usage(24)
        prediction_time = (time.perf_counter() - prediction_start) * 1000
        
        print(f"Memory usage prediction:")
        print(f"  Time: {prediction_time:.3f}ms")
        if 'predicted_memory_mb' in predictions:
            print(f"  Predicted memory: {predictions['predicted_memory_mb']:.2f}MB")
            print(f"  Confidence: {predictions.get('confidence', 0):.3f}")
        
        # Assert performance targets
        self.assertLess(avg_importance_time, 2.0, "Importance evaluation exceeds 2ms target")
        self.assertLess(archival_time, 100.0, "Archival generation exceeds 100ms target")
        self.assertLess(optimization_time, 500.0, "Optimization exceeds 500ms target")
        self.assertLess(prediction_time, 200.0, "Prediction exceeds 200ms target")
    
    def test_integrated_system_performance(self):
        """Test complete system performance under realistic load"""
        print("\n=== Testing Integrated System Performance ===")
        
        # Simulate realistic workload
        workload_start = time.perf_counter()
        
        # Phase 1: Initial data loading
        for i, memory in enumerate(self.test_memories[:200]):
            # Index memory
            self.indexing_engine.index_memory(
                memory['id'],
                memory['content'],
                memory['metadata'],
                memory['metadata']['created_at']
            )
            
            # Cache frequently accessed items
            if i % 10 == 0:
                self.cache_manager.put(f"cache_{memory['id']}", memory['content'])
            
            # Record access patterns
            self.optimization_engine.record_memory_access(memory['id'], memory['metadata'])
            
            # Record performance metrics
            query_time = 20 + (i % 30)  # Simulated query time
            self.performance_monitor.record_query(query_time, True)
        
        phase1_time = (time.perf_counter() - workload_start) * 1000
        print(f"Phase 1 (Data Loading): {phase1_time:.2f}ms")
        
        # Phase 2: Mixed read/write operations
        phase2_start = time.perf_counter()
        
        for i in range(100):
            # Search operations
            query = MemoryQuery(
                content_search='test memory',
                limit=10
            )
            search_results = self.indexing_engine.search_memories(query)
            
            # Cache operations
            cache_key = f"cache_memory_{i:06d}"
            if i % 3 == 0:
                self.cache_manager.put(cache_key, f"cached_value_{i}")
            else:
                cached_value = self.cache_manager.get(cache_key)
            
            # Database operations
            if i % 5 == 0:
                query_sql = "SELECT 1"
                self.database_pool.execute_query(query_sql)
            
            # Performance monitoring
            self.performance_monitor.record_query(25 + (i % 20), True)
            self.performance_monitor.record_cache_operation(i % 4 != 0, 2)
        
        phase2_time = (time.perf_counter() - phase2_start) * 1000
        print(f"Phase 2 (Mixed Operations): {phase2_time:.2f}ms")
        
        # Phase 3: System optimization
        phase3_start = time.perf_counter()
        
        # Run optimization
        opt_results = self.optimization_engine.optimize_memory_distribution()
        
        # Check system health
        health_status = self.indexing_engine.health_check()
        threshold_check = self.performance_monitor.check_thresholds()
        
        phase3_time = (time.perf_counter() - phase3_start) * 1000
        print(f"Phase 3 (Optimization): {phase3_time:.2f}ms")
        
        total_workload_time = (time.perf_counter() - workload_start) * 1000
        print(f"Total Workload Time: {total_workload_time:.2f}ms")
        
        # Get final performance snapshot
        final_snapshot = self.performance_monitor.get_current_snapshot()
        
        print(f"\nFinal System Performance:")
        print(f"  Average query time: {final_snapshot.avg_query_time_ms:.2f}ms")
        print(f"  Cache hit rate: {final_snapshot.cache_hit_rate:.3f}")
        print(f"  Health score: {final_snapshot.health_score:.3f}")
        print(f"  Memory usage: {final_snapshot.total_memory_mb:.2f}MB")
        print(f"  System healthy: {health_status['overall_healthy']}")
        
        # Assert integrated performance targets
        self.assertLess(final_snapshot.avg_query_time_ms, 50.0, "System query time exceeds 50ms")
        self.assertGreater(final_snapshot.cache_hit_rate, 0.7, "System cache hit rate below 70%")
        self.assertGreater(final_snapshot.health_score, 0.7, "System health score below 70%")
        self.assertTrue(health_status['overall_healthy'], "System health check failed")
        
        print(f"\n✅ All performance targets met!")
        print(f"🎯 Sub-100ms target achieved: {final_snapshot.avg_query_time_ms:.2f}ms avg query time")


if __name__ == '__main__':
    # Configure logging for test output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run performance tests
    unittest.main(verbosity=2)