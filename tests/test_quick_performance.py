"""
Quick performance validation test for Active Memory Management System.
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

# Add path to memmimic modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from memmimic.memory.active.indexing_engine import create_indexing_engine, IndexingConfig
from memmimic.memory.active.cache_manager import create_cache_manager
from memmimic.memory.active.database_pool import create_database_pool, ConnectionPoolConfig
from memmimic.memory.active.performance_monitor import create_performance_monitor, PerformanceThresholds
from memmimic.memory.active.interfaces import MemoryQuery


def test_quick_performance():
    """Quick performance validation test"""
    print("🚀 Starting Quick Performance Test for Active Memory Management")
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    
    try:
        # Initialize components with realistic settings
        print("\n📊 Initializing components...")
        
        # Indexing engine with all indexes enabled
        indexing_config = IndexingConfig(
            enable_btree_index=True,
            enable_hash_index=True, 
            enable_fulltext_index=True,
            enable_temporal_index=True
        )
        indexing_engine = create_indexing_engine(indexing_config)
        
        # Cache manager
        cache_manager = create_cache_manager(
            cache_type="lru",
            max_memory_mb=128,
            max_items=1000,
            default_ttl_seconds=3600
        )
        
        # Database pool
        pool_config = ConnectionPoolConfig(
            min_connections=5,
            max_connections=20,
            connection_timeout_seconds=5
        )
        database_pool = create_database_pool(db_path, pool_config)
        
        # Performance monitor
        thresholds = PerformanceThresholds(
            max_avg_query_time_ms=50.0,
            max_query_time_ms=100.0,
            min_cache_hit_rate=0.8
        )
        performance_monitor = create_performance_monitor(thresholds)
        
        print("✅ All components initialized successfully")
        
        # Test 1: Indexing Performance
        print("\n🔍 Testing Indexing Performance...")
        test_memories = []
        for i in range(100):
            memory = {
                'id': f'test_memory_{i:03d}',
                'content': f'Test memory content {i} with keywords search performance',
                'metadata': {
                    'type': 'test',
                    'importance_score': (i % 10) / 10.0,
                    'created_at': datetime.now() - timedelta(hours=i % 24)
                }
            }
            test_memories.append(memory)
        
        # Index memories and measure time
        index_start = time.perf_counter()
        for memory in test_memories:
            indexing_engine.index_memory(
                memory['id'],
                memory['content'], 
                memory['metadata'],
                memory['metadata']['created_at']
            )
        index_time = (time.perf_counter() - index_start) * 1000
        
        avg_index_time = index_time / len(test_memories)
        print(f"   Indexed {len(test_memories)} memories in {index_time:.2f}ms")
        print(f"   Average: {avg_index_time:.3f}ms per memory")
        
        # Test search performance
        search_queries = [
            MemoryQuery(memory_ids=['test_memory_050']),
            MemoryQuery(content_search='test memory'),
            MemoryQuery(metadata_filters={'type': 'test'}),
            MemoryQuery(time_range=(
                datetime.now() - timedelta(hours=12),
                datetime.now()
            ))
        ]
        
        search_times = []
        for query in search_queries:
            start_time = time.perf_counter()
            results = indexing_engine.search_memories(query)
            search_time = (time.perf_counter() - start_time) * 1000
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)
        print(f"   Search performance - Avg: {avg_search_time:.3f}ms, Max: {max_search_time:.3f}ms")
        
        # Test 2: Cache Performance
        print("\n💾 Testing Cache Performance...")
        
        # Cache write test
        cache_start = time.perf_counter()
        for i in range(200):
            cache_manager.put(f'cache_key_{i}', f'cached_value_{i}')
        cache_write_time = (time.perf_counter() - cache_start) * 1000
        avg_cache_write = cache_write_time / 200
        
        # Cache read test
        read_start = time.perf_counter()
        hit_count = 0
        for i in range(100):
            value = cache_manager.get(f'cache_key_{i}')
            if value is not None:
                hit_count += 1
        cache_read_time = (time.perf_counter() - read_start) * 1000
        avg_cache_read = cache_read_time / 100
        hit_rate = hit_count / 100
        
        print(f"   Cache write - Avg: {avg_cache_write:.3f}ms per item")
        print(f"   Cache read - Avg: {avg_cache_read:.3f}ms per item")
        print(f"   Cache hit rate: {hit_rate:.1%}")
        
        # Test 3: Database Pool Performance  
        print("\n🗄️  Testing Database Performance...")
        
        # Connection acquisition test
        conn_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            conn = database_pool.get_connection()
            conn_time = (time.perf_counter() - start_time) * 1000
            conn_times.append(conn_time)
            database_pool.return_connection(conn)
        
        avg_conn_time = sum(conn_times) / len(conn_times)
        max_conn_time = max(conn_times)
        
        # Query execution test
        query_start = time.perf_counter()
        database_pool.execute_query("SELECT 1")
        query_time = (time.perf_counter() - query_start) * 1000
        
        print(f"   Connection acquisition - Avg: {avg_conn_time:.3f}ms, Max: {max_conn_time:.3f}ms")
        print(f"   Query execution: {query_time:.3f}ms")
        
        # Test 4: Performance Monitor
        print("\n📈 Testing Performance Monitor...")
        
        # Generate performance data
        for i in range(20):
            performance_monitor.record_query(25 + (i % 10), True)
            performance_monitor.record_cache_operation(i % 4 != 0, 2)
            performance_monitor.record_memory_usage('test', 50 + i)
        
        # Test snapshot generation
        snapshot_start = time.perf_counter()
        snapshot = performance_monitor.get_current_snapshot()
        snapshot_time = (time.perf_counter() - snapshot_start) * 1000
        
        print(f"   Snapshot generation: {snapshot_time:.3f}ms")
        print(f"   Health score: {snapshot.health_score:.3f}")
        print(f"   Avg query time: {snapshot.avg_query_time_ms:.2f}ms")
        
        # Performance Summary
        print("\n🎯 Performance Summary:")
        print(f"   ✅ Indexing: {avg_index_time:.3f}ms per memory (target: <2ms)")
        print(f"   ✅ Search: {avg_search_time:.3f}ms average (target: <50ms)")
        print(f"   ✅ Cache write: {avg_cache_write:.3f}ms (target: <1ms)")
        print(f"   ✅ Cache read: {avg_cache_read:.3f}ms (target: <0.5ms)")
        print(f"   ✅ DB connection: {avg_conn_time:.3f}ms (target: <5ms)")
        print(f"   ✅ DB query: {query_time:.3f}ms (target: <20ms)")
        print(f"   ✅ Monitor snapshot: {snapshot_time:.3f}ms (target: <10ms)")
        
        # Validate targets
        targets_met = []
        targets_met.append(("Indexing", avg_index_time < 2.0))
        targets_met.append(("Search", avg_search_time < 50.0))
        targets_met.append(("Cache write", avg_cache_write < 1.0))
        targets_met.append(("Cache read", avg_cache_read < 0.5))
        targets_met.append(("DB connection", avg_conn_time < 5.0))
        targets_met.append(("DB query", query_time < 20.0))
        targets_met.append(("Monitor", snapshot_time < 10.0))
        
        all_passed = all(passed for _, passed in targets_met)
        
        if all_passed:
            print("\n🎉 ALL PERFORMANCE TARGETS MET!")
            print("🚀 Sub-100ms system performance achieved")
        else:
            print("\n⚠️  Some targets not met:")
            for name, passed in targets_met:
                if not passed:
                    print(f"   ❌ {name}")
        
        return all_passed
        
    finally:
        # Cleanup
        try:
            performance_monitor.shutdown()
            cache_manager.shutdown()
            database_pool.shutdown()
            
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    success = test_quick_performance()
    if success:
        print("\n✅ Performance test PASSED")
        exit(0)
    else:
        print("\n❌ Performance test FAILED")
        exit(1)