"""Error Handling and Recovery Tests

Tests for enhanced error handling including:
- Exception handling decorators
- Error recovery mechanisms
- Database connection failures
- Memory corruption handling
- Graceful degradation
- Error metrics and logging
"""

import sys
import os
import tempfile
import asyncio
import sqlite3
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestErrorHandling:
    """Test suite for error handling and recovery."""
    
    def setup_temp_db(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        return self.db_path
    
    def cleanup_temp_db(self):
        """Clean up temporary database."""
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def test_database_connection_errors(self):
        """Test handling of database connection failures."""
        print("🧪 Testing database connection error handling...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            
            # Test 1: Invalid database path
            invalid_path = "/invalid/path/database.db"
            try:
                storage = AMMSStorage(invalid_path)
                # If it doesn't fail immediately, try an operation
                memory = Memory("Test memory", {"test": True})
                memory_id = asyncio.run(storage.store_memory(memory))
                print("   ❌ Should have failed with invalid path")
                return False
            except Exception as e:
                print(f"   ✅ Invalid database path properly handled: {type(e).__name__}")
            
            # Test 2: Database file permissions (read-only)
            os.chmod(db_path, 0o444)  # Read-only
            try:
                storage = AMMSStorage(db_path)
                memory = Memory("Test memory", {"test": True})
                # This might work for read operations but fail for writes
                try:
                    memory_id = asyncio.run(storage.store_memory(memory))
                except Exception as e:
                    print(f"   ✅ Read-only database error handled: {type(e).__name__}")
            except Exception as e:
                print(f"   ✅ Permission error handled during initialization: {type(e).__name__}")
            finally:
                os.chmod(db_path, 0o644)  # Restore permissions
            
            # Test 3: Connection pool exhaustion
            storage = AMMSStorage(db_path, pool_size=2)  # Small pool
            
            async def exhaust_pool():
                """Try to exhaust connection pool."""
                tasks = []
                for i in range(5):  # More tasks than pool size
                    memory = Memory(f"Pool test {i}", {"pool_test": i})
                    tasks.append(storage.store_memory(memory))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            results = asyncio.run(exhaust_pool())
            
            # Check if any operations succeeded despite pool pressure
            successes = [r for r in results if not isinstance(r, Exception)]
            errors = [r for r in results if isinstance(r, Exception)]
            
            print(f"   📊 Pool stress test: {len(successes)} successes, {len(errors)} errors")
            
            # Check metrics were updated
            stats = storage.get_stats()
            metrics = stats.get('metrics', {})
            if 'pool_misses' in metrics and metrics['pool_misses'] > 0:
                print(f"   ✅ Pool exhaustion tracked: {metrics['pool_misses']} misses")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Database connection error test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()
    
    def test_data_corruption_handling(self):
        """Test handling of corrupted data."""
        print("🧪 Testing data corruption handling...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            import json
            
            storage = AMMSStorage(db_path)
            
            # Store some normal data first
            memory = Memory("Normal memory", {"type": "normal"})
            memory_id = asyncio.run(storage.store_memory(memory))
            print("   ✅ Normal memory stored")
            
            asyncio.run(storage.close())
            
            # Directly corrupt the database by inserting invalid JSON metadata
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert corrupted metadata (invalid JSON)
            corrupted_metadata = "{ invalid json structure, missing quotes: true"
            cursor.execute("""
                INSERT INTO memories (content, metadata, cxd_function, memory_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("Corrupted memory", corrupted_metadata, "DATA", "test", "2024-01-01 12:00:00"))
            
            conn.commit()
            conn.close()
            print("   ✅ Corrupted data inserted directly")
            
            # Test that storage can handle corrupted data
            storage = AMMSStorage(db_path)
            
            # Test retrieval with corrupted data present
            all_memories = asyncio.run(storage.get_all_memories(limit=10))
            
            # Should have at least the normal memory, corrupted one might be skipped
            normal_memories = [m for m in all_memories if m.content == "Normal memory"]
            assert len(normal_memories) >= 1, "Should find normal memory"
            print(f"   ✅ Retrieved {len(all_memories)} memories despite corruption")
            
            # Test search with corrupted data
            search_results = asyncio.run(storage.search_memories("memory", limit=5))
            print(f"   ✅ Search returned {len(search_results)} results with corrupted data present")
            
            # Check error metrics
            stats = storage.get_stats()
            metrics = stats.get('metrics', {})
            if 'failed_operations' in metrics:
                print(f"   📊 Failed operations tracked: {metrics['failed_operations']}")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Data corruption handling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()
    
    def test_async_error_propagation(self):
        """Test async error handling and propagation."""
        print("🧪 Testing async error propagation...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            
            storage = AMMSStorage(db_path)
            
            # Test 1: Invalid memory data
            try:
                invalid_memory = Memory(None, None)  # Invalid content
                memory_id = await storage.store_memory(invalid_memory)
                print("   ❌ Should have failed with invalid memory data")
                return False
            except Exception as e:
                print(f"   ✅ Invalid memory data error: {type(e).__name__}")
            
            # Test 2: Search with invalid parameters
            try:
                results = await storage.search_memories("", limit=-1)  # Invalid limit
                # This might succeed with corrected parameters
                print(f"   ✅ Search handled invalid parameters gracefully: {len(results)} results")
            except Exception as e:
                print(f"   ✅ Invalid search parameters error: {type(e).__name__}")
            
            # Test 3: Concurrent operations with one failing
            async def mixed_operations():
                """Mix of valid and invalid operations."""
                tasks = []
                
                # Valid operations
                for i in range(3):
                    memory = Memory(f"Valid memory {i}", {"valid": True, "index": i})
                    tasks.append(storage.store_memory(memory))
                
                # Invalid operation (simulate by passing wrong type)
                try:
                    tasks.append(storage.retrieve_memory(12345))  # Invalid ID type
                except:
                    pass
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            results = await mixed_operations()
            
            # Analyze results
            successes = [r for r in results if not isinstance(r, Exception) and r is not None]
            failures = [r for r in results if isinstance(r, Exception)]
            
            print(f"   📊 Mixed operations: {len(successes)} successes, {len(failures)} failures")
            
            # Check that valid operations succeeded despite failures
            if len(successes) >= 3:
                print("   ✅ Valid operations succeeded despite concurrent failures")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Async error propagation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()
    
    def test_graceful_degradation(self):
        """Test graceful degradation under resource constraints."""
        print("🧪 Testing graceful degradation...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            
            # Create storage with minimal resources
            storage = AMMSStorage(db_path, pool_size=1)  # Minimal pool
            
            # Test 1: High load scenario
            print("   📊 Testing high load scenario...")
            
            async def high_load_test():
                """Simulate high load."""
                tasks = []
                for i in range(20):  # Many concurrent operations
                    memory = Memory(f"Load test memory {i}", {"load_test": True, "batch": i // 5})
                    tasks.append(storage.store_memory(memory))
                
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                return results, end_time - start_time
            
            results, duration = await high_load_test()
            
            # Analyze performance under load
            successes = [r for r in results if not isinstance(r, Exception)]
            errors = [r for r in results if isinstance(r, Exception)]
            
            success_rate = len(successes) / len(results) * 100
            print(f"   📊 High load results: {len(successes)}/{len(results)} succeeded ({success_rate:.1f}%)")
            print(f"   ⏱️ Total time: {duration:.2f}s ({duration/len(results)*1000:.1f}ms avg)")
            
            # Should maintain reasonable success rate even under stress
            if success_rate >= 80:  # 80% success rate acceptable under stress
                print("   ✅ Maintained good success rate under load")
            else:
                print(f"   ⚠️ Success rate under load: {success_rate:.1f}% (concerning)")
            
            # Test 2: Memory pressure simulation
            print("   📊 Testing memory pressure handling...")
            
            # Try to create many large memories
            large_memories = []
            for i in range(10):
                large_content = f"Large memory {i} " + "x" * 1000  # 1KB+ content
                large_metadata = {f"key_{j}": f"value_{j}" * 10 for j in range(50)}  # Large metadata
                memory = Memory(large_content, large_metadata)
                
                try:
                    memory_id = await storage.store_memory(memory)
                    large_memories.append(memory_id)
                except Exception as e:
                    print(f"   ⚠️ Large memory {i} failed: {type(e).__name__}")
            
            print(f"   📊 Successfully stored {len(large_memories)}/10 large memories")
            
            # Test that system still responds after memory pressure
            test_memory = Memory("Test after pressure", {"test": "responsiveness"})
            response_test_id = await storage.store_memory(test_memory)
            
            if response_test_id:
                print("   ✅ System responsive after memory pressure")
            
            # Check final metrics
            stats = storage.get_stats()
            metrics = stats.get('metrics', {})
            print(f"   📊 Final metrics: {metrics.get('total_operations', 0)} total ops, "
                  f"{metrics.get('failed_operations', 0)} failed")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Graceful degradation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()
    
    def test_error_metrics_tracking(self):
        """Test error metrics tracking and reporting."""
        print("🧪 Testing error metrics tracking...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            
            storage = AMMSStorage(db_path)
            
            # Get initial metrics
            initial_stats = storage.get_stats()
            initial_metrics = initial_stats.get('metrics', {})
            initial_failures = initial_metrics.get('failed_operations', 0)
            
            print(f"   📊 Initial failed operations: {initial_failures}")
            
            # Perform successful operations
            success_count = 5
            for i in range(success_count):
                memory = Memory(f"Success test {i}", {"test": "success"})
                memory_id = await storage.store_memory(memory)
                assert memory_id is not None
            
            print(f"   ✅ Completed {success_count} successful operations")
            
            # Attempt operations that should fail
            failure_attempts = []
            
            # Invalid retrievals
            for invalid_id in ["", "invalid", None, 12345]:
                try:
                    result = await storage.retrieve_memory(invalid_id)
                    # None result is acceptable, not necessarily a failure
                except Exception as e:
                    failure_attempts.append(f"Invalid ID {invalid_id}: {type(e).__name__}")
            
            print(f"   📊 Attempted {len(failure_attempts)} operations that may fail")
            
            # Get final metrics
            final_stats = storage.get_stats()
            final_metrics = final_stats.get('metrics', {})
            
            # Check that metrics are being tracked
            required_metrics = [
                'total_operations', 'successful_operations', 'failed_operations',
                'avg_response_time_ms', 'pool_hits', 'pool_misses'
            ]
            
            for metric in required_metrics:
                if metric in final_metrics:
                    print(f"   📊 {metric}: {final_metrics[metric]}")
                else:
                    print(f"   ⚠️ Missing metric: {metric}")
            
            # Verify metrics make sense
            total_ops = final_metrics.get('total_operations', 0)
            successful_ops = final_metrics.get('successful_operations', 0)
            failed_ops = final_metrics.get('failed_operations', 0)
            
            if total_ops >= successful_ops and successful_ops >= success_count:
                print("   ✅ Metrics tracking appears accurate")
            else:
                print(f"   ⚠️ Metrics may be inaccurate: total={total_ops}, success={successful_ops}, expected>={success_count}")
            
            # Test metrics reset/persistence
            storage_stats = storage.get_stats()
            if 'uptime_seconds' in storage_stats:
                print(f"   📊 Storage uptime: {storage_stats['uptime_seconds']:.2f}s")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Error metrics tracking test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()
    
    def test_recovery_mechanisms(self):
        """Test automatic recovery mechanisms."""
        print("🧪 Testing recovery mechanisms...")
        
        db_path = self.setup_temp_db()
        try:
            from memmimic.memory.storage.amms_storage import AMMSStorage, Memory
            
            storage = AMMSStorage(db_path)
            
            # Store some data
            memory = Memory("Recovery test", {"test": "recovery"})
            memory_id = await storage.store_memory(memory)
            print("   ✅ Initial data stored")
            
            # Close storage
            asyncio.run(storage.close())
            
            # Simulate recovery by reopening
            storage = AMMSStorage(db_path)
            
            # Verify data survived "crash" and recovery
            recovered_memory = await storage.retrieve_memory(memory_id)
            assert recovered_memory is not None
            assert recovered_memory.content == "Recovery test"
            assert recovered_memory.metadata["test"] == "recovery"
            print("   ✅ Data recovered after restart")
            
            # Test that connection pool is reinitialized
            stats = storage.get_stats()
            pool_info = stats.get('connection_pool', {})
            assert pool_info.get('pool_size', 0) > 0
            print(f"   ✅ Connection pool reinitialized: {pool_info.get('pool_size')} connections")
            
            # Test that metrics are reset but functionality works
            new_memory = Memory("Post-recovery test", {"test": "post_recovery"})
            new_memory_id = await storage.store_memory(new_memory)
            assert new_memory_id is not None
            print("   ✅ Normal operations work after recovery")
            
            # Test search functionality after recovery
            search_results = await storage.search_memories("recovery", limit=5)
            assert len(search_results) >= 1
            print(f"   ✅ Search works after recovery: {len(search_results)} results")
            
            asyncio.run(storage.close())
            return True
            
        except Exception as e:
            print(f"   ❌ Recovery mechanisms test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_temp_db()


def run_error_handling_tests():
    """Run all error handling tests."""
    print("🚀 Running Error Handling and Recovery Tests")
    print("=" * 50)
    
    test_suite = TestErrorHandling()
    
    tests = [
        ("Database Connection Errors", test_suite.test_database_connection_errors),
        ("Data Corruption Handling", test_suite.test_data_corruption_handling),
        ("Async Error Propagation", test_suite.test_async_error_propagation),
        ("Graceful Degradation", test_suite.test_graceful_degradation),
        ("Error Metrics Tracking", test_suite.test_error_metrics_tracking),
        ("Recovery Mechanisms", test_suite.test_recovery_mechanisms),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL ERROR HANDLING TESTS PASSED!")
        return 0
    else:
        print("💥 SOME ERROR HANDLING TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_error_handling_tests())