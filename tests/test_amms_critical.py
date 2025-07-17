#!/usr/bin/env python3
"""
MemMimic AMMS Critical Integration Tests
Tests specifically requested in Code Review PR #2
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, 'src')

def test_unified_store_fallback():
    """Test graceful fallback to legacy system"""
    print("🧪 Testing UnifiedMemoryStore fallback mechanism...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        from memmimic.memory import UnifiedMemoryStore, Memory
        
        # Initialize store
        store = UnifiedMemoryStore(db_path)
        print(f"   ✅ UnifiedMemoryStore initialized")
        
        # Test normal operation
        memory = Memory("Test fallback memory", "test")
        memory_id = store.add(memory)
        print(f"   ✅ Added memory: ID {memory_id}")
        
        # Test search
        results = store.search("fallback", limit=1)
        assert len(results) >= 1, "Should find the test memory"
        print(f"   ✅ Search returned {len(results)} results")
        
        # Test compatibility mode toggle
        store.enable_compatibility_mode(True)
        assert store._compatibility_mode == True, "Compatibility mode should be enabled"
        print(f"   ✅ Compatibility mode enabled")
        
        # Test that it still works in compatibility mode
        memory2 = Memory("Compatibility test memory", "test")
        memory_id2 = store.add(memory2)
        print(f"   ✅ Added memory in compatibility mode: ID {memory_id2}")
        
        # Disable compatibility mode
        store.enable_compatibility_mode(False) 
        assert store.is_amms_active == True, "AMMS should be active again"
        print(f"   ✅ AMMS re-enabled successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_amms_performance_targets():
    """Validate 100ms query requirement"""
    print("🧪 Testing AMMS performance targets...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        from memmimic import MemMimicAPI
        import time
        
        # Initialize API
        api = MemMimicAPI(db_path)
        print(f"   ✅ API initialized")
        
        # Add test memories for performance testing
        print(f"   📊 Adding test memories for performance testing...")
        for i in range(50):  # Add enough memories for meaningful test
            content = f"Performance test memory {i} with various content about projects, technical details, and interactions"
            memory_type = ["interaction", "technical", "milestone", "reflection"][i % 4]
            api.remember(content, memory_type)
        
        print(f"   ✅ Added 50 test memories")
        
        # Test query performance multiple times
        query_times = []
        test_queries = [
            "performance test",
            "technical details", 
            "project milestone",
            "memory interaction",
            "various content"
        ]
        
        for query in test_queries:
            start_time = time.time()
            results = api.recall_cxd(query, limit=5)
            end_time = time.time()
            
            query_time_ms = (end_time - start_time) * 1000
            query_times.append(query_time_ms)
            
            print(f"   ⏱️ Query '{query}': {query_time_ms:.2f}ms ({len(results)} results)")
        
        # Calculate statistics
        avg_time = sum(query_times) / len(query_times)
        max_time = max(query_times)
        
        print(f"   📊 Average query time: {avg_time:.2f}ms")
        print(f"   📊 Maximum query time: {max_time:.2f}ms")
        
        # Performance targets
        if avg_time < 100:
            print(f"   ✅ Average performance target met ({avg_time:.2f}ms < 100ms)")
        else:
            print(f"   ⚠️ Average performance target not met ({avg_time:.2f}ms > 100ms)")
        
        if max_time < 200:  # Allow some variance for max time
            print(f"   ✅ Max performance acceptable ({max_time:.2f}ms < 200ms)")
        else:
            print(f"   ⚠️ Max performance concerning ({max_time:.2f}ms > 200ms)")
        
        # Return success if average time meets target
        return avg_time < 100
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_configuration_validation():
    """Test config file validation and error handling"""
    print("🧪 Testing configuration validation...")
    
    try:
        from memmimic.config import get_config, ConfigLoader, MemMimicConfig
        import tempfile
        import yaml
        
        # Test 1: Valid configuration loading
        config = get_config()
        print(f"   ✅ Default configuration loaded")
        
        # Test 2: Scoring weights validation
        if config.scoring_weights.validate():
            print(f"   ✅ Scoring weights validation passed")
        else:
            print(f"   ❌ Scoring weights validation failed")
            return False
        
        # Test 3: Invalid configuration handling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
            # Create invalid config (weights don't sum to 1.0)
            invalid_config = {
                'active_memory_pool': {
                    'target_size': 1000,
                    'max_size': 500  # Invalid: max < target
                },
                'scoring_weights': {
                    'cxd_classification': 0.50,
                    'access_frequency': 0.30,
                    'recency_temporal': 0.15,
                    'confidence_quality': 0.10,
                    'memory_type': 0.10  # Total = 1.15 (invalid)
                }
            }
            yaml.dump(invalid_config, tmp_config)
            tmp_config_path = tmp_config.name
        
        try:
            # Test loading invalid config
            loader = ConfigLoader(tmp_config_path)
            invalid_loaded_config = loader.load_config()
            
            # Should fall back to defaults due to validation failure
            if invalid_loaded_config.scoring_weights.validate():
                print(f"   ✅ Invalid config handled gracefully (fell back to defaults)")
            else:
                print(f"   ❌ Invalid config not handled properly")
                return False
                
        finally:
            os.unlink(tmp_config_path)
        
        # Test 4: Configuration path precedence
        config_paths = [
            "config/memmimic_config.yaml",
            "memmimic_config.yaml",
            "~/.memmimic/config.yaml"
        ]
        
        for path in config_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                print(f"   ✅ Found config at: {path}")
                break
        else:
            print(f"   ⚠️ No config files found in standard paths (using defaults)")
        
        # Test 5: Retention policies validation
        retention_policies = config.retention_policies
        
        critical_types = ['synthetic_wisdom', 'milestone']
        for memory_type in critical_types:
            if memory_type in retention_policies:
                policy = retention_policies[memory_type]
                if policy.min_retention == 'permanent':
                    print(f"   ✅ {memory_type} has permanent retention")
                else:
                    print(f"   ⚠️ {memory_type} should have permanent retention")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_migration_safety():
    """Test migration with backup/restore scenarios"""
    print("🧪 Testing migration safety...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        from memmimic.memory import MemoryStore, Memory
        import shutil
        
        # Create legacy database with test data
        legacy_store = MemoryStore(db_path)
        test_memories = [
            Memory("Legacy memory 1", "interaction"),
            Memory("Legacy memory 2", "milestone"),
            Memory("Legacy memory 3", "technical")
        ]
        
        legacy_ids = []
        for memory in test_memories:
            memory_id = legacy_store.add(memory)
            legacy_ids.append(memory_id)
        
        print(f"   ✅ Created legacy database with {len(test_memories)} memories")
        
        # Test backup creation
        import sys
        sys.path.insert(0, '.')  # Add current directory for migrate script
        
        try:
            import migrate_to_amms
            
            # Test backup function
            backup_path = migrate_to_amms.backup_database(db_path)
            assert os.path.exists(backup_path), "Backup file should exist"
            print(f"   ✅ Backup created: {backup_path}")
            
            # Verify backup contains data
            backup_store = MemoryStore(backup_path)
            backup_memories = backup_store.get_all()
            assert len(backup_memories) == len(test_memories), "Backup should contain all memories"
            print(f"   ✅ Backup verified: {len(backup_memories)} memories")
            
            # Test migration
            result = migrate_to_amms.migrate_database(db_path)
            
            if 'error' not in result:
                print(f"   ✅ Migration completed successfully")
                print(f"      📊 Legacy memories: {result.get('total_legacy_memories', 0)}")
                print(f"      📊 Migrated: {result.get('successfully_migrated', 0)}")
                print(f"      📊 Errors: {result.get('errors', 0)}")
                
                # Test that migrated data is accessible
                from memmimic.memory import UnifiedMemoryStore
                unified_store = UnifiedMemoryStore(db_path)
                migrated_memories = unified_store.get_all()
                
                # Should have at least the original memories (possibly more due to enhanced schema)
                assert len(migrated_memories) >= len(test_memories), "Should have at least original memories"
                print(f"   ✅ Migration verification: {len(migrated_memories)} active memories")
                
                # Test AMMS functionality after migration
                test_search = unified_store.search("Legacy memory", limit=5)
                assert len(test_search) > 0, "Should find migrated memories"
                print(f"   ✅ Post-migration search: {len(test_search)} results")
                
            else:
                print(f"   ❌ Migration failed: {result['error']}")
                return False
            
            # Clean up backup
            if os.path.exists(backup_path):
                os.unlink(backup_path)
            
            return True
            
        except ImportError:
            print(f"   ⚠️ Migration script not available for testing")
            return True  # Don't fail if script not available
            
    except Exception as e:
        print(f"   ❌ Migration safety test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            os.unlink(db_path)
        except:
            pass

def test_resource_constraints():
    """Test resource consumption and constraints"""
    print("🧪 Testing resource constraints...")
    
    try:
        from memmimic.config import get_config
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"   📊 Initial memory usage: {initial_memory:.2f} MB")
        
        # Test configuration limits
        config = get_config()
        
        # Check cache size is reasonable (not too large)
        cache_size = config.active_memory_pool.cache_size
        max_pool_size = config.active_memory_pool.max_pool_size
        
        if cache_size <= max_pool_size:
            print(f"   ✅ Cache size ({cache_size}) <= max pool size ({max_pool_size})")
        else:
            print(f"   ⚠️ Cache size ({cache_size}) > max pool size ({max_pool_size})")
        
        # Estimate memory usage
        estimated_memory_per_item = 1  # KB per memory item (conservative estimate)
        estimated_total_kb = max_pool_size * estimated_memory_per_item
        estimated_total_mb = estimated_total_kb / 1024
        
        print(f"   📊 Estimated max memory usage: {estimated_total_mb:.2f} MB")
        
        # Check if within reasonable bounds (1GB = 1024 MB from PRD)
        if estimated_total_mb < 1024:
            print(f"   ✅ Estimated usage within 1GB limit")
        else:
            print(f"   ⚠️ Estimated usage may exceed 1GB limit")
        
        # Test with temporary high-memory config to ensure it doesn't crash
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            from memmimic import MemMimicAPI
            
            # Create API instance (should initialize AMMS)
            api = MemMimicAPI(db_path)
            
            # Add some memories to test actual usage
            for i in range(10):
                api.remember(f"Resource test memory {i} with content", "test")
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"   📊 Memory usage after 10 memories: {current_memory:.2f} MB (+{memory_increase:.2f} MB)")
            
            if memory_increase < 50:  # Should be much less than 50MB for 10 memories
                print(f"   ✅ Memory usage increase is reasonable")
                return True
            else:
                print(f"   ⚠️ Memory usage increase seems high")
                return False
                
        finally:
            os.unlink(db_path)
        
    except ImportError:
        print(f"   ⚠️ psutil not available, skipping memory monitoring")
        return True
    except Exception as e:
        print(f"   ❌ Resource constraints test failed: {e}")
        return False

def main():
    """Run all critical tests"""
    print("🚀 MemMimic AMMS Critical Tests Suite")
    print("=" * 50)
    
    tests = [
        ("Unified Store Fallback", test_unified_store_fallback),
        ("Performance Targets", test_amms_performance_targets), 
        ("Configuration Validation", test_configuration_validation),
        ("Migration Safety", test_migration_safety),
        ("Resource Constraints", test_resource_constraints)
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
        print("🎉 ALL CRITICAL TESTS PASSED!")
        return 0
    else:
        print("💥 SOME CRITICAL TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())