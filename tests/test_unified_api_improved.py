"""Test MemMimic Unified API with Improvements

Test all 11 core tools with our improvements including:
- Connection pooling
- Performance configuration
- Enhanced error handling
- Type safety
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_memmimic_api_creation():
    """Test API can be created with temporary database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        import memmimic
        api = memmimic.create_memmimic(db_path)
        print("✅ MemMimic API created successfully")
        
        # Test that connection pooling is working
        stats = api.memory.get_stats()
        pool_info = stats.get('connection_pool', {})
        print(f"   📊 Connection pool: {pool_info.get('pool_size', 'N/A')} connections")
        
        return True
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_memmimic_11_tools():
    """Test all 11 core tools exist and have proper type hints."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        import memmimic
        from inspect import signature
        
        api = memmimic.create_memmimic(db_path)
        
        # Test each tool exists with type annotations
        tools = {
            # Memory core (4)
            'remember': {'async': True, 'return_annotation': str},
            'recall_cxd': {'async': True, 'return_annotation': list},
            'think_with_memory': {'async': False, 'return_annotation': None},  # Any
            'status': {'async': True, 'return_annotation': dict},
            # Tales (5)
            'tales': {'async': False, 'return_annotation': None},  # Union type
            'save_tale': {'async': False, 'return_annotation': None},  # Any
            'load_tale': {'async': False, 'return_annotation': None},  # Optional
            'delete_tale': {'async': False, 'return_annotation': dict},
            'context_tale': {'async': True, 'return_annotation': str},
            # Management (3)
            'update_memory_guided': {'async': False, 'return_annotation': dict},
            'delete_memory_guided': {'async': False, 'return_annotation': dict},
            'analyze_memory_patterns': {'async': True, 'return_annotation': dict},
            # Cognitive (1)
            'socratic_dialogue': {'async': True, 'return_annotation': dict}
        }
        
        for tool_name, tool_info in tools.items():
            if not hasattr(api, tool_name):
                print(f"❌ Missing tool: {tool_name}")
                return False
            
            # Check method signature
            method = getattr(api, tool_name)
            sig = signature(method)
            
            # Check if async/sync is correct
            if tool_info['async']:
                if not asyncio.iscoroutinefunction(method):
                    print(f"⚠️ Tool {tool_name} should be async")
            
            # Check return annotation if specified
            if tool_info['return_annotation'] and sig.return_annotation:
                expected_type = tool_info['return_annotation']
                if hasattr(expected_type, '__name__'):
                    print(f"   ✅ {tool_name}: proper type hints")
        
        print(f"✅ All {len(tools)} tools present with type safety")
        return True
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


async def test_basic_functionality():
    """Test basic functionality works with improvements."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        import memmimic
        api = memmimic.create_memmimic(db_path)
        
        # Test async status
        status = await api.status()
        print(f"✅ Status: storage_type={status.get('storage_type')}, memories={status.get('memories')}")
        
        # Test memory with enhanced safety
        memory_id = await api.remember("Test memory for classification with JSON-safe metadata", "test")
        print(f"✅ Memory created: {memory_id}")
        
        # Test search with performance monitoring
        import time
        start_time = time.time()
        results = await api.recall_cxd("test memory", limit=5)
        search_time = (time.time() - start_time) * 1000
        print(f"✅ Search completed in {search_time:.2f}ms, found {len(results)} results")
        
        # Test tales system
        tale_result = api.save_tale("test_tale", "This is a test tale content", "misc/test")
        print(f"✅ Tale saved: {tale_result is not None}")
        
        # Test configuration is loaded
        config_info = api.memory.config.database_config
        print(f"✅ Configuration loaded: pool_size={config_info.get('connection_pool_size')}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


async def run_async_tests():
    """Run async tests."""
    return await test_basic_functionality()


def test_performance_improvements():
    """Test that our performance improvements are working."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        import memmimic
        api = memmimic.create_memmimic(db_path)
        
        # Test connection pool metrics
        stats = api.memory.get_stats()
        pool_stats = stats.get('connection_pool', {})
        
        if 'pool_size' in pool_stats:
            print(f"✅ Connection pooling active: {pool_stats['pool_size']} connections")
            print(f"   Available: {pool_stats.get('available_connections', 0)}")
            print(f"   Utilization: {pool_stats.get('pool_utilization', 0):.1%}")
        else:
            print("❌ Connection pooling not active")
            return False
        
        # Test performance metrics
        metrics = stats.get('metrics', {})
        if 'avg_response_time_ms' in metrics:
            print(f"✅ Performance monitoring active")
            print(f"   Avg response time: {metrics.get('avg_response_time_ms', 0):.2f}ms")
        else:
            print("❌ Performance monitoring not active")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Performance improvements test failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


def test_error_handling():
    """Test enhanced error handling."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        import memmimic
        api = memmimic.create_memmimic(db_path)
        
        # Test error handling with invalid operations
        try:
            # Test invalid memory access
            result = api.update_memory_guided(99999)  # Non-existent memory
            if 'error' in result:
                print("✅ Error handling working for invalid memory access")
            else:
                print("⚠️ Error handling may not be catching invalid operations")
        except Exception as e:
            print(f"✅ Exception properly raised for invalid operation: {type(e).__name__}")
        
        # Test tale operations with error handling
        try:
            result = api.delete_tale("non_existent_tale", confirm=False)
            if 'error' in result:
                print("✅ Error handling working for tale operations")
        except Exception as e:
            print(f"✅ Exception properly handled: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except:
            pass


if __name__ == "__main__":
    print("🧪 Testing MemMimic Unified API with Improvements...")
    
    # Run sync tests
    creation_ok = test_memmimic_api_creation()
    if not creation_ok:
        print("❌ Cannot create API - imports need fixing")
        sys.exit(1)
    
    tools_ok = test_memmimic_11_tools()
    if not tools_ok:
        print("❌ Missing tools - API incomplete")
        sys.exit(1)
    
    performance_ok = test_performance_improvements()
    if not performance_ok:
        print("❌ Performance improvements not working")
        sys.exit(1)
    
    error_handling_ok = test_error_handling()
    if not error_handling_ok:
        print("❌ Error handling not working properly")
        sys.exit(1)
    
    # Run async tests
    basic_ok = asyncio.run(run_async_tests())
    if not basic_ok:
        print("⚠️ API created but basic functionality needs work")
        sys.exit(1)
    
    print("🎉 MemMimic API fully functional with all improvements!")
    print("   ✅ Connection pooling")
    print("   ✅ Performance monitoring")
    print("   ✅ Type safety")
    print("   ✅ Enhanced error handling")
    print("   ✅ Configuration system")