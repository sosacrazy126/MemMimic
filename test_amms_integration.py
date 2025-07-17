#!/usr/bin/env python3
"""
MemMimic AMMS Integration Test
Quick test to verify Active Memory Management System is working
"""

import sys
import os
import tempfile
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, 'src')

def test_amms_integration():
    """Test that AMMS is properly integrated with MemMimic API"""
    
    print("🧪 Testing AMMS Integration...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        # Test 1: Import and initialize with AMMS
        print("1️⃣ Testing import and initialization...")
        from memmimic import MemMimicAPI
        
        api = MemMimicAPI(db_path=db_path)
        print(f"   ✅ MemMimicAPI initialized")
        print(f"   📊 Memory store type: {type(api.memory)}")
        print(f"   🔄 AMMS Active: {getattr(api.memory, 'is_amms_active', 'Unknown')}")
        
        # Test 2: Store memories
        print("\n2️⃣ Testing memory storage...")
        memory_ids = []
        
        test_memories = [
            ("This is a test interaction", "interaction"),
            ("Important project milestone achieved", "milestone"), 
            ("Technical documentation about API design", "technical"),
            ("Reflection on memory management patterns", "reflection"),
            ("Synthetic wisdom about AI consciousness", "synthetic_wisdom")
        ]
        
        for content, memory_type in test_memories:
            memory_id = api.remember(content, memory_type)
            memory_ids.append(memory_id)
            print(f"   ✅ Stored {memory_type}: ID {memory_id}")
        
        print(f"   📊 Total memories stored: {len(memory_ids)}")
        
        # Test 3: Search and recall
        print("\n3️⃣ Testing memory recall...")
        
        search_queries = [
            "project milestone",
            "technical documentation", 
            "AI consciousness",
            "memory patterns"
        ]
        
        for query in search_queries:
            results = api.recall_cxd(query, limit=3)
            print(f"   🔍 Query '{query}': {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"      {i+1}. {result.type}: {result.content[:50]}...")
        
        # Test 4: System status with AMMS info
        print("\n4️⃣ Testing system status...")
        status = api.status()
        
        print(f"   📊 Total memories: {status.get('memories', 0)}")
        print(f"   🔄 AMMS Active: {status.get('amms_active', False)}")
        print(f"   🎯 CXD Available: {status.get('cxd_available', False)}")
        
        if 'amms_pool_status' in status:
            pool_status = status['amms_pool_status']
            print(f"   💾 Active pool: {pool_status.get('active', {}).get('count', 0)} memories")
            print(f"   📦 Archived: {pool_status.get('archived', {}).get('count', 0)} memories")
        
        if 'amms_performance' in status:
            perf = status['amms_performance']
            print(f"   ⚡ Avg query time: {perf.get('avg_query_time_ms', 0):.2f}ms")
        
        # Test 5: Enhanced features (if available)
        print("\n5️⃣ Testing enhanced AMMS features...")
        
        if hasattr(api.memory, 'search_with_ranking'):
            enhanced_results = api.memory.search_with_ranking("milestone project", limit=2)
            print(f"   🎯 Enhanced search: {len(enhanced_results)} results with ranking")
            
            for result in enhanced_results:
                score = result.get('score', 0)
                print(f"      Score: {score:.3f} - {result.get('content', '')[:40]}...")
        
        if hasattr(api.memory, 'get_active_pool_status'):
            pool_status = api.memory.get_active_pool_status()
            print(f"   📊 Pool status available: {len(pool_status)} metrics")
        
        print("\n✅ AMMS Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ AMMS Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except:
            pass

def test_config_system():
    """Test configuration system"""
    print("\n🔧 Testing Configuration System...")
    
    try:
        from memmimic.config import get_config
        
        config = get_config()
        print(f"   ✅ Configuration loaded")
        print(f"   📊 Target pool size: {config.active_memory_pool.target_size}")
        print(f"   📊 Max pool size: {config.active_memory_pool.max_pool_size}")
        print(f"   🎯 CXD weight: {config.scoring_weights.cxd_classification}")
        print(f"   ⏰ Stale threshold: {config.cleanup_policies.stale_threshold_days} days")
        
        # Test config validation
        if config.scoring_weights.validate():
            print(f"   ✅ Scoring weights validation: PASSED")
        else:
            print(f"   ❌ Scoring weights validation: FAILED")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MemMimic AMMS Integration Test Suite")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_config_system()
    
    # Test integration 
    integration_ok = test_amms_integration()
    
    print("\n" + "=" * 50)
    if config_ok and integration_ok:
        print("🎉 ALL TESTS PASSED - AMMS Integration Successful!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED - Check output above")
        sys.exit(1)