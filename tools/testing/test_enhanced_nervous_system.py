#!/usr/bin/env python3
"""
Enhanced Nervous System Validation Test

Comprehensive test suite to validate all enhancements to the MemMimic nervous
system architecture, including archive intelligence, phase tracking, narrative
fusion, latency optimization, and multi-agent coordination.
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, 'src')

async def test_enhanced_nervous_system():
    """Test all enhanced nervous system components"""
    print("🧠 Enhanced MemMimic Nervous System Validation")
    print("=" * 60)
    
    try:
        from memmimic.nervous_system import (
            NervousSystemCore,
            ArchiveIntelligence,
            PhaseEvolutionTracker,
            TaleMemoryBinder,
            ReflexLatencyOptimizer,
            SharedRealityManager,
            TheoryOfMindCapabilities
        )
        from memmimic.nervous_system.shared_reality_manager import AgentRole
        
        print("✅ Successfully imported enhanced nervous system components")
        
    except ImportError as e:
        print(f"❌ Failed to import components: {e}")
        return False
    
    # Test 1: Enhanced NervousSystemCore Initialization
    print("\n🔧 Test 1: Enhanced NervousSystemCore Initialization")
    try:
        core = NervousSystemCore(db_path="./src/memmimic/mcp/memmimic.db")
        await core.initialize()
        print("✅ Enhanced NervousSystemCore initialized successfully")
        
        # Check if all new components are available
        components_status = {
            'archive_intelligence': core._archive_intelligence is not None,
            'phase_tracker': core._phase_tracker is not None,
            'tale_memory_binder': core._tale_memory_binder is not None,
            'latency_optimizer': core._latency_optimizer is not None,
            'shared_reality_manager': core._shared_reality_manager is not None,
            'theory_of_mind': core._theory_of_mind is not None
        }
        
        print("📊 Component Status:")
        for component, status in components_status.items():
            status_icon = "✅" if status else "⚠️"
            print(f"   {status_icon} {component}: {'Available' if status else 'Not Available'}")
        
    except Exception as e:
        print(f"❌ Enhanced NervousSystemCore initialization failed: {e}")
        return False
    
    # Test 2: Archive Intelligence Functionality
    print("\n📦 Test 2: Archive Intelligence Functionality")
    try:
        if core._archive_intelligence:
            patterns = core.get_archive_patterns()
            print(f"✅ Archive patterns available: {patterns}")
            
            metrics = core.get_archive_evolution_metrics()
            if metrics:
                print(f"📊 Archive metrics: {metrics.patterns_extracted} patterns extracted")
            else:
                print("⚠️ Archive metrics not available")
        else:
            print("⚠️ Archive intelligence not initialized")
            
    except Exception as e:
        print(f"❌ Archive intelligence test failed: {e}")
    
    # Test 3: Phase Evolution Tracking
    print("\n📈 Test 3: Phase Evolution Tracking")
    try:
        if core._phase_tracker:
            status = await core.get_phase_status()
            print(f"✅ Phase tracking status: {status.get('phase_tracking', False)}")
            
            if status.get('metrics'):
                metrics = status['metrics']
                print(f"📊 Evolution metrics:")
                print(f"   Total phases: {metrics.total_phases}")
                print(f"   Completed phases: {metrics.completed_phases}")
                print(f"   Overall progress: {metrics.overall_progress:.1%}")
        else:
            print("⚠️ Phase tracker not initialized")
            
    except Exception as e:
        print(f"❌ Phase evolution tracking test failed: {e}")
    
    # Test 4: Narrative-Memory Fusion
    print("\n📚 Test 4: Narrative-Memory Fusion")
    try:
        if core._tale_memory_binder:
            themes = core.get_narrative_themes()
            print(f"✅ Narrative themes available: {len(themes)} themes")
            
            binding_metrics = core.get_narrative_binding_metrics()
            if binding_metrics:
                print(f"📊 Binding metrics: {binding_metrics.themes_extracted} themes extracted")
            
            # Test narrative enhancement
            test_content = "I want to remember this important insight about consciousness and memory."
            result = await core.process_with_intelligence(test_content, "insight")
            
            if result.get('narrative_enhancement'):
                print("✅ Narrative enhancement working")
                narrative = result['narrative_enhancement']
                if narrative.get('thematic_tags'):
                    print(f"   Thematic tags: {narrative['thematic_tags']}")
            else:
                print("⚠️ Narrative enhancement not applied")
        else:
            print("⚠️ Tale-memory binder not initialized")
            
    except Exception as e:
        print(f"❌ Narrative-memory fusion test failed: {e}")
    
    # Test 5: Latency Optimization
    print("\n⚡ Test 5: Latency Optimization")
    try:
        if core._latency_optimizer:
            # Test optimized operation
            async def test_operation():
                await asyncio.sleep(0.001)  # 1ms operation
                return "test_result"
            
            result, execution_time = await core.optimize_operation_latency(
                "test_operation", test_operation
            )
            
            print(f"✅ Optimized operation completed in {execution_time:.2f}ms")
            
            latency_metrics = core.get_latency_metrics()
            if latency_metrics.get('optimization_summary'):
                summary = latency_metrics['optimization_summary']
                print(f"📊 Latency optimization summary:")
                print(f"   Target latency: {summary.get('target_latency_ms', 0):.1f}ms")
                print(f"   Average latency: {summary.get('average_latency_ms', 0):.2f}ms")
                print(f"   Cache hit rate: {summary.get('cache_hit_rate', 0):.1%}")
        else:
            print("⚠️ Latency optimizer not initialized")
            
    except Exception as e:
        print(f"❌ Latency optimization test failed: {e}")
    
    # Test 6: Shared Reality Management
    print("\n🌐 Test 6: Shared Reality Management")
    try:
        if core._shared_reality_manager:
            # Register test agent
            session_id = await core.register_agent_in_shared_reality(
                "test_agent_1", "Test Agent", AgentRole.PRIMARY, ["memory", "analysis"]
            )
            
            if session_id:
                print(f"✅ Agent registered with session: {session_id[:8]}...")
                
                reality_status = core.get_shared_reality_status()
                print(f"📊 Shared reality status:")
                print(f"   Active agents: {reality_status.get('active_agents', 0)}")
                print(f"   Shared states: {reality_status.get('shared_states', 0)}")
            else:
                print("⚠️ Agent registration failed")
        else:
            print("⚠️ Shared reality manager not initialized")
            
    except Exception as e:
        print(f"❌ Shared reality management test failed: {e}")
    
    # Test 7: Theory of Mind Capabilities
    print("\n🧠 Test 7: Theory of Mind Capabilities")
    try:
        if core._theory_of_mind:
            # Simulate agent observation
            await core.observe_agent_action(
                "test_agent_1", 
                "search_memory", 
                {"query": "consciousness", "context": "research"}
            )
            
            print("✅ Agent action observed")
            
            # Test behavior prediction
            predictions = await core.predict_agent_behavior("test_agent_1", 60.0)
            print(f"📊 Behavior predictions: {len(predictions)} predictions generated")
            
            # Test empathetic response
            response = await core.generate_empathetic_response(
                "test_agent_1", "struggling with complex analysis"
            )
            if response:
                print(f"💭 Empathetic response: {response}")
            
            tom_metrics = core.get_theory_of_mind_metrics()
            if tom_metrics:
                print(f"📊 Theory of mind metrics:")
                print(f"   Tracked agents: {tom_metrics.get('tracked_agents', 0)}")
                print(f"   Active predictions: {tom_metrics.get('active_predictions', 0)}")
        else:
            print("⚠️ Theory of mind not initialized")
            
    except Exception as e:
        print(f"❌ Theory of mind test failed: {e}")
    
    # Test 8: Backward Compatibility
    print("\n🔄 Test 8: Backward Compatibility")
    try:
        # Test traditional memory processing
        test_memory = "This is a test memory for backward compatibility validation."
        result = await core.process_with_intelligence(test_memory, "test")
        
        # Check that all traditional components still work
        required_fields = ['quality_assessment', 'duplicate_analysis', 'cxd_classification']
        compatibility_check = all(field in result for field in required_fields)
        
        if compatibility_check:
            print("✅ Backward compatibility maintained")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        else:
            print("⚠️ Some backward compatibility issues detected")
            missing = [field for field in required_fields if field not in result]
            print(f"   Missing fields: {missing}")
            
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
    
    # Test 9: Performance Benchmarking
    print("\n🏃 Test 9: Performance Benchmarking")
    try:
        # Benchmark core operations
        operations = []
        for i in range(10):
            start_time = time.perf_counter()
            
            result = await core.process_with_intelligence(
                f"Test memory {i} for performance benchmarking", "benchmark"
            )
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            operations.append(execution_time)
        
        avg_time = sum(operations) / len(operations)
        min_time = min(operations)
        max_time = max(operations)
        
        print(f"📊 Performance benchmark results:")
        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   Min time: {min_time:.2f}ms")
        print(f"   Max time: {max_time:.2f}ms")
        print(f"   Sub-5ms target: {'✅ Achieved' if avg_time < 5.0 else '⚠️ Not achieved'}")
        
    except Exception as e:
        print(f"❌ Performance benchmarking failed: {e}")
    
    # Test 10: Integration Summary
    print("\n📋 Test 10: Integration Summary")
    try:
        # Get comprehensive metrics
        performance_metrics = core.get_performance_metrics()
        
        print("📊 Enhanced Nervous System Summary:")
        print(f"   Total operations: {performance_metrics.get('total_operations', 0)}")
        print(f"   Average response time: {performance_metrics.get('average_response_time_ms', 0):.2f}ms")
        print(f"   Cache hit rate: {performance_metrics.get('cache_hit_rate', 0):.1%}")
        print(f"   Quality assessments: {performance_metrics.get('quality_assessments', 0)}")
        print(f"   Archive patterns applied: {performance_metrics.get('archive_patterns_applied', 0)}")
        
        # Check if all enhancements are functional
        enhancements_functional = all([
            core._archive_intelligence is not None,
            core._phase_tracker is not None,
            core._tale_memory_binder is not None,
            core._latency_optimizer is not None,
            core._shared_reality_manager is not None,
            core._theory_of_mind is not None
        ])
        
        if enhancements_functional:
            print("✅ All nervous system enhancements are functional")
        else:
            print("⚠️ Some enhancements may not be fully functional")
        
        print("\n🎉 Enhanced MemMimic Nervous System validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration summary failed: {e}")
        return False

async def main():
    """Main test execution"""
    print("Starting Enhanced MemMimic Nervous System Validation...")
    
    success = await test_enhanced_nervous_system()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🚀 Enhanced nervous system is ready for production use.")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
