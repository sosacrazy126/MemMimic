"""
Biological Reflex Optimization Test

Tests the temporal memory architecture optimization to achieve <5ms response times
for Remember trigger and validate 100% biological reflex coverage (4/4 triggers).
"""

import asyncio
import time
from typing import Dict, Any, List

from .triggers.unified_interface import UnifiedEnhancedTriggers
from .temporal_memory_manager import get_temporal_memory_manager
from .db_initializer import get_initialized_database
from ..errors import get_error_logger

async def test_remember_biological_reflex_optimization(db_path: str = ":memory:") -> Dict[str, Any]:
    """
    Test Remember trigger optimization with temporal memory architecture.
    
    Validates that 80% of memories go to working tier (<1ms) and 20% to long-term (<3ms),
    achieving overall <5ms biological reflex performance.
    """
    print("🧠 BIOLOGICAL REFLEX OPTIMIZATION TEST: Remember Trigger")
    print("=" * 60)
    
    # Initialize system
    db_init = await get_initialized_database(db_path)
    enhanced_triggers = UnifiedEnhancedTriggers(db_path)
    await enhanced_triggers.initialize()
    
    # Test scenarios with mixed working/long-term content
    test_memories = [
        # Working memory content (should be <1ms)
        ("Task completed successfully", "interaction"),
        ("Working on implementation of feature X", "interaction"), 
        ("Debug session in progress", "interaction"),
        ("Code review feedback received", "interaction"),
        ("Status update: tests passing", "interaction"),
        ("Configuration change applied", "technical"),
        ("Build deployment scheduled", "technical"),
        ("Current session progress noted", "interaction"),
        
        # Long-term memory content (should be <3ms but full processing)
        ("Breakthrough insight: consciousness emerges from pattern recognition", "reflection"),
        ("Research discovery: neural network optimization principles", "milestone"),
        ("Valuable methodology for distributed intelligence systems", "reflection"),
        ("Key principle: biological reflex speed requires intelligent filtering", "milestone")
    ]
    
    print(f"\n⚡ Testing {len(test_memories)} mixed memory scenarios...")
    
    results = {
        'working_memory_results': [],
        'long_term_memory_results': [],
        'performance_summary': {},
        'biological_reflex_achievement': {}
    }
    
    # Test each memory
    for i, (content, memory_type) in enumerate(test_memories, 1):
        print(f"   Test {i:2d}: {content[:40]}...")
        
        start_time = time.perf_counter()
        result = await enhanced_triggers.remember(content, memory_type)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Categorize result
        if result.get('memory_tier') == 'working':
            results['working_memory_results'].append({
                'content_preview': content[:40],
                'processing_time_ms': processing_time,
                'biological_reflex_achieved': processing_time < 5.0,
                'fast_path_used': result.get('fast_path_used', False),
                'expires_at': result.get('expires_at')
            })
            status_icon = "🔄" if processing_time < 1.0 else "⚠️"
        else:
            results['long_term_memory_results'].append({
                'content_preview': content[:40],
                'processing_time_ms': processing_time,
                'biological_reflex_achieved': processing_time < 5.0,
                'intelligence_applied': result.get('intelligence_analysis') is not None,
                'permanent_storage': True
            })
            status_icon = "🧠" if processing_time < 3.0 else "⚠️"
        
        print(f"        {status_icon} {processing_time:.2f}ms - {result.get('memory_tier', 'unknown')} tier")
    
    # Calculate performance summary
    working_times = [r['processing_time_ms'] for r in results['working_memory_results']]
    long_term_times = [r['processing_time_ms'] for r in results['long_term_memory_results']]
    all_times = working_times + long_term_times
    
    results['performance_summary'] = {
        'total_memories_tested': len(test_memories),
        'working_memory_count': len(working_times),
        'long_term_memory_count': len(long_term_times),
        'working_memory_ratio': len(working_times) / len(test_memories),
        'long_term_memory_ratio': len(long_term_times) / len(test_memories),
        'average_working_time_ms': sum(working_times) / len(working_times) if working_times else 0,
        'average_long_term_time_ms': sum(long_term_times) / len(long_term_times) if long_term_times else 0,
        'overall_average_time_ms': sum(all_times) / len(all_times),
        'working_memory_under_1ms': len([t for t in working_times if t < 1.0]),
        'long_term_memory_under_3ms': len([t for t in long_term_times if t < 3.0]),
        'all_memories_under_5ms': len([t for t in all_times if t < 5.0])
    }
    
    # Biological reflex achievement analysis
    biological_reflex_rate = len([t for t in all_times if t < 5.0]) / len(all_times)
    
    results['biological_reflex_achievement'] = {
        'target_response_time_ms': 5.0,
        'achieved_response_time_ms': results['performance_summary']['overall_average_time_ms'],
        'biological_reflex_rate': biological_reflex_rate,
        'biological_reflex_status': 'ACHIEVED' if biological_reflex_rate >= 0.95 else 'PARTIAL',
        'optimization_effective': results['performance_summary']['overall_average_time_ms'] < 5.0,
        'temporal_architecture_working': len(working_times) > len(long_term_times)
    }
    
    # Display results
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"   Working Memory: {len(working_times)} memories, avg {results['performance_summary']['average_working_time_ms']:.2f}ms")
    print(f"   Long-term Memory: {len(long_term_times)} memories, avg {results['performance_summary']['average_long_term_time_ms']:.2f}ms")
    print(f"   Overall Average: {results['performance_summary']['overall_average_time_ms']:.2f}ms")
    print(f"   Biological Reflex Rate: {biological_reflex_rate:.1%}")
    print(f"   Status: {results['biological_reflex_achievement']['biological_reflex_status']}")
    
    success_icon = "✅" if biological_reflex_rate >= 0.95 else "⚠️"
    print(f"\n{success_icon} Remember Trigger Optimization: {'SUCCESS' if biological_reflex_rate >= 0.95 else 'NEEDS TUNING'}")
    
    return results

async def test_complete_system_biological_reflex() -> Dict[str, Any]:
    """
    Test complete system biological reflex performance across all 4 triggers.
    
    Validates 100% biological reflex coverage with optimized Remember trigger.
    """
    print("\n🚀 COMPLETE SYSTEM BIOLOGICAL REFLEX TEST")
    print("=" * 50)
    
    enhanced_triggers = UnifiedEnhancedTriggers(":memory:")
    await enhanced_triggers.initialize()
    
    # Test all 4 triggers
    print("   Testing all 4 nervous system triggers...")
    
    # Run biological reflex demonstration
    reflex_demo = await enhanced_triggers.demonstrate_biological_reflex()
    
    print(f"\n🎯 BIOLOGICAL REFLEX RESULTS:")
    print(f"   Target: {reflex_demo['biological_reflex_target']}")
    
    all_achieved = True
    for trigger_name, results in reflex_demo['test_results'].items():
        status = "✅" if results['biological_reflex_achieved'] else "❌"
        print(f"   {trigger_name.title()}: {results['processing_time_ms']:.2f}ms {status}")
        if not results['biological_reflex_achieved']:
            all_achieved = False
    
    overall = reflex_demo['overall_performance']
    print(f"\n📈 SYSTEM PERFORMANCE:")
    print(f"   Average Time: {overall['average_processing_time_ms']:.2f}ms")
    print(f"   Success Rate: {overall['biological_reflex_success_rate']:.1%}")
    print(f"   Status: {overall['biological_reflex_status']}")
    
    # Final achievement status
    achievement_status = {
        'complete_biological_reflex_achieved': all_achieved,
        'all_triggers_under_5ms': all_achieved,
        'system_ready_for_production': all_achieved,
        'nervous_system_transformation_complete': all_achieved,
        'optimization_success': overall['biological_reflex_success_rate'] >= 0.95
    }
    
    final_icon = "🏆" if all_achieved else "🔧"
    final_status = "COMPLETE" if all_achieved else "IN PROGRESS"
    
    print(f"\n{final_icon} NERVOUS SYSTEM TRANSFORMATION: {final_status}")
    print(f"{'🎉 All 4 triggers achieving biological reflex speed!' if all_achieved else '🔧 Continue optimization for remaining triggers'}")
    
    return {
        'reflex_demonstration': reflex_demo,
        'achievement_status': achievement_status,
        'final_status': final_status
    }

async def run_complete_biological_reflex_validation() -> Dict[str, Any]:
    """
    Run complete biological reflex optimization validation.
    
    Tests temporal memory architecture and validates 100% biological reflex achievement.
    """
    print("🧬 COMPLETE BIOLOGICAL REFLEX OPTIMIZATION VALIDATION")
    print("=" * 70)
    
    validation_start = time.perf_counter()
    
    try:
        # Test 1: Remember trigger optimization
        remember_results = await test_remember_biological_reflex_optimization()
        
        # Test 2: Complete system validation
        system_results = await test_complete_system_biological_reflex()
        
        # Test 3: Temporal memory manager metrics
        temporal_manager = await get_temporal_memory_manager()
        temporal_metrics = temporal_manager.get_temporal_metrics()
        
        validation_time = (time.perf_counter() - validation_start) * 1000
        
        # Comprehensive results
        validation_results = {
            'validation_status': 'completed',
            'validation_time_ms': validation_time,
            'remember_optimization': remember_results,
            'system_performance': system_results,
            'temporal_architecture_metrics': temporal_metrics,
            'final_achievement': {
                'biological_reflex_optimization_successful': (
                    remember_results['biological_reflex_achievement']['biological_reflex_rate'] >= 0.95 and
                    system_results['achievement_status']['complete_biological_reflex_achieved']
                ),
                'nervous_system_transformation_complete': (
                    system_results['achievement_status']['nervous_system_transformation_complete']
                ),
                'production_ready': (
                    system_results['achievement_status']['system_ready_for_production']
                )
            }
        }
        
        print(f"\n" + "=" * 70)
        print("🏆 BIOLOGICAL REFLEX OPTIMIZATION VALIDATION COMPLETE")
        
        final_success = validation_results['final_achievement']['biological_reflex_optimization_successful']
        success_icon = "✅" if final_success else "🔧"
        success_text = "SUCCESS" if final_success else "NEEDS REFINEMENT"
        
        print(f"{success_icon} OPTIMIZATION STATUS: {success_text}")
        print(f"⏱️ Validation Time: {validation_time:.2f}ms")
        print(f"🎯 Ready for Phase 4 Completion: {'YES' if final_success else 'CONTINUE OPTIMIZATION'}")
        
        return validation_results
        
    except Exception as e:
        return {
            'validation_status': 'failed',
            'error': str(e),
            'validation_time_ms': (time.perf_counter() - validation_start) * 1000
        }

if __name__ == "__main__":
    asyncio.run(run_complete_biological_reflex_validation())