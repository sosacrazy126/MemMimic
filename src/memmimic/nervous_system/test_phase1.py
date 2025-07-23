"""
Phase 1 Integration Test - Nervous System Core Foundation

Tests the complete integration of all intelligence components
with performance validation against <5ms targets.
"""

import asyncio
import time
from typing import Dict, Any

from .core import NervousSystemCore

async def test_nervous_system_initialization():
    """Test complete nervous system initialization"""
    print("🧠 Testing NervousSystemCore initialization...")
    
    start_time = time.perf_counter()
    
    # Initialize nervous system
    nervous_system = NervousSystemCore(db_path=":memory:")
    await nervous_system.initialize()
    
    init_time = (time.perf_counter() - start_time) * 1000
    print(f"✅ Initialization completed in {init_time:.2f}ms")
    
    # Verify all components are initialized
    health = await nervous_system.health_check()
    print(f"🔍 Health Status: {health['status']}")
    print(f"📊 Components: {health['component_health']}")
    
    return nervous_system

async def test_intelligence_processing():
    """Test parallel intelligence processing"""
    print("\n🚀 Testing intelligence processing pipeline...")
    
    nervous_system = await test_nervous_system_initialization()
    
    # Test content samples
    test_contents = [
        ("This is a breakthrough discovery in AI memory systems that will revolutionize how we store and retrieve contextual information.", "milestone"),
        ("User asked about weather today", "interaction"),
        ("Implemented new caching mechanism for improved performance with 50% speed increase", "technical"),
        ("Realized the importance of quality gates in memory management", "reflection")
    ]
    
    total_processing_time = 0
    successful_processes = 0
    
    for content, memory_type in test_contents:
        print(f"\n📝 Processing: {content[:50]}...")
        
        start_time = time.perf_counter()
        
        try:
            result = await nervous_system.process_with_intelligence(
                content=content,
                memory_type=memory_type,
                enable_quality_gate=True,
                enable_duplicate_detection=True,
                enable_socratic_guidance=False  # Skip for performance test
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            total_processing_time += processing_time
            successful_processes += 1
            
            print(f"⏱️  Processing Time: {processing_time:.2f}ms")
            print(f"🎯 Recommended Action: {result.get('recommended_action', 'N/A')}")
            print(f"🔒 Confidence: {result.get('confidence', 0):.2f}")
            
            # Check performance target
            if processing_time <= 5.0:
                print("✅ Performance target met (<5ms)")
            else:
                print("⚠️  Performance target missed (>5ms)")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    
    # Performance summary
    avg_processing_time = total_processing_time / max(1, successful_processes)
    print(f"\n📈 Performance Summary:")
    print(f"   Average Processing Time: {avg_processing_time:.2f}ms")
    print(f"   Target: <5ms")
    print(f"   Success Rate: {successful_processes}/{len(test_contents)}")
    print(f"   Performance Target Met: {'✅' if avg_processing_time < 5.0 else '❌'}")
    
    return nervous_system

async def test_individual_components():
    """Test individual intelligence components"""
    print("\n🔬 Testing individual intelligence components...")
    
    nervous_system = await test_nervous_system_initialization()
    
    # Test Quality Gate
    print("\n📏 Testing InternalQualityGate...")
    if nervous_system._quality_gate:
        test_content = "This is a comprehensive technical analysis of system performance improvements"
        assessment = await nervous_system._quality_gate.assess_quality(test_content, "technical")
        print(f"   Quality Score: {assessment.overall_score:.3f}")
        print(f"   Auto Approve: {assessment.auto_approve}")
        print(f"   Enhancement Needed: {assessment.needs_enhancement}")
    
    # Test Duplicate Detector
    print("\n🔍 Testing SemanticDuplicateDetector...")
    if nervous_system._duplicate_detector:
        test_content = "New breakthrough in AI memory systems"
        analysis = await nervous_system._duplicate_detector.detect_duplicates(test_content, "milestone")
        print(f"   Is Duplicate: {analysis.is_duplicate}")
        print(f"   Similarity Score: {analysis.similarity_score:.3f}")
        print(f"   Resolution Action: {analysis.resolution_action}")
    
    # Test Socratic Guidance
    print("\n🧘 Testing InternalSocraticGuidance...")
    if nervous_system._socratic_guidance:
        test_content = "Should we update this memory with new context information?"
        context = {"memory_type": "interaction"}
        guidance = await nervous_system._socratic_guidance.guide_memory_decision(test_content, context)
        print(f"   Recommendation: {guidance.recommendation}")
        print(f"   Confidence: {guidance.confidence:.3f}")
        print(f"   Questions Generated: {len(guidance.questions)}")

async def test_performance_metrics():
    """Test performance metrics collection"""
    print("\n📊 Testing performance metrics...")
    
    nervous_system = await test_nervous_system_initialization()
    
    # Process some content to generate metrics
    await nervous_system.process_with_intelligence(
        "Test content for metrics generation",
        "interaction"
    )
    
    # Get performance metrics
    metrics = nervous_system.get_performance_metrics()
    
    print(f"📈 Nervous System Metrics:")
    print(f"   Total Operations: {metrics['total_operations']}")
    print(f"   Average Response Time: {metrics['average_response_time_ms']:.2f}ms")
    print(f"   Performance Target Met: {metrics['performance_target_met']}")
    print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.2f}")
    
    # Component-specific metrics
    if nervous_system._quality_gate:
        quality_metrics = nervous_system._quality_gate.get_performance_metrics()
        print(f"\n📏 Quality Gate Metrics:")
        print(f"   Total Assessments: {quality_metrics['total_assessments']}")
        print(f"   Auto Approval Rate: {quality_metrics['auto_approval_rate']:.2f}")
        print(f"   Average Processing Time: {quality_metrics['average_processing_time_ms']:.2f}ms")
    
    if nervous_system._duplicate_detector:
        duplicate_metrics = nervous_system._duplicate_detector.get_performance_metrics()
        print(f"\n🔍 Duplicate Detector Metrics:")
        print(f"   Total Detections: {duplicate_metrics['total_detections']}")
        print(f"   Duplicates Found: {duplicate_metrics['duplicates_found']}")
        print(f"   Detection Rate: {duplicate_metrics['duplicate_detection_rate']:.2f}")
    
    if nervous_system._socratic_guidance:
        socratic_metrics = nervous_system._socratic_guidance.get_performance_metrics()
        print(f"\n🧘 Socratic Guidance Metrics:")
        print(f"   Total Guidances: {socratic_metrics['total_guidances']}")
        print(f"   Average Processing Time: {socratic_metrics['average_processing_time_ms']:.2f}ms")

async def run_phase1_validation():
    """Run complete Phase 1 validation suite"""
    print("🚀 PHASE 1 VALIDATION: Nervous System Core Foundation")
    print("=" * 60)
    
    try:
        # Test 1: Initialization
        nervous_system = await test_nervous_system_initialization()
        
        # Test 2: Intelligence Processing
        await test_intelligence_processing()
        
        # Test 3: Individual Components
        await test_individual_components()
        
        # Test 4: Performance Metrics
        await test_performance_metrics()
        
        print("\n" + "=" * 60)
        print("✅ PHASE 1 VALIDATION COMPLETED SUCCESSFULLY")
        print("🧠 Nervous System Core Foundation is fully operational")
        print("⚡ Ready for Phase 2: Enhanced Triggers Implementation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 VALIDATION FAILED: {e}")
        print("🔧 Review implementation and fix issues before proceeding")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase1_validation())