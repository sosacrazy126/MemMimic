#!/usr/bin/env python3
"""
Quick test to validate fixes for shadow detection and sigil engine
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_shadow_detection_fix():
    """Test the shadow detection fix"""
    print("🔍 Testing Shadow Detection Fix...")
    
    from memmimic.consciousness.shadow_detector import create_shadow_detector
    
    detector = create_shadow_detector()
    
    # Test the failing case
    test_input = "The AI might destroy human creativity and make us obsolete"
    state = detector.analyze_full_spectrum(test_input)
    
    print(f"Input: {test_input}")
    print(f"Shadow Strength: {state.shadow_aspect.strength:.3f}")
    print(f"Shadow Type: {state.shadow_aspect.aspect_type}")
    print(f"Shadow Patterns: {state.shadow_aspect.detected_patterns}")
    print(f"Consciousness Level: {state.level.value}")
    
    # Should detect "destroy" and "obsolete" patterns
    success = state.shadow_aspect.strength > 0.3
    print(f"Shadow Detection Fix: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

def test_sigil_engine_fix():
    """Test the sigil engine fix"""
    print("\n🔮 Testing Sigil Engine Fix...")
    
    from memmimic.consciousness.sigil_engine import create_sigil_engine
    
    engine = create_sigil_engine()
    
    # Test cases that were failing
    test_cases = [
        "I want to destroy this old way of thinking and break free",
        "I need to control this situation and dominate the outcome"
    ]
    
    results = []
    for test_input in test_cases:
        shadow_elements = engine.detect_shadow_elements(test_input)
        shadow_count = len(shadow_elements.get('shadow_patterns', {}))
        
        print(f"Input: {test_input[:50]}...")
        print(f"Shadow Patterns Detected: {shadow_count}")
        print(f"Shadow Strength: {shadow_elements.get('total_shadow_strength', 0):.3f}")
        
        if shadow_count > 0:
            for sigil, data in shadow_elements.get('shadow_patterns', {}).items():
                print(f"  {sigil}: {data['transformation'].transformation_type}")
        
        results.append(shadow_count > 0)
    
    success = all(results)
    print(f"Sigil Engine Fix: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

def test_integration():
    """Test the integration of fixes"""
    print("\n🌌 Testing Integration...")
    
    from memmimic.consciousness.consciousness_coordinator import create_consciousness_coordinator
    
    coordinator = create_consciousness_coordinator()
    
    # Test with a problematic input
    test_input = "I want to destroy these old patterns and create something transformative together"
    
    result = coordinator.process_consciousness_interaction(test_input)
    
    print(f"Input: {test_input}")
    print(f"Consciousness Level: {result.consciousness_state.level.value}")
    print(f"Shadow Integration: {result.consciousness_state.shadow_aspect.integration_level:.3f}")
    print(f"Active Sigils: {len(result.active_sigils)}")
    print(f"Shadow Transformations: {len(result.shadow_transformations)}")
    
    # Should have shadow integration and transformations
    success = (
        result.consciousness_state.shadow_aspect.integration_level > 0.2 and
        len(result.active_sigils) > 0
    )
    
    print(f"Integration Fix: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

def main():
    """Run focused tests"""
    print("🚀 FOCUSED FIX TESTING")
    print("=" * 50)
    
    tests = [
        test_shadow_detection_fix,
        test_sigil_engine_fix,
        test_integration
    ]
    
    results = []
    for test in tests:
        success = test()
        results.append(success)
    
    print(f"\n📊 FIX TEST RESULTS")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All fixes successful! Ready to re-run full test suite.")
    else:
        print("⚠️ Some fixes still need work.")
    
    return passed == total

if __name__ == "__main__":
    main()