#!/usr/bin/env python3
"""
Debug consciousness coordinator issues
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_coordinator_issues():
    """Debug consciousness coordinator shadow integration issues"""
    
    from memmimic.consciousness.shadow_detector import create_shadow_detector
    from memmimic.consciousness.consciousness_coordinator import create_consciousness_coordinator
    
    # Test cases that are failing
    failing_cases = [
        "This recursive exploration of consciousness is fascinating, but I worry about getting lost in infinite loops",
        "We're achieving a unity that feels both exhilarating and terrifying"
    ]
    
    detector = create_shadow_detector()
    coordinator = create_consciousness_coordinator()
    
    print("🔍 DEBUGGING CONSCIOUSNESS COORDINATOR SHADOW INTEGRATION ISSUES")
    print("=" * 80)
    
    for i, test_input in enumerate(failing_cases):
        print(f"\n🧪 Test Case {i+1}: {test_input[:60]}...")
        
        # Test shadow detector directly
        state = detector.analyze_full_spectrum(test_input)
        print(f"  📊 Direct Shadow Detection:")
        print(f"    Shadow Aspect: {state.shadow_aspect.aspect_type}")
        print(f"    Shadow Strength: {state.shadow_aspect.strength:.3f}")
        print(f"    Integration Level: {state.shadow_aspect.integration_level:.3f}")
        print(f"    Detected Patterns: {state.shadow_aspect.detected_patterns}")
        
        # Test coordinator
        result = coordinator.process_consciousness_interaction(test_input)
        print(f"  🌌 Coordinator Processing:")
        print(f"    Shadow Integration: {result.consciousness_state.shadow_aspect.integration_level:.3f}")
        print(f"    Shadow Transformations: {len(result.shadow_transformations)}")
        print(f"    Active Sigils: {len(result.active_sigils)}")
        
        # Check if this meets test criteria
        shadow_integration_pass = result.consciousness_state.shadow_aspect.integration_level > 0.2
        print(f"  ✅ Shadow Integration Pass: {shadow_integration_pass}")
        
        if not shadow_integration_pass:
            print(f"  ❌ FAILING: Shadow integration {result.consciousness_state.shadow_aspect.integration_level:.3f} <= 0.2")
    
    return True

if __name__ == "__main__":
    debug_coordinator_issues()