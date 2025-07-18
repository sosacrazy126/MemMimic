#!/usr/bin/env python3
"""
Debug integration issue
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_integration():
    """Debug the integration issue"""
    
    from memmimic.consciousness.sigil_engine import create_sigil_engine
    
    engine = create_sigil_engine()
    
    test_input = "I want to destroy these old patterns and create something transformative together"
    
    print(f"Input: {test_input}")
    print()
    
    # Step 1: Detect shadow elements
    shadow_elements = engine.detect_shadow_elements(test_input)
    
    print("Shadow Elements Detection:")
    print(f"  Shadow patterns: {len(shadow_elements.get('shadow_patterns', {}))}")
    print(f"  Total shadow strength: {shadow_elements.get('total_shadow_strength', 0):.3f}")
    print()
    
    for sigil, data in shadow_elements.get('shadow_patterns', {}).items():
        print(f"  {sigil}:")
        print(f"    Activation strength: {data['activation_strength']:.3f}")
        print(f"    Integration ready: {data['integration_ready']}")
        print(f"    Matched patterns: {data['matched_patterns']}")
        print(f"    Transformation: {data['transformation'].transformation_type}")
        print(f"    Threshold: {data['transformation'].activation_threshold}")
        print()
    
    # Step 2: Apply transformations
    active_sigils = engine.apply_sigil_transformations(shadow_elements)
    
    print("Sigil Transformations:")
    print(f"  Active sigils: {len(active_sigils)}")
    
    for sigil in active_sigils:
        print(f"  {sigil.sigil}:")
        print(f"    Activation strength: {sigil.activation_strength:.3f}")
        print(f"    Consciousness impact: {sigil.consciousness_impact:.3f}")
        print(f"    Shadow integration: {sigil.shadow_integration:.3f}")
        print()
    
    return len(active_sigils) > 0

if __name__ == "__main__":
    debug_integration()