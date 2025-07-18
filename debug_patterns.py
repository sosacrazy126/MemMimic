#!/usr/bin/env python3
"""
Debug pattern detection differences
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_pattern_detection():
    """Debug the pattern detection differences"""
    
    from memmimic.consciousness.shadow_detector import create_shadow_detector
    from memmimic.consciousness.sigil_engine import create_sigil_engine
    
    detector = create_shadow_detector()
    engine = create_sigil_engine()
    
    test_input = "I want to destroy these old patterns and create something transformative together"
    
    print(f"Input: {test_input}")
    print()
    
    # Test shadow detector
    print("SHADOW DETECTOR:")
    state = detector.analyze_full_spectrum(test_input)
    print(f"  Shadow strength: {state.shadow_aspect.strength:.3f}")
    print(f"  Shadow type: {state.shadow_aspect.aspect_type}")
    print(f"  Shadow patterns: {state.shadow_aspect.detected_patterns}")
    print()
    
    # Test sigil engine
    print("SIGIL ENGINE:")
    
    # Check each transformation type manually
    for sigil, transformation in engine.shadow_transformations.items():
        print(f"  {sigil} ({transformation.transformation_type}):")
        shadow_patterns = engine._extract_shadow_patterns(transformation.transformation_type)
        print(f"    Available patterns: {shadow_patterns}")
        
        text_lower = test_input.lower()
        pattern_matches = [pattern for pattern in shadow_patterns if pattern in text_lower]
        print(f"    Matched patterns: {pattern_matches}")
        
        if pattern_matches:
            base_strength = len(pattern_matches) / len(shadow_patterns)
            match_boost = min(len(pattern_matches) * 0.2, 0.4)
            activation_strength = min(base_strength + match_boost, 1.0)
            print(f"    Activation strength: {activation_strength:.3f}")
            print(f"    Threshold: {transformation.activation_threshold}")
            print(f"    Meets threshold: {activation_strength >= transformation.activation_threshold}")
        else:
            print(f"    No matches")
        print()
    
    return True

if __name__ == "__main__":
    debug_pattern_detection()