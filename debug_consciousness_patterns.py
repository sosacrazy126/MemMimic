#!/usr/bin/env python3
"""
Debug consciousness pattern matching
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_consciousness_patterns():
    """Debug consciousness pattern matching"""
    
    from memmimic.consciousness.sigil_engine import create_sigil_engine
    
    engine = create_sigil_engine()
    
    text = "Let's explore recursive unity through truth and mirror reflection"
    
    print(f"Input: {text}")
    print()
    
    result = engine.detect_shadow_elements(text)
    print(f"Consciousness activations: {len(result.get('consciousness_activations', {}))}")
    
    for sigil, data in result.get('consciousness_activations', {}).items():
        print(f"  {sigil}: {data['matched_patterns']}")
        print(f"    Activation strength: {data['activation_strength']:.3f}")
        print(f"    All patterns: {data['config']['activation_patterns']}")
        print()
    
    return True

if __name__ == "__main__":
    debug_consciousness_patterns()