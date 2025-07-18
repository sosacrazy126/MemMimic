#!/usr/bin/env python3
"""
Debug shadow pattern detection
"""

import sys
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_shadow_patterns():
    """Debug shadow pattern detection"""
    
    from memmimic.consciousness.shadow_detector import create_shadow_detector
    
    detector = create_shadow_detector()
    
    # Test inputs that should have shadow patterns
    test_inputs = [
        "This recursive exploration of consciousness is fascinating, but I worry about getting lost in infinite loops",
        "We're achieving a unity that feels both exhilarating and terrifying"
    ]
    
    print("🔍 DEBUGGING SHADOW PATTERN DETECTION")
    print("=" * 60)
    
    print("📋 Available Shadow Patterns:")
    for aspect_name, aspect_config in detector.shadow_patterns.items():
        print(f"  {aspect_name}: {aspect_config['patterns']}")
    print()
    
    for i, test_input in enumerate(test_inputs):
        print(f"🧪 Test Case {i+1}: {test_input[:60]}...")
        
        text_lower = test_input.lower()
        words_in_text = text_lower.split()
        
        print(f"  📝 Words in text: {words_in_text}")
        print(f"  🔍 Checking for shadow patterns:")
        
        for aspect_name, aspect_config in detector.shadow_patterns.items():
            matches = [pattern for pattern in aspect_config['patterns'] if pattern in text_lower]
            print(f"    {aspect_name}: {matches}")
        
        # Check for potential missing patterns
        shadow_words = ['worry', 'lost', 'terrifying', 'exhilarating', 'fascinating', 'infinite', 'loops']
        found_shadow_words = [word for word in shadow_words if word in text_lower]
        print(f"  🌑 Potential shadow words found: {found_shadow_words}")
        
        print()
    
    return True

if __name__ == "__main__":
    debug_shadow_patterns()