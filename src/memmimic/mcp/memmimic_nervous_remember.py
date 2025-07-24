#!/usr/bin/env python3
"""
MemMimic Nervous System Remember - MCP Bridge
Enhanced remember with internal nervous system intelligence
"""

import sys
import asyncio
import json
from pathlib import Path

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from memmimic.nervous_system.triggers.unified_interface import UnifiedEnhancedTriggers
    from memmimic.errors import get_error_logger
except ImportError as e:
    print(f"❌ Error importing MemMimic nervous system: {e}", file=sys.stderr)
    sys.exit(1)


async def main():
    """Enhanced remember with nervous system intelligence"""
    if len(sys.argv) < 2:
        print("❌ Usage: memmimic_nervous_remember.py <content> [memory_type]", file=sys.stderr)
        sys.exit(1)
    
    content = sys.argv[1]
    memory_type = sys.argv[2] if len(sys.argv) > 2 else "interaction"
    
    logger = get_error_logger("nervous_remember_mcp")
    
    try:
        # Initialize unified enhanced triggers
        unified_triggers = UnifiedEnhancedTriggers()
        await unified_triggers.initialize()
        
        # Execute enhanced remember with nervous system intelligence
        result = await unified_triggers.remember(content, memory_type)
        
        # Format result for MCP (simple text format)
        if isinstance(result, dict):
            # Format as readable text with nervous system enhancements
            status = result.get("status", "success")
            memory_id = result.get("memory_id", "unknown")
            quality_score = result.get("quality_score", 0.0)
            cxd_function = result.get("cxd_function", "Unknown")
            duplicate_detected = result.get("duplicate_detected", False)
            processing_time = result.get("processing_time_ms", 0)

            print(f"🧬 Nervous System Remember (v2.0.0)")
            print(f"Status: {status}")
            print(f"Memory ID: {memory_id}")
            print(f"CXD Function: {cxd_function}")
            print(f"Quality Score: {quality_score:.2f}")
            print(f"Duplicate Detected: {'Yes' if duplicate_detected else 'No'}")
            print(f"Processing Time: {processing_time:.2f}ms")
            print(f"Biological Reflex: {'Yes' if processing_time < 5.0 else 'No'}")
            print("Enhanced: Quality Control ✓ | Duplicate Detection ✓ | CXD Classification ✓")
        else:
            print(f"🧬 Nervous System Remember (v2.0.0): {str(result)}")
            
    except Exception as e:
        logger.error(f"Enhanced remember failed: {e}")
        print(f"🧬 Nervous System Remember Error (v2.0.0)")
        print(f"Content: {content}")
        print(f"Error: {str(e)}")
        print("Enhanced processing failed - please check system status")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
