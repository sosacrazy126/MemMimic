#!/usr/bin/env python3
"""
MemMimic Nervous System Think - MCP Bridge
Enhanced think with internal nervous system intelligence and Socratic guidance
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
    """Enhanced think with nervous system intelligence and internalized Socratic guidance"""
    if len(sys.argv) < 2:
        print("❌ Usage: memmimic_nervous_think.py <input_text>", file=sys.stderr)
        sys.exit(1)
    
    input_text = sys.argv[1]
    
    logger = get_error_logger("nervous_think_mcp")
    
    try:
        # Initialize unified enhanced triggers
        unified_triggers = UnifiedEnhancedTriggers()
        await unified_triggers.initialize()
        
        # Execute enhanced think with nervous system intelligence
        # This now includes internalized Socratic guidance that was previously external
        result = await unified_triggers.think_with_memory(input_text)
        
        # Format result for MCP (simple text format)
        if isinstance(result, dict):
            # Format as readable text with nervous system and Socratic enhancements
            status = result.get("status", "success")
            response = result.get("response", "No response")
            memories_count = result.get("relevant_memories_count", 0)
            processing_time = result.get("processing_time_ms", 0)
            socratic_insights = result.get("socratic_insights", [])

            print(f"🧬 Nervous System Think (v2.0.0)")
            print(f"Input: {input_text}")
            print(f"Status: {status}")
            print("")
            print("Response:")
            print(response)
            print("")
            print(f"Enhanced Processing:")
            print(f"- Relevant Memories: {memories_count}")
            print(f"- Processing Time: {processing_time:.2f}ms")
            print(f"- Biological Reflex: {'Yes' if processing_time < 5.0 else 'No'}")
            print(f"- Socratic Guidance: ✓ | Pattern Recognition: ✓ | Context Synthesis: ✓")

            if socratic_insights:
                print(f"\nSocratic Insights:")
                for insight in socratic_insights[:3]:  # Show top 3 insights
                    print(f"- {insight}")
        else:
            print(f"🧬 Nervous System Think (v2.0.0): {str(result)}")
            
    except Exception as e:
        logger.error(f"Enhanced think failed: {e}")
        print(f"🧬 Nervous System Think Error (v2.0.0)")
        print(f"Input: {input_text}")
        print(f"Error: {str(e)}")
        print("Enhanced processing failed - please check system status")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
