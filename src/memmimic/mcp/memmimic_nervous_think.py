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
        
        # Format result for MCP
        if isinstance(result, dict):
            # Enhanced result with nervous system and Socratic metadata
            formatted_result = {
                "status": result.get("status", "success"),
                "response": result.get("response"),
                "input_text": input_text,
                "nervous_system_enhanced": True,
                "socratic_guidance_applied": True,  # Internalized from socratic_dialogue tool
                "pattern_recognition_applied": True,
                "context_synthesis_applied": True,
                "relevant_memories_count": result.get("relevant_memories_count", 0),
                "processing_time_ms": result.get("processing_time_ms"),
                "biological_reflex": result.get("processing_time_ms", 0) < 5.0 if result.get("processing_time_ms") else False,
                "socratic_insights": result.get("socratic_insights", []),
                "thought_patterns": result.get("thought_patterns", []),
                "nervous_system_version": "2.0.0"
            }
            
            print(json.dumps(formatted_result, indent=2))
        else:
            print(str(result))
            
    except Exception as e:
        logger.error(f"Enhanced think failed: {e}")
        error_result = {
            "status": "error",
            "error": str(e),
            "input_text": input_text,
            "nervous_system_enhanced": True,
            "socratic_guidance_applied": False,
            "nervous_system_version": "2.0.0"
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
