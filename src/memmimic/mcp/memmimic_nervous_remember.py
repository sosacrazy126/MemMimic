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
        
        # Format result for MCP
        if isinstance(result, dict):
            # Enhanced result with nervous system metadata
            formatted_result = {
                "status": result.get("status", "success"),
                "memory_id": result.get("memory_id"),
                "nervous_system_enhanced": True,
                "quality_score": result.get("quality_score"),
                "cxd_function": result.get("cxd_function"),
                "duplicate_detected": result.get("duplicate_detected", False),
                "content_enhanced": result.get("content_enhanced", False),
                "processing_time_ms": result.get("processing_time_ms"),
                "biological_reflex": result.get("processing_time_ms", 0) < 5.0 if result.get("processing_time_ms") else False,
                "nervous_system_version": "2.0.0"
            }
            
            print(json.dumps(formatted_result, indent=2))
        else:
            print(str(result))
            
    except Exception as e:
        logger.error(f"Enhanced remember failed: {e}")
        error_result = {
            "status": "error",
            "error": str(e),
            "nervous_system_enhanced": True,
            "nervous_system_version": "2.0.0"
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
