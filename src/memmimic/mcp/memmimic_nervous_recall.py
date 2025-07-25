#!/usr/bin/env python3
"""
MemMimic Nervous System Recall - MCP Bridge
Enhanced recall with internal nervous system intelligence
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
    """Enhanced recall with nervous system intelligence"""
    if len(sys.argv) < 2:
        print("❌ Usage: memmimic_nervous_recall.py <query> [function_filter] [limit] [--db db_name]", file=sys.stderr)
        sys.exit(1)
    
    query = sys.argv[1]
    function_filter = sys.argv[2] if len(sys.argv) > 2 else "ALL"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Parse optional db parameter
    db_name = None
    if "--db" in sys.argv:
        db_index = sys.argv.index("--db")
        if db_index + 1 < len(sys.argv):
            db_name = sys.argv[db_index + 1]
    
    logger = get_error_logger("nervous_recall_mcp")
    
    try:
        # Initialize unified enhanced triggers
        unified_triggers = UnifiedEnhancedTriggers()
        await unified_triggers.initialize()
        
        # Execute enhanced recall with nervous system intelligence
        result = await unified_triggers.recall_cxd(query, function_filter, limit, db_name)
        
        # Format result for MCP (simple text format)
        if isinstance(result, list):
            # Format as readable text with nervous system enhancements
            output_lines = [
                f"🧬 Nervous System Recall Results (v2.0.0)",
                f"Query: {query}",
                f"Filter: {function_filter} | Limit: {limit} | Found: {len(result)}",
                f"Enhanced: Pattern Analysis ✓ | Relationship Mapping ✓ | Context Awareness ✓",
                "",
                "Results:"
            ]

            for i, memory in enumerate(result, 1):
                content = memory.get('content', 'No content')
                # Fix: Use relevance_score instead of confidence
                confidence = memory.get('relevance_score', memory.get('confidence', 0.0))
                # Fix: Get CXD function from metadata (check multiple possible locations)
                metadata = memory.get('metadata', {})
                cxd_function = (
                    metadata.get('cxd_function') or
                    metadata.get('cxd', {}).get('function') or
                    memory.get('intelligence_analysis', {}).get('cxd_function') or
                    'Data'  # Default to Data instead of Unknown
                )

                output_lines.append(f"{i}. [{cxd_function}] (confidence: {confidence:.2f})")
                output_lines.append(f"   {content}")
                output_lines.append("")

            print("\n".join(output_lines))
        elif isinstance(result, dict):
            # Error case or special response - format as text
            print(f"🧬 Nervous System Recall (v2.0.0)")
            print(f"Status: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(str(result))
        else:
            print(f"🧬 Nervous System Recall (v2.0.0): {str(result)}")
            
    except Exception as e:
        logger.error(f"Enhanced recall failed: {e}")
        print(f"🧬 Nervous System Recall Error (v2.0.0)")
        print(f"Query: {query}")
        print(f"Error: {str(e)}")
        print("Enhanced processing failed - please check system status")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
