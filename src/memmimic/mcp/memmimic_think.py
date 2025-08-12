#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Think Tool - Enhanced Thinking with Memory
Combines sequential thinking with iterative memory retrieval
"""

import sys
import os
import json

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, '/home/evilbastardxd/Desktop/tools/memmimicc')
os.environ['MEMMIMIC_STORAGE'] = 'markdown'
os.environ['MEMMIMIC_MD_DIR'] = '/home/evilbastardxd/Desktop/tools/memmimicc'

try:
    from updated_mcp_tools import MemMimicMCP
    HAS_ENHANCED = True
except ImportError as e:
    HAS_ENHANCED = False
    error_msg = str(e)

def main():
    if not HAS_ENHANCED:
        print(f"❌ Error: Enhanced thinking not available: {error_msg}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("❌ Error: Missing input text")
        sys.exit(1)
    
    try:
        # Parse input - expects plain text as first argument
        input_text = sys.argv[1]
        
        # Initialize MemMimic MCP
        mcp = MemMimicMCP()
        
        # Process with enhanced thinking
        result = mcp.think_with_memory(
            input_text=input_text,
            mode='enhanced',
            max_thoughts=10
        )
        
        # Format professional response
        response_parts = []
        
        if result['status'] == 'success':
            # Header
            response_parts.append("🧠 ENHANCED THINKING WITH MEMORY")
            response_parts.append("=" * 50)
            response_parts.append(f"📝 Query: {result['query']}")
            response_parts.append(f"💪 Confidence: {result['confidence']:.0%}")
            response_parts.append(f"🔍 Memories Examined: {result['memories_examined']}")
            response_parts.append(f"💭 Thoughts Generated: {result['thoughts_generated']}")
            response_parts.append("")
            
            # Show thought progression
            if result.get('thought_process'):
                response_parts.append("🔄 THOUGHT PROGRESSION:")
                response_parts.append("-" * 30)
                for thought in result['thought_process']:
                    phase_symbol = {
                        'exploration': '🔍',
                        'refinement': '🎯',
                        'synthesis': '🔗',
                        'validation': '✓'
                    }.get(thought['phase'], '?')
                    
                    response_parts.append(f"{phase_symbol} Thought #{thought['number']} ({thought['phase'].upper()})")
                    response_parts.append(f"   Memories found: {thought['memories_found']}")
                    
                    if thought.get('insights'):
                        response_parts.append("   Insights:")
                        for insight in thought['insights'][:2]:
                            response_parts.append(f"   • {insight}")
                response_parts.append("")
            
            # Understanding summary
            if result.get('understanding'):
                response_parts.append("📊 UNDERSTANDING GAINED:")
                response_parts.append("-" * 30)
                for key, value in result['understanding'].items():
                    response_parts.append(f"• {value}")
                response_parts.append("")
            
            # Final analysis
            response_parts.append("💡 FINAL ANALYSIS:")
            response_parts.append("-" * 30)
            response_parts.append(result.get('final_analysis', 'Analysis complete'))
            response_parts.append("")
            
            # Key memories if available
            if result.get('key_memories'):
                response_parts.append("🗂️ KEY MEMORIES:")
                response_parts.append("-" * 30)
                for mem in result['key_memories'][:3]:
                    preview = mem.get('content_preview', mem.get('content', ''))[:100]
                    response_parts.append(f"• {preview}...")
        else:
            # Error response
            response_parts.append("❌ Error in enhanced thinking")
            response_parts.append(f"Error: {result.get('error', 'Unknown error')}")
        
        # Output the formatted response
        print('\n'.join(response_parts))
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()