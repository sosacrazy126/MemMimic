#!/usr/bin/env python3
"""
MemMimic Context Tale Tool - Generate narrative from memory fragments
"""

import sys
import os
sys.path.insert(0, '/home/evilbastardxd/Desktop/tools/memmimicc')
os.environ['MEMMIMIC_STORAGE'] = 'markdown'
os.environ['MEMMIMIC_MD_DIR'] = '/home/evilbastardxd/Desktop/tools/memmimicc'

try:
    from updated_mcp_tools import MemMimicMCP
except ImportError as e:
    print(f"❌ Error importing MemMimicMCP: {e}")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Missing query argument")
        sys.exit(1)
    
    try:
        query = sys.argv[1]
        style = sys.argv[2] if len(sys.argv) > 2 else 'auto'
        max_memories = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        
        mcp = MemMimicMCP()
        
        # Search for relevant memories
        memories = mcp.recall_cxd(query, function_filter='ALL', limit=max_memories)
        
        response = []
        response.append("📖 CONTEXT TALE GENERATION")
        response.append("=" * 50)
        response.append(f"Topic: {query}")
        response.append(f"Style: {style}")
        response.append(f"Memories found: {len(memories)}")
        response.append("")
        
        if memories:
            response.append("NARRATIVE:")
            response.append("-" * 30)
            response.append(f"Based on {len(memories)} relevant memories, here's the story of {query}:")
            response.append("")
            
            # Create simple narrative from memories
            for i, memory in enumerate(memories[:5], 1):
                if hasattr(memory, 'content'):
                    content = memory.content[:200]
                else:
                    content = memory.get('content', '')[:200]
                
                response.append(f"Chapter {i}: {content}...")
                response.append("")
        else:
            response.append("No memories found to create a narrative.")
            response.append(f"Try using remember() to store information about {query} first.")
        
        print('\n'.join(response))
        
    except Exception as e:
        print(f"❌ Error generating context tale: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()