#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Recall CXD Tool - Search memories with CXD filtering
"""

import sys
import os
import json
from datetime import datetime

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
except ImportError as e:
    print(f"❌ Error importing MemMimicMCP: {e}")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Missing arguments")
        sys.exit(1)
    
    try:
        # Handle both JSON and positional arguments
        if sys.argv[1].startswith('{'):
            # JSON format
            args = json.loads(sys.argv[1])
            query = args.get('query', '')
            function_filter = args.get('function_filter', 'ALL')
            limit = args.get('limit', 10)
            db_name = args.get('db_name', 'memmimic')
        else:
            # Positional format from server.js
            query = sys.argv[1]
            function_filter = sys.argv[2] if len(sys.argv) > 2 else 'ALL'
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            # Check for --db flag
            db_name = 'memmimic'
            if '--db' in sys.argv:
                db_idx = sys.argv.index('--db')
                if db_idx + 1 < len(sys.argv):
                    db_name = sys.argv[db_idx + 1]
        
        if not query:
            print("❌ Error: Query is required")
            sys.exit(1)
        
        # Initialize MemMimic MCP
        mcp = MemMimicMCP()
        
        # Search memories
        memories = mcp.recall_cxd(
            query=query,
            function_filter=function_filter,
            limit=limit
        )
        
        # Format response
        response_parts = []
        response_parts.append("🔍 MEMORY SEARCH RESULTS")
        response_parts.append("=" * 50)
        response_parts.append(f"📝 Query: {query}")
        response_parts.append(f"🏷️ CXD Filter: {function_filter}")
        response_parts.append(f"📊 Results: {len(memories)} memories found")
        response_parts.append("")
        
        if memories:
            response_parts.append("MEMORIES:")
            response_parts.append("-" * 30)
            
            for i, memory in enumerate(memories, 1):
                # Handle both dict and Memory object formats
                if hasattr(memory, 'content'):
                    content = memory.content
                    memory_id = memory.id
                    cxd = getattr(memory, 'cxd', 'unknown')
                    timestamp = getattr(memory, 'timestamp', '')
                else:
                    content = memory.get('content', '')
                    memory_id = memory.get('id', 'unknown')
                    cxd = memory.get('cxd', 'unknown')
                    timestamp = memory.get('timestamp', '')
                
                # Truncate content for display
                preview = content[:150] + '...' if len(content) > 150 else content
                
                response_parts.append(f"\n{i}. Memory ID: {memory_id}")
                response_parts.append(f"   CXD: {cxd}")
                if timestamp:
                    response_parts.append(f"   Time: {timestamp}")
                response_parts.append(f"   Content: {preview}")
        else:
            response_parts.append("No memories found matching your query.")
            response_parts.append("")
            response_parts.append("💡 Tips:")
            response_parts.append("   • Try broader search terms")
            response_parts.append("   • Check your CXD filter (use 'ALL' to search everything)")
            response_parts.append("   • Use remember() to store new memories first")
        
        print('\n'.join(response_parts))
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing arguments: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()