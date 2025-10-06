#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Remember Tool - Store memories with CXD classification
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
sys.path.insert(0, '/home/sigilzo/tools-raw/MemMimic')
os.environ['MEMMIMIC_STORAGE'] = 'markdown'
os.environ['MEMMIMIC_MD_DIR'] = '/home/sigilzo/tools-raw/MemMimic'

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
            content = args.get('content', '')
            memory_type = args.get('memory_type', 'interaction')
            metadata = args.get('metadata', {})
        else:
            # Positional format from server.js
            content = sys.argv[1]
            memory_type = sys.argv[2] if len(sys.argv) > 2 else 'interaction'
            metadata = {}
        
        if not content:
            print("❌ Error: Content is required")
            sys.exit(1)
        
        # Initialize MemMimic MCP
        mcp = MemMimicMCP()
        
        # Store the memory
        result = mcp.remember(
            content=content,
            memory_type=memory_type,
            metadata=metadata
        )
        
        # Format response
        if result.get('status') == 'success':
            response_parts = []
            response_parts.append("✅ MEMORY STORED SUCCESSFULLY")
            response_parts.append("=" * 50)
            response_parts.append(f"📝 Memory ID: {result['memory_id']}")
            response_parts.append(f"📂 Type: {memory_type}")
            response_parts.append(f"📏 Size: {len(content)} characters")
            
            if result.get('cxd'):
                response_parts.append(f"🏷️ CXD Classification: {result['cxd']}")
            
            response_parts.append(f"⏰ Timestamp: {result.get('timestamp', datetime.now().isoformat())}")
            response_parts.append("")
            response_parts.append("💡 Your memory has been securely stored and indexed.")
            response_parts.append("   Use recall_cxd() to search for it later.")
            
            print('\n'.join(response_parts))
        else:
            print(f"❌ Error storing memory: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing arguments: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()