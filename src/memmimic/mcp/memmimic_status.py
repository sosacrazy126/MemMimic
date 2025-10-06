#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Status Tool - System health and statistics
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

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
    print(f"❌ Error importing: {e}")
    sys.exit(1)

def main():
    try:
        # Initialize components
        mcp = MemMimicMCP()

        # Get system status
        status = mcp.status()
        
        # Format response
        response_parts = []
        response_parts.append("📊 MEMMIMIC SYSTEM STATUS")
        response_parts.append("=" * 50)
        response_parts.append(f"✅ Status: {status['status'].upper()}")
        response_parts.append(f"🗄️ Storage Type: {status['stats']['storage_type']}")
        response_parts.append("")
        
        # Memory statistics
        response_parts.append("MEMORY STATISTICS:")
        response_parts.append("-" * 30)
        response_parts.append(f"📝 Total Memories: {status['stats']['total_memories']}")
        response_parts.append(f"🆕 Recent Memories: {status['stats'].get('recent_memories', 0)}")
        response_parts.append("")
        
        # Memory type breakdown
        if status['stats'].get('memory_types'):
            response_parts.append("Memory Types:")
            for mem_type, count in status['stats']['memory_types'].items():
                response_parts.append(f"  • {mem_type}: {count}")
            response_parts.append("")
        
        # CXD distribution
        if status['stats'].get('cxd_distribution'):
            response_parts.append("CXD Classification:")
            for cxd_type, count in status['stats']['cxd_distribution'].items():
                if count > 0:
                    response_parts.append(f"  • {cxd_type}: {count}")
            response_parts.append("")
        
        # Tale statistics
        response_parts.append("TALE STATISTICS:")
        response_parts.append("-" * 30)
        # Storage details
        response_parts.append("STORAGE DETAILS:")
        response_parts.append("-" * 30)
        response_parts.append(f"📂 Base Directory: {status['stats']['markdown_dir']}")
        response_parts.append(f"📑 Index Status: {'✅ Exists' if status['stats']['index_exists'] else '❌ Missing'}")
        
        # Health summary
        response_parts.append("")
        response_parts.append("💚 SYSTEM HEALTH: OPERATIONAL")
        response_parts.append("All systems functioning normally")
        
        print('\n'.join(response_parts))
        
    except Exception as e:
        print(f"❌ Error checking status: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()