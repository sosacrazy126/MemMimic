#!/usr/bin/env python3
"""
MemMimic Analyze Patterns Tool - Simple memory pattern analysis
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
    try:
        mcp = MemMimicMCP()
        status = mcp.status()
        
        response = []
        response.append("🔍 MEMORY PATTERN ANALYSIS")
        response.append("=" * 50)
        
        stats = status['stats']
        
        # Memory type patterns
        if stats.get('memory_types'):
            response.append("MEMORY TYPE PATTERNS:")
            response.append("-" * 30)
            total = sum(stats['memory_types'].values())
            for mem_type, count in stats['memory_types'].items():
                percentage = (count / total) * 100
                response.append(f"• {mem_type}: {count} ({percentage:.1f}%)")
            response.append("")
        
        # CXD patterns
        if stats.get('cxd_distribution'):
            response.append("CXD CLASSIFICATION PATTERNS:")
            response.append("-" * 30)
            total_cxd = sum(stats['cxd_distribution'].values())
            for cxd_type, count in stats['cxd_distribution'].items():
                if count > 0:
                    percentage = (count / total_cxd) * 100
                    response.append(f"• {cxd_type}: {count} ({percentage:.1f}%)")
            response.append("")
        
        # Usage insights
        response.append("KEY INSIGHTS:")
        response.append("-" * 30)
        response.append(f"• Total memories: {stats['total_memories']}")
        response.append(f"• Most common type: {max(stats['memory_types'].items(), key=lambda x: x[1])[0]}")
        response.append(f"• Storage health: {'✅ Good' if stats['index_exists'] else '❌ Index missing'}")
        
        print('\n'.join(response))
        
    except Exception as e:
        print(f"❌ Error analyzing patterns: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()