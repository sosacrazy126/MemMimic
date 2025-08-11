#!/usr/bin/env python3
"""
Simplified MCP Server for MemMimic Markdown Storage
Works with our storage adapter system
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.insert(0, '.')

try:
    from storage_adapter import MarkdownAdapter, Memory
    from updated_mcp_tools import MemMimicMCP
except ImportError as e:
    print(f"❌ Error importing: {e}", file=sys.stderr)
    sys.exit(1)

class SimpleMCPServer:
    """Simple MCP server for MemMimic"""
    
    def __init__(self):
        self.storage_type = os.environ.get('MEMMIMIC_STORAGE', 'markdown')
        self.md_dir = os.environ.get('MEMMIMIC_MD_DIR', '.')
        
        if self.storage_type == 'markdown':
            self.adapter = MarkdownAdapter(self.md_dir)
        else:
            from updated_mcp_tools import MemMimicMCP
            self.mcp = MemMimicMCP()
    
    def status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            if hasattr(self, 'adapter'):
                total = self.adapter.count()
                return {
                    'status': 'success',
                    'storage_type': 'markdown',
                    'total_memories': total,
                    'index_path': str(self.adapter.index_path),
                    'memories_dir': str(self.adapter.memories_dir)
                }
            else:
                return self.mcp.status()
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def recall_cxd(self, query: str, function_filter: str = "ALL", limit: int = 10) -> List[Dict]:
        """Recall memories with CXD filtering"""
        try:
            if hasattr(self, 'adapter'):
                # Search using adapter
                memories = self.adapter.search(query, limit * 2)
                
                # Filter by CXD if specified
                if function_filter != "ALL":
                    memories = [m for m in memories 
                              if m.metadata.get('cxd', '').upper() == function_filter.upper()]
                
                # Convert to response format
                results = []
                for memory in memories[:limit]:
                    results.append({
                        'id': memory.id,
                        'content': memory.content,
                        'metadata': memory.metadata,
                        'importance': memory.importance,
                        'created': memory.created_at.isoformat(),
                        'cxd': memory.metadata.get('cxd', 'unknown'),
                        'type': memory.metadata.get('type', 'interaction')
                    })
                
                return results
            else:
                return self.mcp.recall_cxd(query, function_filter, limit)
        except Exception as e:
            return [{'error': str(e), 'status': 'error'}]
    
    def remember(self, content: str, memory_type: str = "interaction", metadata: Dict = None) -> Dict:
        """Store a memory"""
        try:
            if hasattr(self, 'adapter'):
                # Create memory object
                full_metadata = metadata or {}
                full_metadata['type'] = memory_type
                
                # Auto-classify CXD
                cxd_type = self._classify_cxd(content)
                full_metadata['cxd'] = cxd_type
                
                importance = self._calculate_importance(content, memory_type)
                
                memory = Memory(
                    content=content,
                    metadata=full_metadata,
                    importance=importance
                )
                
                memory_id = self.adapter.store(memory)
                
                return {
                    'status': 'success',
                    'memory_id': memory_id,
                    'type': memory_type,
                    'cxd': cxd_type,
                    'importance': importance,
                    'storage': 'markdown'
                }
            else:
                return self.mcp.remember(content, memory_type, metadata)
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def tales_list(self, category: str = None, limit: int = 10) -> Dict:
        """List tale-like memories"""
        try:
            if hasattr(self, 'adapter'):
                tales = []
                for memory_id, info in self.adapter.index.items():
                    memory = self.adapter.retrieve(memory_id)
                    if memory:
                        # Check if it's tale-like
                        is_tale = (
                            'tale' in memory.content.lower() or
                            memory.metadata.get('type') == 'tale' or
                            'title' in memory.metadata or
                            len(memory.content) > 500
                        )
                        if is_tale:
                            tales.append({
                                'id': memory.id,
                                'title': memory.metadata.get('title', f'Memory {memory.id}'),
                                'type': memory.metadata.get('type', 'memory'),
                                'importance': memory.importance,
                                'created': memory.created_at.isoformat(),
                                'preview': memory.content[:200] + '...' if len(memory.content) > 200 else memory.content
                            })
                
                # Sort by importance and date
                tales.sort(key=lambda t: (t['importance'], t['created']), reverse=True)
                
                if limit:
                    tales = tales[:limit]
                
                return {
                    'status': 'success',
                    'tales': tales,
                    'count': len(tales),
                    'storage': 'markdown'
                }
            else:
                return {'status': 'error', 'error': 'Tales not implemented for non-markdown storage'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _classify_cxd(self, content: str) -> str:
        """Simple CXD classification"""
        content_lower = content.lower()
        
        control_keywords = ['do', 'execute', 'run', 'command', 'action', 'perform', 
                          'create', 'delete', 'update', 'modify', 'build']
        context_keywords = ['why', 'because', 'understand', 'explain', 'context',
                          'background', 'history', 'reason', 'meaning', 'purpose']
        data_keywords = ['data', 'information', 'fact', 'number', 'statistic',
                        'result', 'value', 'measurement', 'record', 'detail']
        
        control_score = sum(1 for kw in control_keywords if kw in content_lower)
        context_score = sum(1 for kw in context_keywords if kw in content_lower)
        data_score = sum(1 for kw in data_keywords if kw in content_lower)
        
        if control_score >= context_score and control_score >= data_score:
            return 'CONTROL'
        elif context_score >= data_score:
            return 'CONTEXT'
        else:
            return 'DATA'
    
    def _calculate_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score"""
        base_score = 0.5
        
        if memory_type == 'milestone':
            base_score += 0.3
        elif memory_type == 'reflection':
            base_score += 0.2
        elif memory_type == 'tale':
            base_score += 0.4
        
        if len(content) > 500:
            base_score += 0.1
        elif len(content) < 50:
            base_score -= 0.1
        
        important_keywords = ['important', 'critical', 'essential', 'key', 'vital', 'tale', 'story']
        if any(kw in content.lower() for kw in important_keywords):
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple MemMimic MCP Server')
    parser.add_argument('command', choices=['status', 'recall', 'remember', 'tales'],
                       help='Command to execute')
    parser.add_argument('--query', help='Query for recall')
    parser.add_argument('--content', help='Content for remember')
    parser.add_argument('--type', default='interaction', help='Memory type')
    parser.add_argument('--cxd', default='ALL', help='CXD filter')
    parser.add_argument('--limit', type=int, default=10, help='Result limit')
    parser.add_argument('--category', help='Tale category filter')
    
    args = parser.parse_args()
    
    # Set environment for markdown storage
    os.environ['MEMMIMIC_STORAGE'] = 'markdown'
    os.environ['MEMMIMIC_MD_DIR'] = '.'
    
    server = SimpleMCPServer()
    
    if args.command == 'status':
        result = server.status()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'recall':
        if not args.query:
            print("Error: --query required for recall")
            return
        results = server.recall_cxd(args.query, args.cxd, args.limit)
        print(json.dumps(results, indent=2))
    
    elif args.command == 'remember':
        if not args.content:
            print("Error: --content required for remember")
            return
        result = server.remember(args.content, args.type)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'tales':
        result = server.tales_list(args.category, args.limit)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()