#!/usr/bin/env python3
"""
Updated MCP tools that work with the storage adapter
Supports both SQLite and Markdown backends transparently
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import the storage adapter
from storage_adapter import create_storage_adapter, Memory

# Configuration - can be overridden by environment variables
STORAGE_TYPE = os.environ.get('MEMMIMIC_STORAGE', 'hybrid')  # 'sqlite', 'markdown', or 'hybrid'
SQLITE_PATH = os.environ.get('MEMMIMIC_DB_PATH', 'memmimic.db')
MARKDOWN_DIR = os.environ.get('MEMMIMIC_MD_DIR', '.')
WRITE_TO = os.environ.get('MEMMIMIC_WRITE_TO', 'both')  # For hybrid: 'sqlite', 'markdown', or 'both'


class MemMimicMCP:
    """Updated MCP interface using storage adapter"""
    
    def __init__(self):
        """Initialize with configured storage adapter"""
        self.adapter = create_storage_adapter(
            storage_type=STORAGE_TYPE,
            sqlite_path=SQLITE_PATH,
            markdown_dir=MARKDOWN_DIR,
            write_to=WRITE_TO
        )
        
        # For backward compatibility
        self.storage = self.adapter
    
    def remember(self, content: str, memory_type: str = "interaction", 
                 metadata: Dict = None) -> Dict:
        """
        Store a memory
        
        Args:
            content: Memory content
            memory_type: Type of memory (interaction, reflection, milestone)
            metadata: Additional metadata
        
        Returns:
            Success response with memory ID
        """
        try:
            # Prepare metadata
            full_metadata = metadata or {}
            full_metadata['type'] = memory_type
            
            # Auto-classify with CXD if available
            cxd_type = self._classify_cxd(content)
            full_metadata['cxd'] = cxd_type
            
            # Calculate importance
            importance = self._calculate_importance(content, memory_type)
            
            # Create memory object
            memory = Memory(
                content=content,
                metadata=full_metadata,
                importance=importance
            )
            
            # Store using adapter
            memory_id = self.adapter.store(memory)
            
            return {
                'status': 'success',
                'memory_id': memory_id,
                'type': memory_type,
                'cxd': cxd_type,
                'importance': importance,
                'storage': STORAGE_TYPE
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def recall_cxd(self, query: str, function_filter: str = "ALL", 
                   limit: int = 10) -> List[Dict]:
        """
        Recall memories with CXD filtering
        
        Args:
            query: Search query
            function_filter: CXD filter (CONTROL, CONTEXT, DATA, or ALL)
            limit: Maximum results
        
        Returns:
            List of matching memories
        """
        try:
            # Search using adapter
            memories = self.adapter.search(query, limit * 2)  # Get extra for filtering
            
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
            
        except Exception as e:
            return [{
                'error': str(e),
                'status': 'error'
            }]
    
    def think_with_memory(self, input_text: str) -> Dict:
        """
        Process input with memory context
        
        Args:
            input_text: Input to process
        
        Returns:
            Response with relevant memories and analysis
        """
        try:
            # Find relevant memories
            memories = self.adapter.search(input_text, limit=5)
            
            # Build context
            context = []
            for memory in memories:
                context.append({
                    'content': memory.content[:200],  # Preview
                    'relevance': self._calculate_relevance(input_text, memory.content),
                    'cxd': memory.metadata.get('cxd', 'unknown')
                })
            
            # Sort by relevance
            context.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Generate response
            response = {
                'status': 'success',
                'input': input_text,
                'relevant_memories': len(memories),
                'context': context[:3],  # Top 3 most relevant
                'analysis': self._generate_analysis(input_text, memories),
                'storage': STORAGE_TYPE
            }
            
            return response
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def status(self) -> Dict:
        """
        Get system status
        
        Returns:
            Status information
        """
        try:
            total_memories = self.adapter.count()
            
            # Get recent memories
            recent = self.adapter.get_all(limit=5)
            
            # Calculate statistics
            stats = {
                'total_memories': total_memories,
                'storage_type': STORAGE_TYPE,
                'recent_memories': len(recent)
            }
            
            # Add storage-specific info
            if STORAGE_TYPE == 'markdown':
                stats['markdown_dir'] = MARKDOWN_DIR
                stats['index_exists'] = (Path(MARKDOWN_DIR) / 'memories' / 'index.json').exists()
            elif STORAGE_TYPE == 'sqlite':
                stats['database_path'] = SQLITE_PATH
                stats['database_exists'] = Path(SQLITE_PATH).exists()
            elif STORAGE_TYPE == 'hybrid':
                stats['sqlite_path'] = SQLITE_PATH
                stats['markdown_dir'] = MARKDOWN_DIR
                stats['write_to'] = WRITE_TO
            
            # Memory type breakdown
            type_counts = {}
            for memory in self.adapter.get_all(limit=100):
                mem_type = memory.metadata.get('type', 'unknown')
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            stats['memory_types'] = type_counts
            
            # CXD breakdown
            cxd_counts = {}
            for memory in self.adapter.get_all(limit=100):
                cxd = memory.metadata.get('cxd', 'unknown')
                cxd_counts[cxd] = cxd_counts.get(cxd, 0) + 1
            
            stats['cxd_distribution'] = cxd_counts
            
            return {
                'status': 'success',
                'stats': stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_memory(self, memory_id: str, content: str = None, 
                     metadata: Dict = None) -> Dict:
        """
        Update an existing memory
        
        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
        
        Returns:
            Success response
        """
        try:
            # Retrieve existing memory
            memory = self.adapter.retrieve(memory_id)
            if not memory:
                return {
                    'status': 'error',
                    'error': f"Memory {memory_id} not found"
                }
            
            # Update fields
            if content:
                memory.content = content
            if metadata:
                memory.metadata.update(metadata)
            
            # Update importance if content changed
            if content:
                memory.importance = self._calculate_importance(content, 
                                                              memory.metadata.get('type', 'interaction'))
            
            # Save updates
            success = self.adapter.update(memory_id, memory)
            
            return {
                'status': 'success' if success else 'error',
                'memory_id': memory_id,
                'updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def delete_memory(self, memory_id: str) -> Dict:
        """
        Delete a memory
        
        Args:
            memory_id: Memory ID to delete
        
        Returns:
            Success response
        """
        try:
            success = self.adapter.delete(memory_id)
            
            return {
                'status': 'success' if success else 'error',
                'memory_id': memory_id,
                'deleted': success
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def migrate_to_markdown(self) -> Dict:
        """
        Special tool to migrate from SQLite to Markdown
        
        Returns:
            Migration status
        """
        try:
            if STORAGE_TYPE != 'hybrid':
                return {
                    'status': 'error',
                    'error': 'Migration requires hybrid storage mode'
                }
            
            # Get all memories from SQLite
            sqlite_adapter = create_storage_adapter('sqlite', db_path=SQLITE_PATH)
            all_memories = sqlite_adapter.get_all()
            
            # Store each in markdown
            markdown_adapter = create_storage_adapter('markdown', base_dir=MARKDOWN_DIR)
            migrated = 0
            failed = 0
            
            for memory in all_memories:
                try:
                    markdown_adapter.store(memory)
                    migrated += 1
                except Exception as e:
                    print(f"Failed to migrate {memory.id}: {e}")
                    failed += 1
            
            return {
                'status': 'success',
                'total': len(all_memories),
                'migrated': migrated,
                'failed': failed
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # Helper methods
    
    def _classify_cxd(self, content: str) -> str:
        """Simple CXD classification based on keywords"""
        content_lower = content.lower()
        
        # CONTROL indicators
        control_keywords = ['do', 'execute', 'run', 'command', 'action', 'perform', 
                          'create', 'delete', 'update', 'modify', 'build']
        
        # CONTEXT indicators  
        context_keywords = ['why', 'because', 'understand', 'explain', 'context',
                          'background', 'history', 'reason', 'meaning', 'purpose']
        
        # DATA indicators
        data_keywords = ['data', 'information', 'fact', 'number', 'statistic',
                        'result', 'value', 'measurement', 'record', 'detail']
        
        # Count keyword matches
        control_score = sum(1 for kw in control_keywords if kw in content_lower)
        context_score = sum(1 for kw in context_keywords if kw in content_lower)
        data_score = sum(1 for kw in data_keywords if kw in content_lower)
        
        # Return highest scoring category
        if control_score >= context_score and control_score >= data_score:
            return 'CONTROL'
        elif context_score >= data_score:
            return 'CONTEXT'
        else:
            return 'DATA'
    
    def _calculate_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score for a memory"""
        base_score = 0.5
        
        # Type-based adjustments
        if memory_type == 'milestone':
            base_score += 0.3
        elif memory_type == 'reflection':
            base_score += 0.2
        
        # Length-based adjustments
        if len(content) > 500:
            base_score += 0.1
        elif len(content) < 50:
            base_score -= 0.1
        
        # Keyword-based adjustments
        important_keywords = ['important', 'critical', 'essential', 'key', 'vital']
        if any(kw in content.lower() for kw in important_keywords):
            base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(query_words & content_words)
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)
    
    def _generate_analysis(self, input_text: str, memories: List[Memory]) -> str:
        """Generate analysis based on input and memories"""
        if not memories:
            return "No relevant memories found for context."
        
        # Analyze memory types
        types = [m.metadata.get('type', 'unknown') for m in memories]
        type_summary = f"Found {len(memories)} relevant memories"
        
        # Analyze CXD distribution
        cxd_types = [m.metadata.get('cxd', 'unknown') for m in memories]
        cxd_counts = {}
        for cxd in cxd_types:
            cxd_counts[cxd] = cxd_counts.get(cxd, 0) + 1
        
        # Generate analysis
        analysis = f"{type_summary}. "
        if cxd_counts:
            dominant_cxd = max(cxd_counts, key=cxd_counts.get)
            analysis += f"Context primarily relates to {dominant_cxd} aspects. "
        
        # Add temporal analysis
        if memories:
            latest = max(m.created_at for m in memories)
            oldest = min(m.created_at for m in memories)
            time_span = (latest - oldest).days
            if time_span > 0:
                analysis += f"Memories span {time_span} days of history."
        
        return analysis


# Command-line interface for testing
def main():
    """CLI for testing MCP tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MemMimic MCP Tools')
    parser.add_argument('command', choices=['remember', 'recall', 'think', 'status', 'migrate'],
                       help='Command to execute')
    parser.add_argument('--content', help='Content for remember/think')
    parser.add_argument('--query', help='Query for recall')
    parser.add_argument('--type', default='interaction', help='Memory type')
    parser.add_argument('--cxd', default='ALL', help='CXD filter')
    parser.add_argument('--limit', type=int, default=10, help='Result limit')
    
    args = parser.parse_args()
    
    # Initialize MCP
    mcp = MemMimicMCP()
    
    # Execute command
    if args.command == 'remember':
        if not args.content:
            print("Error: --content required for remember")
            return
        result = mcp.remember(args.content, args.type)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'recall':
        if not args.query:
            print("Error: --query required for recall")
            return
        results = mcp.recall_cxd(args.query, args.cxd, args.limit)
        print(json.dumps(results, indent=2, default=str))
    
    elif args.command == 'think':
        if not args.content:
            print("Error: --content required for think")
            return
        result = mcp.think_with_memory(args.content)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'status':
        result = mcp.status()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'migrate':
        result = mcp.migrate_to_markdown()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()