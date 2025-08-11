#!/usr/bin/env python3
"""
Search functionality for markdown-based memories
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import re

class MarkdownMemorySearch:
    """Search engine for markdown memories"""
    
    def __init__(self, memories_dir):
        self.memories_dir = Path(memories_dir)
        self.index_path = self.memories_dir / 'index.json'
        self._load_index()
    
    def _load_index(self):
        """Load the search index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = []
    
    def search_content(self, query, limit=10):
        """Search memory content using ripgrep"""
        try:
            # Use ripgrep for fast content search
            result = subprocess.run(
                ['rg', '-l', '-i', '--max-count', str(limit), query, str(self.memories_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                memories = []
                
                for file_path in files[:limit]:
                    if file_path:
                        memory = self._load_memory_from_file(file_path)
                        if memory:
                            memories.append(memory)
                
                return memories
        except FileNotFoundError:
            # Fallback to Python search if ripgrep not available
            return self._python_search(query, limit)
        
        return []
    
    def _python_search(self, query, limit=10):
        """Fallback Python-based search"""
        query_lower = query.lower()
        results = []
        
        # Search through all markdown files
        for md_file in self.memories_dir.rglob('*.md'):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query_lower in content.lower():
                        memory = self._parse_markdown_file(content)
                        memory['file_path'] = str(md_file)
                        results.append(memory)
                        
                        if len(results) >= limit:
                            break
            except Exception:
                continue
        
        return results
    
    def _load_memory_from_file(self, file_path):
        """Load a memory from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return self._parse_markdown_file(content)
        except Exception:
            return None
    
    def _parse_markdown_file(self, content):
        """Parse markdown file with frontmatter"""
        # Extract frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    markdown_content = parts[2].strip()
                    
                    return {
                        'metadata': frontmatter,
                        'content': markdown_content
                    }
                except:
                    pass
        
        return {'content': content, 'metadata': {}}
    
    def search_by_cxd(self, cxd_type, limit=10):
        """Search memories by CXD classification"""
        results = []
        
        for entry in self.index:
            if entry.get('cxd', '').upper() == cxd_type.upper():
                # Load full memory from file
                file_pattern = f"mem_{entry['id']}_*.md"
                file_path = self.memories_dir / entry['path']
                
                for md_file in file_path.glob(file_pattern):
                    memory = self._load_memory_from_file(md_file)
                    if memory:
                        results.append(memory)
                        break
                
                if len(results) >= limit:
                    break
        
        return results
    
    def search_by_date_range(self, start_date, end_date):
        """Search memories within date range"""
        results = []
        
        for entry in self.index:
            created = datetime.fromisoformat(entry['created'])
            if start_date <= created <= end_date:
                # Load full memory
                file_pattern = f"mem_{entry['id']}_*.md"
                file_path = self.memories_dir / entry['path']
                
                for md_file in file_path.glob(file_pattern):
                    memory = self._load_memory_from_file(md_file)
                    if memory:
                        results.append(memory)
                        break
        
        return results
    
    def get_recent(self, limit=10):
        """Get most recent memories"""
        # Sort index by date
        sorted_index = sorted(self.index, key=lambda x: x['created'], reverse=True)
        
        results = []
        for entry in sorted_index[:limit]:
            file_pattern = f"mem_{entry['id']}_*.md"
            file_path = self.memories_dir / entry['path']
            
            for md_file in file_path.glob(file_pattern):
                memory = self._load_memory_from_file(md_file)
                if memory:
                    results.append(memory)
                    break
        
        return results


# Example usage
if __name__ == "__main__":
    search = MarkdownMemorySearch("/home/evilbastardxd/Desktop/tools/memmimicc/memories")
    
    # Search examples
    results = search.search_content("architecture")
    print(f"Found {len(results)} memories about 'architecture'")
    
    cxd_results = search.search_by_cxd("CONTEXT")
    print(f"Found {len(cxd_results)} CONTEXT memories")
    
    recent = search.get_recent(5)
    print(f"Got {len(recent)} recent memories")