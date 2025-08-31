"""
FileSystem backend implementation for memory storage.

Stores each memory as a markdown file with YAML frontmatter.
Perfect for human-readable backups, version control, and portability.
"""

import os
import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import hashlib

from .base import MemoryBackend
from ..memory import Memory


class FileSystemBackend(MemoryBackend):
    """
    FileSystem backend that stores memories as markdown files.
    
    Benefits:
    - Human-readable format
    - Git-friendly (can track changes)
    - Portable (just files)
    - Easy backup/restore
    - Can be edited manually
    """
    
    def __init__(self, base_path: str = "~/.memmimic/memories"):
        """
        Initialize FileSystem backend.
        
        Args:
            base_path: Base directory for memory files
        """
        self.base_path = Path(base_path).expanduser()
        self.memories_path = self.base_path / "memories"
        self.index_file = self.base_path / "index.json"
        self.next_id = 1
        self.index = {}
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize filesystem structure and load index."""
        # Create directory structure
        self.memories_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for memory types
        for memory_type in ['interaction', 'milestone', 'socratic', 'reflection', 'synthetic']:
            (self.memories_path / memory_type).mkdir(exist_ok=True)
        
        # Load or create index
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                self.index = index_data.get('memories', {})
                self.next_id = index_data.get('next_id', 1)
        else:
            self._save_index()
    
    def _save_index(self):
        """Save the index to disk."""
        index_data = {
            'memories': self.index,
            'next_id': self.next_id,
            'updated': datetime.now().isoformat()
        }
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _generate_filename(self, memory_id: int, memory_type: str, created_at: str) -> str:
        """Generate a filename for a memory."""
        # Use timestamp and ID for unique, sortable filenames
        timestamp = created_at.split('T')[0]  # Get date part
        safe_type = re.sub(r'[^a-z0-9_]', '_', memory_type.lower())
        return f"{timestamp}_{memory_id:06d}_{safe_type}.md"
    
    def _memory_to_markdown(self, memory_id: int, content: str, memory_type: str,
                            confidence: float, created_at: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Convert memory to markdown with YAML frontmatter."""
        # Prepare frontmatter
        frontmatter = {
            'id': memory_id,
            'type': memory_type,
            'confidence': confidence,
            'created_at': created_at,
            'access_count': 0
        }
        
        if metadata:
            frontmatter['metadata'] = metadata
        
        # Create markdown content
        markdown = "---\n"
        markdown += yaml.dump(frontmatter, default_flow_style=False)
        markdown += "---\n\n"
        markdown += content
        
        return markdown
    
    def _parse_markdown_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a markdown file with YAML frontmatter."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                else:
                    return None
            else:
                return None
            
            # Build memory dict
            memory_dict = {
                'id': frontmatter.get('id'),
                'content': body,
                'type': frontmatter.get('type', 'interaction'),
                'confidence': frontmatter.get('confidence', 0.8),
                'created_at': frontmatter.get('created_at'),
                'access_count': frontmatter.get('access_count', 0),
                'metadata': frontmatter.get('metadata', {})
            }
            
            return memory_dict
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def add(self, content: str, memory_type: str = "interaction",
            confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a new memory as a markdown file."""
        memory_id = self.next_id
        self.next_id += 1
        
        created_at = datetime.now().isoformat()
        filename = self._generate_filename(memory_id, memory_type, created_at)
        file_path = self.memories_path / memory_type / filename
        
        # Create markdown content
        markdown = self._memory_to_markdown(
            memory_id, content, memory_type, confidence, created_at, metadata
        )
        
        # Write file
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        # Update index
        self.index[str(memory_id)] = {
            'path': str(file_path.relative_to(self.base_path)),
            'type': memory_type,
            'created_at': created_at,
            'confidence': confidence
        }
        self._save_index()
        
        return memory_id
    
    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """
        Search memories using simple grep-like search.
        
        Note: This is less efficient than SQLite but more transparent.
        """
        query_lower = query.lower()
        results = []
        
        # Search through all memory files
        for memory_id_str, info in self.index.items():
            file_path = self.base_path / info['path']
            
            # Apply filters
            if filters:
                if 'type' in filters and info['type'] != filters['type']:
                    continue
                if 'min_confidence' in filters and info['confidence'] < filters['min_confidence']:
                    continue
            
            # Parse file and search content
            memory_dict = self._parse_markdown_file(file_path)
            if memory_dict and query_lower in memory_dict['content'].lower():
                memory = Memory(
                    content=memory_dict['content'],
                    memory_type=memory_dict['type'],
                    confidence=memory_dict['confidence'],
                    id=memory_dict['id']
                )
                memory.created_at = memory_dict['created_at']
                memory.access_count = memory_dict['access_count']
                results.append(memory)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a specific memory by ID."""
        memory_id_str = str(memory_id)
        if memory_id_str not in self.index:
            return None
        
        file_path = self.base_path / self.index[memory_id_str]['path']
        memory_dict = self._parse_markdown_file(file_path)
        
        if memory_dict:
            memory = Memory(
                content=memory_dict['content'],
                memory_type=memory_dict['type'],
                confidence=memory_dict['confidence'],
                id=memory_dict['id']
            )
            memory.created_at = memory_dict['created_at']
            memory.access_count = memory_dict['access_count']
            return memory
        
        return None
    
    def update(self, memory_id: int, content: Optional[str] = None,
               confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory file."""
        memory_id_str = str(memory_id)
        if memory_id_str not in self.index:
            return False
        
        file_path = self.base_path / self.index[memory_id_str]['path']
        memory_dict = self._parse_markdown_file(file_path)
        
        if not memory_dict:
            return False
        
        # Update fields
        if content is not None:
            memory_dict['content'] = content
        if confidence is not None:
            memory_dict['confidence'] = confidence
            self.index[memory_id_str]['confidence'] = confidence
        if metadata is not None:
            memory_dict['metadata'] = metadata
        
        # Rewrite file
        markdown = self._memory_to_markdown(
            memory_dict['id'],
            memory_dict['content'],
            memory_dict['type'],
            memory_dict['confidence'],
            memory_dict['created_at'],
            memory_dict.get('metadata')
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        self._save_index()
        return True
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory file."""
        memory_id_str = str(memory_id)
        if memory_id_str not in self.index:
            return False
        
        file_path = self.base_path / self.index[memory_id_str]['path']
        
        # Delete file
        if file_path.exists():
            file_path.unlink()
        
        # Remove from index
        del self.index[memory_id_str]
        self._save_index()
        
        return True
    
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """Get all memories from filesystem."""
        memories = []
        
        for memory_id_str, info in self.index.items():
            # Apply filters
            if filters:
                if 'type' in filters and info['type'] != filters['type']:
                    continue
                if 'min_confidence' in filters and info['confidence'] < filters['min_confidence']:
                    continue
            
            memory = self.get(int(memory_id_str))
            if memory:
                memories.append(memory)
        
        # Sort by created_at descending
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories
    
    def export(self, format: str = "json") -> str:
        """Export all memories."""
        memories = self.get_all()
        
        if format == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "backend": "filesystem",
                "base_path": str(self.base_path),
                "count": len(memories),
                "memories": [m.to_dict() for m in memories]
            }
            return json.dumps(data, indent=2)
        
        elif format == "markdown":
            # For filesystem, we can create a single concatenated markdown
            lines = [
                f"# Memory Export - {datetime.now().isoformat()}",
                f"Backend: FileSystem",
                f"Path: {self.base_path}",
                f"Total: {len(memories)} memories",
                "",
            ]
            
            for memory in memories:
                lines.extend([
                    f"## Memory {memory.id}",
                    f"- Type: {memory.type}",
                    f"- Confidence: {memory.confidence}",
                    f"- Created: {memory.created_at}",
                    "",
                    memory.content,
                    "",
                    "---",
                    ""
                ])
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_data(self, data: str, format: str = "json") -> int:
        """Import memories from exported data."""
        count = 0
        
        if format == "json":
            import_data = json.loads(data)
            memories = import_data.get("memories", [])
            
            for mem_dict in memories:
                self.add(
                    content=mem_dict["content"],
                    memory_type=mem_dict.get("type", "interaction"),
                    confidence=mem_dict.get("confidence", 0.8),
                    metadata=mem_dict.get("metadata")
                )
                count += 1
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        return count
    
    @property
    def backend_type(self) -> str:
        """Return backend type identifier."""
        return "filesystem"
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Return FileSystem backend capabilities."""
        return {
            'semantic_search': False,  # Simple text search only
            'exact_search': True,
            'regex_search': False,  # Could be added
            'vector_operations': False,
            'transactions': False,
            'concurrent_access': False,  # File locking not implemented
            'persistence': True,
            'export_import': True,
            'human_readable': True,  # Key advantage!
            'version_control': True,  # Git-friendly
            'manual_edit': True  # Can edit files directly
        }
    
    def close(self) -> None:
        """Save index on close."""
        self._save_index()