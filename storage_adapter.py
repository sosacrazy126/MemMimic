#!/usr/bin/env python3
"""
Storage Adapter Pattern for SQLite/Markdown dual support
Allows seamless transition between storage backends
"""

import sqlite3
import json
import yaml
import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import fcntl  # For file locking on Unix
import tempfile
import shutil

class Memory:
    """Universal memory representation"""
    def __init__(self, content: str, metadata: Dict = None, 
                 importance: float = 0.5, memory_id: str = None):
        self.id = memory_id or f"mem_{int(datetime.now().timestamp() * 1000000)}"
        self.content = content
        self.metadata = metadata or {}
        self.importance = importance
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'importance': self.importance,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        memory = cls(
            content=data['content'],
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5),
            memory_id=data.get('id')
        )
        if 'created_at' in data:
            memory.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            memory.updated_at = datetime.fromisoformat(data['updated_at'])
        return memory


class StorageAdapter(ABC):
    """Abstract base class for storage adapters"""
    
    @abstractmethod
    def store(self, memory: Memory) -> str:
        """Store a memory and return its ID"""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content"""
        pass
    
    @abstractmethod
    def update(self, memory_id: str, memory: Memory) -> bool:
        """Update an existing memory"""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        pass
    
    @abstractmethod
    def get_all(self, limit: int = None) -> List[Memory]:
        """Get all memories"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Count total memories"""
        pass


class SQLiteAdapter(StorageAdapter):
    """SQLite storage adapter"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
    
    def store(self, memory: Memory) -> str:
        """Store memory in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories (id, content, metadata, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                json.dumps(memory.metadata),
                memory.importance,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat()
            ))
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            
            if row:
                return Memory(
                    content=row['content'],
                    metadata=json.loads(row['metadata'] or '{}'),
                    importance=row['importance'],
                    memory_id=row['id']
                )
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM memories 
                WHERE content LIKE ? 
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
            
            memories = []
            for row in rows:
                memory = Memory(
                    content=row['content'],
                    metadata=json.loads(row['metadata'] or '{}'),
                    importance=row['importance'],
                    memory_id=row['id']
                )
                memory.created_at = datetime.fromisoformat(row['created_at'])
                memory.updated_at = datetime.fromisoformat(row['updated_at'])
                memories.append(memory)
            
            return memories
    
    def update(self, memory_id: str, memory: Memory) -> bool:
        """Update memory in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE memories 
                SET content = ?, metadata = ?, importance = ?, updated_at = ?
                WHERE id = ?
            """, (
                memory.content,
                json.dumps(memory.metadata),
                memory.importance,
                datetime.now().isoformat(),
                memory_id
            ))
            return cursor.rowcount > 0
    
    def delete(self, memory_id: str) -> bool:
        """Delete memory from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0
    
    def get_all(self, limit: int = None) -> List[Memory]:
        """Get all memories from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM memories ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            rows = conn.execute(query).fetchall()
            memories = []
            for row in rows:
                memory = Memory(
                    content=row['content'],
                    metadata=json.loads(row['metadata'] or '{}'),
                    importance=row['importance'],
                    memory_id=row['id']
                )
                memory.created_at = datetime.fromisoformat(row['created_at'])
                memory.updated_at = datetime.fromisoformat(row['updated_at'])
                memories.append(memory)
            
            return memories
    
    def count(self) -> int:
        """Count memories in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


class MarkdownAdapter(StorageAdapter):
    """Markdown file storage adapter"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.memories_dir = self.base_dir / 'memories'
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memories_dir / 'index.json'
        self.lock_file = self.memories_dir / '.lock'
        self._load_index()
    
    def _load_index(self):
        """Load or create the index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                # Handle migration from list format to dict format
                if isinstance(data, list):
                    self.index = {str(item['id']): item for item in data}
                    # Update paths to include full filename if needed
                    for item_id, item in self.index.items():
                        if 'path' in item and not item['path'].endswith('.md'):
                            # Need to find the actual file
                            date_path = self.memories_dir / item['path']
                            for md_file in date_path.glob(f"mem_{item_id}_*.md"):
                                item['path'] = str(md_file.relative_to(self.memories_dir))
                                break
                    self._save_index()
                else:
                    self.index = data
        else:
            self.index = {}
            self._save_index()
    
    def _save_index(self):
        """Save the index with atomic write"""
        temp_path = self.index_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        temp_path.replace(self.index_path)
    
    def _get_file_path(self, memory: Memory) -> Path:
        """Generate file path for a memory"""
        date = memory.created_at
        dir_path = self.memories_dir / f"{date.year:04d}" / f"{date.month:02d}" / f"{date.day:02d}"
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract CXD type for filename
        cxd = memory.metadata.get('cxd', 'unknown').lower()
        
        # Create safe filename
        content_hash = hashlib.md5(memory.content.encode()).hexdigest()[:8]
        filename = f"{memory.id}_{cxd}_{content_hash}.md"
        
        return dir_path / filename
    
    def _memory_to_markdown(self, memory: Memory) -> str:
        """Convert memory to markdown with frontmatter"""
        frontmatter = {
            'id': memory.id,
            'importance': memory.importance,
            'created': memory.created_at.isoformat(),
            'updated': memory.updated_at.isoformat(),
        }
        
        # Add metadata to frontmatter
        for key, value in memory.metadata.items():
            frontmatter[key] = value
        
        yaml_front = yaml.dump(frontmatter, default_flow_style=False)
        
        return f"---\n{yaml_front}---\n\n# Memory {memory.id}\n\n{memory.content}"
    
    def _markdown_to_memory(self, content: str) -> Memory:
        """Parse markdown file to memory"""
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                markdown_content = parts[2].strip()
                
                # Remove the header line if present
                if markdown_content.startswith(f"# Memory"):
                    markdown_content = '\n'.join(markdown_content.split('\n')[2:]).strip()
                
                memory = Memory(
                    content=markdown_content,
                    metadata={k: v for k, v in frontmatter.items() 
                             if k not in ['id', 'importance', 'created', 'updated']},
                    importance=frontmatter.get('importance', 0.5),
                    memory_id=frontmatter.get('id')
                )
                
                if 'created' in frontmatter:
                    memory.created_at = datetime.fromisoformat(frontmatter['created'])
                if 'updated' in frontmatter:
                    memory.updated_at = datetime.fromisoformat(frontmatter['updated'])
                
                return memory
        
        # Fallback for malformed files
        return Memory(content=content)
    
    def _acquire_lock(self):
        """Acquire file lock for write operations"""
        self.lock_file.touch(exist_ok=True)
        lock_fd = os.open(str(self.lock_file), os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd
    
    def _release_lock(self, lock_fd):
        """Release file lock"""
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    
    def store(self, memory: Memory) -> str:
        """Store memory as markdown file"""
        lock_fd = self._acquire_lock()
        try:
            file_path = self._get_file_path(memory)
            content = self._memory_to_markdown(memory)
            
            # Atomic write
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            temp_path.replace(file_path)
            
            # Update index
            self.index[memory.id] = {
                'path': str(file_path.relative_to(self.memories_dir)),
                'created': memory.created_at.isoformat(),
                'importance': memory.importance,
                'cxd': memory.metadata.get('cxd', 'unknown')
            }
            self._save_index()
            
            return memory.id
        finally:
            self._release_lock(lock_fd)
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory from markdown file"""
        if memory_id in self.index:
            file_path = self.memories_dir / self.index[memory_id]['path']
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return self._markdown_to_memory(content)
        return None
    
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories in markdown files"""
        query_lower = query.lower()
        results = []
        
        for memory_id, info in self.index.items():
            if len(results) >= limit:
                break
            
            file_path = self.memories_dir / info['path']
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query_lower in content.lower():
                        memory = self._markdown_to_memory(content)
                        results.append(memory)
        
        # Sort by importance and recency
        results.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        return results[:limit]
    
    def update(self, memory_id: str, memory: Memory) -> bool:
        """Update memory in markdown file"""
        if memory_id in self.index:
            lock_fd = self._acquire_lock()
            try:
                file_path = self.memories_dir / self.index[memory_id]['path']
                memory.updated_at = datetime.now()
                content = self._memory_to_markdown(memory)
                
                # Atomic write
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                temp_path.replace(file_path)
                
                # Update index
                self.index[memory_id]['importance'] = memory.importance
                self._save_index()
                
                return True
            finally:
                self._release_lock(lock_fd)
        return False
    
    def delete(self, memory_id: str) -> bool:
        """Delete memory markdown file"""
        if memory_id in self.index:
            lock_fd = self._acquire_lock()
            try:
                file_path = self.memories_dir / self.index[memory_id]['path']
                if file_path.exists():
                    file_path.unlink()
                
                del self.index[memory_id]
                self._save_index()
                return True
            finally:
                self._release_lock(lock_fd)
        return False
    
    def get_all(self, limit: int = None) -> List[Memory]:
        """Get all memories from markdown files"""
        memories = []
        count = 0
        
        # Sort by created date (newest first)
        sorted_items = sorted(self.index.items(), 
                            key=lambda x: x[1]['created'], 
                            reverse=True)
        
        for memory_id, info in sorted_items:
            if limit and count >= limit:
                break
            
            memory = self.retrieve(memory_id)
            if memory:
                memories.append(memory)
                count += 1
        
        return memories
    
    def count(self) -> int:
        """Count memories in index"""
        return len(self.index)


class HybridAdapter(StorageAdapter):
    """Hybrid adapter that can read from SQLite and write to Markdown during migration"""
    
    def __init__(self, sqlite_path: str, markdown_dir: str, write_to='both'):
        self.sqlite = SQLiteAdapter(sqlite_path)
        self.markdown = MarkdownAdapter(markdown_dir)
        self.write_to = write_to  # 'sqlite', 'markdown', or 'both'
    
    def store(self, memory: Memory) -> str:
        """Store to configured backend(s)"""
        if self.write_to == 'sqlite':
            return self.sqlite.store(memory)
        elif self.write_to == 'markdown':
            return self.markdown.store(memory)
        else:  # both
            self.sqlite.store(memory)
            return self.markdown.store(memory)
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Try markdown first, fallback to SQLite"""
        memory = self.markdown.retrieve(memory_id)
        if not memory:
            memory = self.sqlite.retrieve(memory_id)
        return memory
    
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search both backends and merge results"""
        md_results = self.markdown.search(query, limit)
        sql_results = self.sqlite.search(query, limit)
        
        # Merge and deduplicate
        seen_ids = set()
        merged = []
        
        for memory in md_results + sql_results:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                merged.append(memory)
        
        # Sort and limit
        merged.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        return merged[:limit]
    
    def update(self, memory_id: str, memory: Memory) -> bool:
        """Update in both backends"""
        if self.write_to == 'sqlite':
            return self.sqlite.update(memory_id, memory)
        elif self.write_to == 'markdown':
            return self.markdown.update(memory_id, memory)
        else:  # both
            md_success = self.markdown.update(memory_id, memory)
            sql_success = self.sqlite.update(memory_id, memory)
            return md_success or sql_success
    
    def delete(self, memory_id: str) -> bool:
        """Delete from both backends"""
        md_success = self.markdown.delete(memory_id)
        sql_success = self.sqlite.delete(memory_id)
        return md_success or sql_success
    
    def get_all(self, limit: int = None) -> List[Memory]:
        """Get all from both backends, deduplicated"""
        md_memories = self.markdown.get_all()
        sql_memories = self.sqlite.get_all()
        
        # Deduplicate
        seen_ids = set()
        merged = []
        
        for memory in md_memories + sql_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                merged.append(memory)
        
        # Sort and limit
        merged.sort(key=lambda m: m.created_at, reverse=True)
        if limit:
            merged = merged[:limit]
        
        return merged
    
    def count(self) -> int:
        """Count unique memories across both backends"""
        md_ids = set(self.markdown.index.keys())
        sql_count = self.sqlite.count()
        
        # This is approximate - for exact count would need to query SQL IDs
        return len(md_ids) + sql_count


# Factory function for easy instantiation
def create_storage_adapter(storage_type: str = 'hybrid', **kwargs) -> StorageAdapter:
    """
    Create a storage adapter
    
    Args:
        storage_type: 'sqlite', 'markdown', or 'hybrid'
        **kwargs: Backend-specific arguments
    """
    if storage_type == 'sqlite':
        return SQLiteAdapter(kwargs.get('db_path', 'memmimic.db'))
    elif storage_type == 'markdown':
        return MarkdownAdapter(kwargs.get('base_dir', '.'))
    elif storage_type == 'hybrid':
        return HybridAdapter(
            kwargs.get('sqlite_path', 'memmimic.db'),
            kwargs.get('markdown_dir', '.'),
            kwargs.get('write_to', 'both')
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


# Example usage
if __name__ == "__main__":
    # Create hybrid adapter for migration
    adapter = create_storage_adapter('hybrid', 
                                    sqlite_path='memmimic.db',
                                    markdown_dir='.',
                                    write_to='both')
    
    # Store a memory
    memory = Memory(
        content="This is a test memory about architecture",
        metadata={'cxd': 'CONTEXT', 'tags': ['test', 'architecture']},
        importance=0.8
    )
    
    memory_id = adapter.store(memory)
    print(f"Stored memory: {memory_id}")
    
    # Search for it
    results = adapter.search("architecture")
    print(f"Found {len(results)} results")
    
    # Count total
    total = adapter.count()
    print(f"Total memories: {total}")