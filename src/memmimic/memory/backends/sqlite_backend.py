"""
SQLite backend implementation for memory storage.

This is a refactored version of the original MemoryStore, now implementing
the MemoryBackend interface for use with UnifiedMemoryStore.
"""

import json
import sqlite3
import threading
import yaml
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .base import MemoryBackend
from ..memory import Memory  # Import the existing Memory class


class SQLiteBackend(MemoryBackend):
    """SQLite backend with intelligent search and thread safety."""
    
    def __init__(self, db_path: str = "memories.db"):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = None
        self.semantic_expansions = {}
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize SQLite database with tables and indexes."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA encoding = 'UTF-8'")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        
        # Create memories table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'interaction',
                confidence REAL DEFAULT 0.8,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Create performance indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type 
            ON memories(type)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at 
            ON memories(created_at DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_confidence 
            ON memories(confidence DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type_confidence 
            ON memories(type, confidence DESC)
        """)
        
        self.conn.commit()
        
        # Load semantic expansions
        self.semantic_expansions = self._load_semantic_expansions()
    
    def _load_semantic_expansions(self) -> Dict:
        """Load semantic expansions from YAML configuration."""
        possible_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'semantic_expansions.yaml',
            Path('src/memmimic/config/semantic_expansions.yaml')
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    expansions = {}
                    for category in config.values():
                        for term, related in category.items():
                            expansions[term] = related
                    return expansions
                except Exception:
                    pass
        
        # Fallback expansions
        return {
            "uncertainty": ["certainty", "doubt", "honesty", "admit", "principle"],
            "philosophy": ["principle", "wisdom", "approach", "belief"],
            "architecture": ["component", "structure", "design", "system"],
            "search": ["find", "recall", "memory", "relevant"],
            "context": ["memory", "preserve", "continuity", "remember"]
        }
    
    def add(self, content: str, memory_type: str = "interaction",
            confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a new memory to SQLite."""
        with self.lock:
            created_at = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.conn.execute(
                """INSERT INTO memories (content, type, confidence, created_at, access_count, metadata) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (content, memory_type, confidence, created_at, 0, metadata_json)
            )
            self.conn.commit()
            return cursor.lastrowid
    
    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """
        Search memories using semantic expansion and ranking.
        
        Implements the intelligent search from the original MemoryStore.
        """
        query_lower = query.lower()
        search_results = []
        
        # Expand query with semantic terms
        expanded_terms = set([query_lower])
        for base_term, expansions in self.semantic_expansions.items():
            if base_term in query_lower:
                expanded_terms.update(exp.lower() for exp in expansions)
        
        # Search with expanded terms
        with self.lock:
            # Build WHERE clause with filters
            where_clauses = []
            params = []
            
            if filters:
                if 'type' in filters:
                    where_clauses.append("type = ?")
                    params.append(filters['type'])
                if 'min_confidence' in filters:
                    where_clauses.append("confidence >= ?")
                    params.append(filters['min_confidence'])
                if 'date_from' in filters:
                    where_clauses.append("created_at >= ?")
                    params.append(filters['date_from'])
            
            where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            cursor = self.conn.execute(
                f"""SELECT * FROM memories {where_clause}
                    ORDER BY created_at DESC""",
                params
            )
            
            # Score and rank results
            for row in cursor:
                memory = self._row_to_memory(row)
                content_lower = memory.content.lower()
                
                # Calculate relevance score
                score = 0
                if query_lower in content_lower:
                    score += 10
                
                for term in expanded_terms:
                    if term in content_lower:
                        score += 5
                
                score += memory.confidence * 5
                
                if memory.type in ['synthetic', 'milestone']:
                    score += 3
                
                if score > 0:
                    search_results.append((memory, score))
        
        # Sort by score and return top results
        search_results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in search_results[:limit]]
    
    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a specific memory by ID."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            return self._row_to_memory(row) if row else None
    
    def update(self, memory_id: int, content: Optional[str] = None,
               confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory."""
        with self.lock:
            updates = []
            params = []
            
            if content is not None:
                updates.append("content = ?")
                params.append(content)
            
            if confidence is not None:
                updates.append("confidence = ?")
                params.append(confidence)
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if not updates:
                return False
            
            params.append(memory_id)
            query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
            
            cursor = self.conn.execute(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        with self.lock:
            cursor = self.conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )
            self.conn.commit()
            return cursor.rowcount > 0
    
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """Get all memories with optional filtering."""
        with self.lock:
            where_clauses = []
            params = []
            
            if filters:
                if 'type' in filters:
                    where_clauses.append("type = ?")
                    params.append(filters['type'])
                if 'min_confidence' in filters:
                    where_clauses.append("confidence >= ?")
                    params.append(filters['min_confidence'])
            
            where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            cursor = self.conn.execute(
                f"""SELECT * FROM memories {where_clause}
                    ORDER BY created_at DESC""",
                params
            )
            
            return [self._row_to_memory(row) for row in cursor]
    
    def export(self, format: str = "json") -> str:
        """Export all memories."""
        memories = self.get_all()
        
        if format == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "backend": "sqlite",
                "count": len(memories),
                "memories": [m.to_dict() for m in memories]
            }
            return json.dumps(data, indent=2)
        
        elif format == "markdown":
            lines = [
                f"# Memory Export - {datetime.now().isoformat()}",
                f"Backend: SQLite",
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
    
    def _row_to_memory(self, row) -> Memory:
        """Convert SQLite row to Memory object."""
        memory = Memory(
            content=row['content'],
            memory_type=row['type'],
            confidence=row['confidence'],
            id=row['id']
        )
        memory.created_at = row['created_at']
        memory.access_count = row['access_count']
        
        # Update access count
        with self.lock:
            self.conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (row['id'],)
            )
            self.conn.commit()
        
        return memory
    
    @property
    def backend_type(self) -> str:
        """Return backend type identifier."""
        return "sqlite"
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Return SQLite backend capabilities."""
        return {
            'semantic_search': True,  # Via semantic expansions
            'exact_search': True,
            'regex_search': False,
            'vector_operations': False,
            'transactions': True,
            'concurrent_access': True,  # Via threading locks
            'persistence': True,
            'export_import': True
        }
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None