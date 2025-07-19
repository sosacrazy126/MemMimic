"""
AMMS Storage - Post-Migration High-Performance Storage

Simplified AMMS-only storage without migration overhead.
"""

import asyncio
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    """Memory object for AMMS storage"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    id: Optional[str] = None


class AMMSStorage:
    """
    High-performance AMMS-only storage - Post-migration architecture.
    
    Simplified storage implementation without legacy compatibility overhead.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Performance metrics
        self._metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_response_time_ms': 0.0
        }
        
        self.logger.info(f"AMMSStorage initialized - {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance_score REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Add importance_score column if it doesn't exist (migration safety)
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN importance_score REAL DEFAULT 0.5")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add metadata column if it doesn't exist (migration safety)
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN metadata TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add updated_at column if it doesn't exist (migration safety)
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN updated_at TEXT")
                # Update existing records with created_at value
                conn.execute("UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Create indexes for performance  
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_fts ON memories(content)")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def store_memory(self, memory: Memory) -> str:
        """Store memory in AMMS"""
        start_time = time.perf_counter()
        
        try:
            self._metrics['total_operations'] += 1
            
            if not memory.id:
                memory.id = f"mem_{int(time.time() * 1000000)}"
            
            with self._get_connection() as conn:
                # Convert metadata to string for storage
                metadata_str = str(memory.metadata) if memory.metadata else "{}"
                
                cursor = conn.execute(
                    """INSERT INTO memories 
                       (content, metadata, importance_score, created_at, updated_at) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        memory.content,
                        metadata_str,
                        memory.importance_score,
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat()
                    )
                )
                
                if not memory.id.startswith('mem_'):
                    memory.id = str(cursor.lastrowid)
            
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['successful_operations'] += 1
            self._update_avg_response_time(operation_time)
            
            self.logger.debug(f"Stored memory {memory.id} in {operation_time:.2f}ms")
            return memory.id
            
        except Exception as e:
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['failed_operations'] += 1
            self.logger.error(f"Failed to store memory: {e}")
            raise RuntimeError(f"Memory storage failed: {e}") from e
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID"""
        start_time = time.perf_counter()
        
        try:
            self._metrics['total_operations'] += 1
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=eval(row['metadata']) if row['metadata'] else {},
                        importance_score=row['importance_score'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self._metrics['successful_operations'] += 1
                    self._update_avg_response_time(operation_time)
                    
                    return memory
            
            return None
            
        except Exception as e:
            self._metrics['failed_operations'] += 1
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise RuntimeError(f"Memory retrieval failed: {e}") from e
    
    async def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content"""
        start_time = time.perf_counter()
        
        try:
            self._metrics['total_operations'] += 1
            
            memories = []
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM memories 
                       WHERE content LIKE ? 
                       ORDER BY importance_score DESC, created_at DESC 
                       LIMIT ?""",
                    (f"%{query}%", limit)
                )
                
                for row in cursor.fetchall():
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=eval(row['metadata']) if row['metadata'] else {},
                        importance_score=row['importance_score'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    memories.append(memory)
            
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['successful_operations'] += 1
            self._update_avg_response_time(operation_time)
            
            self.logger.debug(f"Search returned {len(memories)} results in {operation_time:.2f}ms")
            return memories
            
        except Exception as e:
            self._metrics['failed_operations'] += 1
            self.logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Memory search failed: {e}") from e
    
    async def list_memories(self, offset: int = 0, limit: int = 100) -> List[Memory]:
        """List memories with pagination"""
        try:
            memories = []
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM memories 
                       ORDER BY created_at DESC 
                       LIMIT ? OFFSET ?""",
                    (limit, offset)
                )
                
                for row in cursor.fetchall():
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=eval(row['metadata']) if row['metadata'] else {},
                        importance_score=row['importance_score'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            raise RuntimeError(f"Memory listing failed: {e}") from e
    
    async def count_memories(self) -> int:
        """Count total memories"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to count memories: {e}")
            raise RuntimeError(f"Memory count failed: {e}") from e
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise RuntimeError(f"Memory deletion failed: {e}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            'storage_type': 'amms_only',
            'metrics': self._metrics.copy(),
            'db_path': self.db_path
        }
    
    def _update_avg_response_time(self, operation_time: float):
        """Update average response time"""
        if self._metrics['successful_operations'] > 0:
            current_avg = self._metrics['avg_response_time_ms']
            count = self._metrics['successful_operations']
            self._metrics['avg_response_time_ms'] = (
                (current_avg * (count - 1) + operation_time) / count
            )
    
    async def close(self):
        """Close storage (cleanup if needed)"""
        self.logger.info("AMMSStorage closed")


def create_amms_storage(db_path: str) -> AMMSStorage:
    """Factory function to create AMMS storage"""
    return AMMSStorage(db_path)