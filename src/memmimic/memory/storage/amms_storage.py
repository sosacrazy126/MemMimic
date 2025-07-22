"""
AMMS Storage - Post-Migration High-Performance Storage

Simplified AMMS-only storage without migration overhead.
"""

import asyncio
import json
import logging
import queue
import sqlite3
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...config import get_performance_config
from ...errors import (
    MemoryStorageError, MemoryRetrievalError, DatabaseError,
    handle_errors, with_error_context, get_error_logger
)


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
    
    def __init__(self, db_path: str, pool_size: Optional[int] = None):
        self.db_path = db_path
        
        # Load configuration
        self.config = get_performance_config()
        db_config = self.config.database_config
        
        # Event for loop readiness synchronization
        self._loop_ready = threading.Event()
        
        self.pool_size = pool_size if pool_size is not None else db_config.get('connection_pool_size', 5)
        self.connection_timeout = db_config.get('connection_timeout', 5.0)
        self.enable_wal = db_config.get('wal_mode', True)
        self.cache_size = db_config.get('cache_size', 10000)
        
        self.logger = get_error_logger("amms_storage")
        
        # Initialize metrics first
        self._metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_response_time_ms': 0.0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Initialize connection pool before database initialization
        self._connection_pool = queue.Queue(maxsize=self.pool_size)
        self._pool_lock = threading.Lock()
        self._initialize_connection_pool()
        
        self._init_database()
        
        # Shared event loop for sync operations
        self._loop = None
        self._loop_thread = None
        self._loop_lock = threading.Lock()
        
        self.logger.info(f"AMMSStorage initialized - {db_path} with pool size {pool_size}")
    
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
    
    def _initialize_connection_pool(self):
        """Initialize the connection pool with pre-configured connections"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Configure connection based on settings
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA cache_size={self.cache_size}")
            
            temp_store = self.config.get('database.temp_store', 'MEMORY')
            conn.execute(f"PRAGMA temp_store={temp_store}")
            
            self._connection_pool.put(conn)
    
    def _get_connection_from_pool(self):
        """Get connection from pool with timeout"""
        try:
            conn = self._connection_pool.get(timeout=self.connection_timeout)
            self._metrics['pool_hits'] += 1
            return conn
        except queue.Empty:
            # Pool exhausted, create temporary connection
            self._metrics['pool_misses'] += 1
            self.logger.warning("Connection pool exhausted, creating temporary connection")
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Apply same configuration to temporary connection
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA cache_size={self.cache_size}")
            
            return conn
    
    def _return_connection_to_pool(self, conn):
        """Return connection to pool or close if pool is full"""
        try:
            self._connection_pool.put_nowait(conn)
        except queue.Full:
            # Pool is full, close the connection
            conn.close()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup and pooling"""
        conn = self._get_connection_from_pool()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            # Return to pool instead of closing
            self._return_connection_to_pool(conn)
    
    @handle_errors(catch=[sqlite3.Error, json.JSONDecodeError], reraise=True)
    async def store_memory(self, memory: Memory) -> str:
        """Store memory in AMMS"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="store_memory",
            component="amms_storage",
            metadata={"memory_content_length": len(memory.content) if memory.content else 0}
        ):
            self._metrics['total_operations'] += 1
            
            if not memory.id:
                memory.id = f"mem_{int(time.time() * 1000000)}"
            
            with self._get_connection() as conn:
                # Convert metadata to JSON string for safe storage
                try:
                    metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"Failed to serialize metadata, using empty dict: {e}")
                    metadata_str = "{}"
                
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
    
    @handle_errors(catch=[sqlite3.Error, json.JSONDecodeError], reraise=True)
    async def retrieve_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="retrieve_memory",
            component="amms_storage",
            metadata={"memory_id": memory_id}
        ):
            self._metrics['total_operations'] += 1
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Safely parse metadata JSON
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except (json.JSONDecodeError, TypeError):
                        self.logger.warning(f"Invalid metadata for memory {row['id']}, using empty dict")
                        metadata = {}
                    
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=metadata,
                        importance_score=row['importance_score'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self._metrics['successful_operations'] += 1
                    self._update_avg_response_time(operation_time)
                    
                    return memory
            
            return None
    
    @handle_errors(catch=[sqlite3.Error, json.JSONDecodeError], reraise=True)
    async def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="search_memories",
            component="amms_storage",
            metadata={"query_length": len(query), "limit": limit}
        ):
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
                    # Safely parse metadata JSON
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except (json.JSONDecodeError, TypeError):
                        self.logger.warning(f"Invalid metadata for memory {row['id']}, using empty dict")
                        metadata = {}
                    
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=metadata,
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
                    # Safely parse metadata JSON
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except (json.JSONDecodeError, TypeError):
                        self.logger.warning(f"Invalid metadata for memory {row['id']}, using empty dict")
                        metadata = {}
                    
                    memory = Memory(
                        id=str(row['id']),
                        content=row['content'],
                        metadata=metadata,
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
        pool_stats = {
            'pool_size': self.pool_size,
            'available_connections': self._connection_pool.qsize(),
            'pool_utilization': (self.pool_size - self._connection_pool.qsize()) / self.pool_size
        }
        
        return {
            'storage_type': 'amms_only',
            'metrics': self._metrics.copy(),
            'connection_pool': pool_stats,
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
    
    def _get_or_create_loop(self):
        """Get or create shared event loop for sync operations"""
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                # Create new loop in a separate thread
                def run_loop():
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                    # Signal that loop is ready
                    self._loop_ready.set()
                    self._loop.run_forever()
                
                self._loop_thread = threading.Thread(target=run_loop, daemon=True)
                self._loop_thread.start()
                
                # Wait for loop to be ready (non-blocking)
                self._loop_ready.wait(timeout=5.0)
                if not self._loop_ready.is_set():
                    raise RuntimeError("Failed to initialize event loop within timeout")
            
            return self._loop
    
    def _run_async_safe(self, coro):
        """Safely run async coroutine from sync context"""
        try:
            # Try to get current loop
            current_loop = asyncio.get_running_loop()
            # If we're already in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No current loop, safe to use shared loop
            loop = self._get_or_create_loop()
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Sync wrapper for search_memories for compatibility"""
        return self._run_async_safe(self.search_memories(query, limit))
    
    def get_all(self, limit: int = 1000) -> List[Memory]:
        """Sync wrapper for list_memories for compatibility"""
        return self._run_async_safe(self.list_memories(0, limit))
    
    def list_all(self, limit: int = 1000) -> List[Memory]:
        """Sync wrapper for list_memories for compatibility (alias)"""
        return self.get_all(limit)
    
    def add(self, memory: Memory) -> str:
        """Sync wrapper for store_memory for compatibility"""
        return self._run_async_safe(self.store_memory(memory))
    
    def delete(self, memory_id: str) -> bool:
        """Sync wrapper for delete_memory for compatibility"""
        return self._run_async_safe(self.delete_memory(memory_id))
    
    def update_memory(self, memory_id: str, memory: Memory) -> bool:
        """Sync wrapper for updating memory - stores new version"""
        async def _update():
            # Delete old and store new (update pattern)
            deleted = await self.delete_memory(memory_id)
            if deleted:
                memory.id = memory_id
                await self.store_memory(memory)
                return True
            return False
        
        return self._run_async_safe(_update())
    
    async def close(self):
        """Close storage (cleanup if needed)"""
        # Stop the shared event loop
        with self._loop_lock:
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread and self._loop_thread.is_alive():
                    self._loop_thread.join(timeout=1.0)
                self._loop = None
                self._loop_thread = None
        
        # Close all pooled connections
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        self.logger.info("AMMSStorage closed - all connections cleaned up")


def create_amms_storage(db_path: str) -> AMMSStorage:
    """Factory function to create AMMS storage"""
    return AMMSStorage(db_path)