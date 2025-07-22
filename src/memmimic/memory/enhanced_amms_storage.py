"""
Enhanced AMMS Storage for MemMimic v2.0
Extends existing AMMS storage with dual-layer support and performance optimization.
"""

import asyncio
import json
import sqlite3
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import lru_cache

from ..errors import (
    MemoryStorageError, MemoryRetrievalError, DatabaseError,
    handle_errors, with_error_context, get_error_logger
)
from .storage.amms_storage import AMMSStorage
from .enhanced_memory import EnhancedMemory
from ..telemetry.integration import TelemetryMixin, storage_telemetry


class EnhancedAMMSStorage(AMMSStorage, TelemetryMixin):
    """
    Enhanced AMMS Storage preserving existing high-performance architecture.
    
    Extends AMMSStorage with:
    - v2.0 database schema with dual-layer fields
    - Performance-optimized indexes for <5ms summary retrieval
    - LRU cache for ultra-fast summary access
    - Safe schema migration with backward compatibility
    - Governance metadata tracking
    """
    
    def __init__(self, db_path: str, pool_size: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Enhanced AMMS Storage with v2.0 capabilities"""
        # Initialize parent with existing connection pooling and performance optimizations
        super().__init__(db_path, pool_size)
        
        # Enhanced configuration
        self.config_v2 = config or {}
        self.cache_enabled = self.config_v2.get('enable_summary_cache', True)
        self.cache_size = self.config_v2.get('summary_cache_size', 1000)
        
        # Performance tracking for v2.0 features
        self._v2_metrics = {
            'summary_retrievals': 0,
            'full_context_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'governance_validations': 0,
            'schema_migrations': 0
        }
        
        # High-performance summary cache
        if self.cache_enabled:
            self.summary_cache = {}
            self._cache_lock = threading.RLock()
            self._cache_access_order = []  # For LRU implementation
        else:
            self.summary_cache = None
        
        # Initialize enhanced schema (safe migration)
        self._init_enhanced_schema()
        
        self.logger.info(f"Enhanced AMMS Storage initialized with v2.0 capabilities")
    
    def _init_enhanced_schema(self):
        """Safely extend existing schema with v2.0 enhancements"""
        # Ensure base schema exists first
        # (This is already done by parent __init__)
        
        with self._get_connection() as conn:
            # Track schema migrations
            self._v2_metrics['schema_migrations'] += 1
            
            # Add v2.0 columns with safe ALTER TABLE operations
            v2_schema_operations = [
                # Core dual-layer fields
                ("ALTER TABLE memories ADD COLUMN summary TEXT", "summary support"),
                ("ALTER TABLE memories ADD COLUMN full_context TEXT", "full context storage"),
                ("ALTER TABLE memories ADD COLUMN tags TEXT DEFAULT '[]'", "tag system"),
                
                # Governance and metrics
                ("ALTER TABLE memories ADD COLUMN governance_status TEXT DEFAULT 'approved'", "governance tracking"),
                ("ALTER TABLE memories ADD COLUMN context_size INTEGER DEFAULT 0", "size metrics"),
                ("ALTER TABLE memories ADD COLUMN tag_count INTEGER DEFAULT 0", "tag metrics"),
                
                # Performance optimization
                ("ALTER TABLE memories ADD COLUMN summary_hash TEXT", "summary deduplication"),
                ("ALTER TABLE memories ADD COLUMN context_hash TEXT", "context integrity"),
                ("ALTER TABLE memories ADD COLUMN last_accessed TIMESTAMP", "access tracking"),
            ]
            
            # Execute schema operations with error handling
            for sql, description in v2_schema_operations:
                try:
                    conn.execute(sql)
                    self.logger.debug(f"Added v2.0 schema: {description}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        # Column already exists - this is fine for migrations
                        continue
                    else:
                        self.logger.error(f"Failed to add {description}: {e}")
                        raise DatabaseError(f"Schema migration failed for {description}: {e}") from e
            
            # Create performance-optimized indexes
            v2_indexes = [
                # Summary layer optimization (<5ms target)
                "CREATE INDEX IF NOT EXISTS idx_summary_fast ON memories(id, summary, tags) WHERE summary IS NOT NULL",
                
                # Governance queries optimization
                "CREATE INDEX IF NOT EXISTS idx_governance ON memories(governance_status, created_at)",
                
                # Tag-based retrieval optimization (JSON support)
                "CREATE INDEX IF NOT EXISTS idx_tags_json ON memories(tags) WHERE tags != '[]'",
                
                # Context size governance and performance
                "CREATE INDEX IF NOT EXISTS idx_context_metrics ON memories(context_size, tag_count)",
                
                # Access pattern optimization for caching
                "CREATE INDEX IF NOT EXISTS idx_access_patterns ON memories(last_accessed DESC, importance_score DESC)",
                
                # Hash-based deduplication
                "CREATE INDEX IF NOT EXISTS idx_summary_hash ON memories(summary_hash) WHERE summary_hash IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_context_hash ON memories(context_hash) WHERE context_hash IS NOT NULL",
            ]
            
            for index_sql in v2_indexes:
                try:
                    conn.execute(index_sql)
                    self.logger.debug(f"Created v2.0 index: {index_sql.split()[5]}")
                except sqlite3.Error as e:
                    self.logger.warning(f"Index creation warning: {e}")
    
    @handle_errors(catch=[sqlite3.Error, json.JSONDecodeError], reraise=True)
    @storage_telemetry("store_enhanced_memory_optimized")
    async def store_enhanced_memory_optimized(self, memory: EnhancedMemory) -> str:
        """Store enhanced memory with <15ms performance target (including governance)"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="store_enhanced_memory",
            component="enhanced_amms_storage",
            metadata={
                "context_size": memory.context_size,
                "tag_count": memory.tag_count,
                "governance_status": memory.governance_status
            }
        ):
            self._metrics['total_operations'] += 1
            
            # Update access time for cache optimization
            memory.updated_at = datetime.now()
            access_time = datetime.now().isoformat()
            
            # Note: ID will be set from database after insertion (following parent pattern)
            
            # Use existing AMMS connection pooling for performance
            with self._get_connection() as conn:
                # Prepare enhanced data for storage
                try:
                    metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"
                    tags_str = json.dumps(memory.tags) if memory.tags else "[]"
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"Failed to serialize enhanced data, using defaults: {e}")
                    metadata_str = "{}"
                    tags_str = "[]"
                
                # Enhanced INSERT with all v2.0 fields
                cursor = conn.execute("""
                    INSERT INTO memories (
                        content, summary, full_context, tags, metadata,
                        importance_score, governance_status, context_size, tag_count,
                        summary_hash, context_hash, created_at, updated_at, last_accessed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.content,
                    memory.summary,
                    memory.full_context,
                    tags_str,
                    metadata_str,
                    memory.importance_score,
                    memory.governance_status,
                    memory.context_size,
                    memory.tag_count,
                    memory.summary_hash,
                    memory.context_hash,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    access_time
                ))
                
                # Use database-generated ID (following parent class pattern)
                memory.id = str(cursor.lastrowid)
            
            # Update performance metrics
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['successful_operations'] += 1
            self._update_avg_response_time(operation_time)
            
            # Cache summary for fast retrieval if cache is enabled
            if self.cache_enabled and memory.summary:
                self._cache_summary(memory.id, memory.summary)
            
            self.logger.debug(f"Stored enhanced memory {memory.id} in {operation_time:.2f}ms")
            return memory.id
    
    @storage_telemetry("retrieve_summary_optimized")
    async def retrieve_summary_optimized(self, memory_id: str) -> Optional[str]:
        """Ultra-fast summary retrieval with <5ms target"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="retrieve_summary_optimized",
            component="enhanced_amms_storage",
            metadata={"memory_id": memory_id}
        ):
            self._metrics['total_operations'] += 1
            self._v2_metrics['summary_retrievals'] += 1
            
            # Check cache first (sub-millisecond performance)
            if self.cache_enabled:
                cached_summary = self._get_cached_summary(memory_id)
                if cached_summary is not None:
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self._v2_metrics['cache_hits'] += 1
                    # Record cache hit telemetry
                    self._record_storage_operation(
                        "retrieve_summary_optimized",
                        operation_time,
                        context_size=len(cached_summary),
                        cache_hit=True,
                        success=True
                    )
                    self.logger.debug(f"Summary cache hit for {memory_id} in {operation_time:.2f}ms")
                    return cached_summary
                else:
                    self._v2_metrics['cache_misses'] += 1
            
            # Database retrieval with optimized query
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT summary FROM memories WHERE id = ? AND summary IS NOT NULL",
                    (memory_id,)
                )
                row = cursor.fetchone()
                
                if row and row[0]:
                    summary = row[0]
                    
                    # Cache for future <5ms retrieval
                    if self.cache_enabled:
                        self._cache_summary(memory_id, summary)
                    
                    # Update access tracking for performance optimization
                    conn.execute(
                        "UPDATE memories SET last_accessed = ? WHERE id = ?",
                        (datetime.now().isoformat(), memory_id)
                    )
                    
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self._metrics['successful_operations'] += 1
                    self._update_avg_response_time(operation_time)
                    
                    self.logger.debug(f"Summary retrieved for {memory_id} in {operation_time:.2f}ms")
                    return summary
            
            return None
    
    @storage_telemetry("retrieve_full_context_optimized")
    async def retrieve_full_context_optimized(self, memory_id: str) -> Optional[EnhancedMemory]:
        """Full context retrieval with <50ms target and lazy loading"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="retrieve_full_context_optimized",
            component="enhanced_amms_storage",
            metadata={"memory_id": memory_id}
        ):
            self._metrics['total_operations'] += 1
            self._v2_metrics['full_context_retrievals'] += 1
            
            with self._get_connection() as conn:
                # Optimized query to get all enhanced fields
                cursor = conn.execute("""
                    SELECT id, content, summary, full_context, tags, metadata,
                           importance_score, governance_status, context_size, tag_count,
                           summary_hash, context_hash, created_at, updated_at
                    FROM memories WHERE id = ?
                """, (memory_id,))
                row = cursor.fetchone()
                
                if row:
                    # Convert row to EnhancedMemory with all fields
                    memory = self._row_to_enhanced_memory(row)
                    
                    # Update access tracking for future optimization
                    conn.execute(
                        "UPDATE memories SET last_accessed = ? WHERE id = ?",
                        (datetime.now().isoformat(), memory_id)
                    )
                    
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self._metrics['successful_operations'] += 1
                    self._update_avg_response_time(operation_time)
                    
                    self.logger.debug(f"Full context retrieved for {memory_id} in {operation_time:.2f}ms")
                    return memory
            
            return None
    
    def _row_to_enhanced_memory(self, row) -> EnhancedMemory:
        """Convert database row to EnhancedMemory object"""
        # Safely parse JSON fields
        try:
            tags = json.loads(row['tags']) if row['tags'] else []
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Invalid tags for memory {row['id']}, using empty list")
            tags = []
        
        try:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Invalid metadata for memory {row['id']}, using empty dict")
            metadata = {}
        
        return EnhancedMemory(
            id=str(row['id']),
            content=row['content'] or "",
            summary=row['summary'],
            full_context=row['full_context'],
            tags=tags,
            metadata=metadata,
            importance_score=row['importance_score'] or 0.5,
            governance_status=row['governance_status'] or "approved",
            context_size=row['context_size'] or 0,
            tag_count=row['tag_count'] or 0,
            summary_hash=row['summary_hash'],
            context_hash=row['context_hash'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def _cache_summary(self, memory_id: str, summary: str):
        """Cache summary with LRU eviction"""
        if not self.cache_enabled:
            return
        
        with self._cache_lock:
            # Remove from access order if already exists
            if memory_id in self._cache_access_order:
                self._cache_access_order.remove(memory_id)
            
            # Add to front of access order
            self._cache_access_order.insert(0, memory_id)
            self.summary_cache[memory_id] = summary
            
            # LRU eviction if cache is full
            while len(self.summary_cache) > self.cache_size:
                oldest_key = self._cache_access_order.pop()
                self.summary_cache.pop(oldest_key, None)
    
    def _get_cached_summary(self, memory_id: str) -> Optional[str]:
        """Get cached summary with LRU update"""
        if not self.cache_enabled:
            return None
        
        with self._cache_lock:
            if memory_id in self.summary_cache:
                # Move to front of access order (mark as recently used)
                if memory_id in self._cache_access_order:
                    self._cache_access_order.remove(memory_id)
                self._cache_access_order.insert(0, memory_id)
                return self.summary_cache[memory_id]
        
        return None
    
    @handle_errors(catch=[sqlite3.Error, json.JSONDecodeError], reraise=True)
    async def search_enhanced_memories(
        self, 
        query: str, 
        limit: int = 10,
        context_level: str = "summary",
        tags_filter: Optional[List[str]] = None
    ) -> List[EnhancedMemory]:
        """Enhanced search with dual-layer support and tag filtering"""
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="search_enhanced_memories",
            component="enhanced_amms_storage",
            metadata={
                "query_length": len(query),
                "limit": limit,
                "context_level": context_level,
                "tags_filter": len(tags_filter) if tags_filter else 0
            }
        ):
            self._metrics['total_operations'] += 1
            
            memories = []
            with self._get_connection() as conn:
                # Build query based on context level and filters
                base_query = """
                    SELECT id, content, summary, full_context, tags, metadata,
                           importance_score, governance_status, context_size, tag_count,
                           summary_hash, context_hash, created_at, updated_at
                    FROM memories 
                    WHERE (content LIKE ? OR summary LIKE ? OR full_context LIKE ?)
                """
                
                params = [f"%{query}%", f"%{query}%", f"%{query}%"]
                
                # Add tag filtering if specified
                if tags_filter:
                    # Simple tag filtering (can be optimized with JSON operations in future)
                    for tag in tags_filter:
                        base_query += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')
                
                base_query += " ORDER BY importance_score DESC, created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(base_query, params)
                
                for row in cursor.fetchall():
                    memory = self._row_to_enhanced_memory(row)
                    memories.append(memory)
            
            operation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['successful_operations'] += 1
            self._update_avg_response_time(operation_time)
            
            self.logger.debug(f"Enhanced search returned {len(memories)} results in {operation_time:.2f}ms")
            return memories
    
    async def count_enhanced_memories(self) -> Dict[str, int]:
        """Enhanced memory count with governance statistics"""
        try:
            with self._get_connection() as conn:
                # Total count
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                total = cursor.fetchone()[0]
                
                # Count by governance status
                cursor = conn.execute("""
                    SELECT governance_status, COUNT(*) 
                    FROM memories 
                    GROUP BY governance_status
                """)
                governance_counts = dict(cursor.fetchall())
                
                # Count with summaries
                cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE summary IS NOT NULL")
                with_summaries = cursor.fetchone()[0]
                
                # Count with full context
                cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE full_context IS NOT NULL")
                with_full_context = cursor.fetchone()[0]
                
                return {
                    'total': total,
                    'with_summaries': with_summaries,
                    'with_full_context': with_full_context,
                    'governance_status': governance_counts
                }
        except Exception as e:
            self.logger.error(f"Failed to count enhanced memories: {e}")
            raise RuntimeError(f"Enhanced memory count failed: {e}") from e
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced storage statistics"""
        base_stats = self.get_stats()
        
        # Add v2.0 specific metrics
        v2_stats = {
            'storage_type': 'enhanced_amms',
            'v2_capabilities': [
                'dual_layer_storage', 
                'summary_caching', 
                'governance_tracking',
                'performance_optimization'
            ],
            'v2_metrics': self._v2_metrics.copy(),
            'cache_stats': {
                'enabled': self.cache_enabled,
                'size': len(self.summary_cache) if self.cache_enabled else 0,
                'max_size': self.cache_size,
                'hit_rate': (
                    self._v2_metrics['cache_hits'] / 
                    (self._v2_metrics['cache_hits'] + self._v2_metrics['cache_misses'])
                    if (self._v2_metrics['cache_hits'] + self._v2_metrics['cache_misses']) > 0 
                    else 0.0
                )
            } if self.cache_enabled else {'enabled': False}
        }
        
        # Merge base stats with v2 stats
        base_stats.update(v2_stats)
        return base_stats
    
    def clear_cache(self):
        """Clear summary cache for testing/maintenance"""
        if self.cache_enabled:
            with self._cache_lock:
                self.summary_cache.clear()
                self._cache_access_order.clear()
                self.logger.info("Summary cache cleared")
    
    async def close(self):
        """Enhanced cleanup including cache clearing"""
        # Clear cache
        if self.cache_enabled:
            self.clear_cache()
        
        # Call parent cleanup
        await super().close()
        
        self.logger.info("Enhanced AMMS Storage closed - all resources cleaned up")


def create_enhanced_amms_storage(db_path: str, config: Optional[Dict[str, Any]] = None) -> EnhancedAMMSStorage:
    """Factory function to create Enhanced AMMS storage"""
    return EnhancedAMMSStorage(db_path, config=config)