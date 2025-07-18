#!/usr/bin/env python3
"""
Active Memory Pool Manager - Core Implementation
Manages dynamic memory pool with intelligent ranking and lifecycle management
"""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from contextlib import contextmanager
from .memory import Memory, MemoryStore
from .active_schema import ActiveMemorySchema

@dataclass
class ActiveMemoryConfig:
    """Configuration for active memory pool management"""
    target_pool_size: int = 1000
    max_pool_size: int = 1500
    importance_threshold: float = 0.3
    stale_threshold_days: int = 30
    archive_threshold: float = 0.2
    prune_threshold: float = 0.1
    
    # Performance settings
    batch_size: int = 100
    max_query_time_ms: int = 100
    cache_size: int = 500
    
    # CXD integration weights
    cxd_weight: float = 0.40
    access_frequency_weight: float = 0.25
    recency_weight: float = 0.20
    confidence_weight: float = 0.10
    type_weight: float = 0.05

class ActiveMemoryPool:
    """
    Core active memory pool manager with intelligent ranking and lifecycle management
    
    Features:
    - Dynamic memory pool sizing based on importance scores
    - Intelligent ranking using CXD classification integration
    - Automated stale memory detection and archival
    - Access pattern tracking for relevance optimization
    - Memory consolidation for related memories
    """
    
    def __init__(self, db_path: str, config: Optional[ActiveMemoryConfig] = None):
        self.db_path = db_path
        self.config = config or ActiveMemoryConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Initialize schema and migrate if needed
        self.schema = ActiveMemorySchema(db_path)
        self.schema.create_enhanced_schema()
        
        # Cache for active memories
        self._active_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_timestamp = datetime.now()
        self._cache_valid = False
        
        # Performance metrics
        self._query_count = 0
        self._total_query_time = 0.0
        
        self.logger.info(f"ActiveMemoryPool initialized with config: {self.config}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def add_memory(self, memory: Memory, cxd_function: Optional[str] = None, 
                   cxd_confidence: float = 0.0) -> int:
        """
        Add memory to the pool with active management
        
        Args:
            memory: Memory object to add
            cxd_function: CXD classification (CONTROL, CONTEXT, DATA)
            cxd_confidence: Confidence in CXD classification
            
        Returns:
            Memory ID
        """
        with self._lock:
            start_time = datetime.now()
            
            # Calculate initial importance score
            importance_score = self._calculate_importance_score(
                memory, cxd_function, cxd_confidence
            )
            
            # Get memory type weight
            type_weight = self._get_memory_type_weight(memory.type)
            
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO memories_enhanced 
                    (content, type, confidence, created_at, access_count,
                     last_access_time, importance_score, archive_status,
                     cxd_function, cxd_confidence, access_frequency, 
                     recency_score, temporal_decay, memory_type_weight,
                     tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 0.0, 1.0, 1.0, ?, '[]', '{}')
                """, (
                    memory.content, memory.type, memory.confidence, 
                    memory.created_at, memory.access_count,
                    datetime.now().isoformat(), importance_score,
                    cxd_function, cxd_confidence, type_weight
                ))
                
                memory_id = cursor.lastrowid
                
                # Record importance calculation
                self._record_importance_calculation(conn, memory_id, importance_score)
                
                conn.commit()
            
            # Invalidate cache
            self._cache_valid = False
            
            # Check if pool maintenance is needed
            self._maybe_trigger_maintenance()
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            self.logger.debug(f"Added memory {memory_id} with importance {importance_score:.3f}")
            return memory_id
    
    def search_active_memories(self, query: str, limit: int = 10, 
                              boost_factors: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Search active memory pool with intelligent ranking
        
        Args:
            query: Search query
            limit: Maximum results to return
            boost_factors: Optional boost factors for ranking
            
        Returns:
            List of ranked memory dictionaries
        """
        with self._lock:
            start_time = datetime.now()
            
            # Refresh cache if needed
            if not self._cache_valid or self._is_cache_stale():
                self._refresh_active_cache()
            
            # Search and rank memories
            results = self._search_and_rank(query, limit, boost_factors)
            
            # Update access patterns for returned memories
            for result in results:
                self._update_access_pattern(result['id'], query)
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return results
    
    def get_active_pool_status(self) -> Dict[str, Any]:
        """Get current status of the active memory pool"""
        with self._get_connection() as conn:
            # Pool size statistics
            cursor = conn.execute("""
                SELECT archive_status, COUNT(*) as count, AVG(importance_score) as avg_importance
                FROM memories_enhanced 
                GROUP BY archive_status
            """)
            status_stats = {row['archive_status']: {
                'count': row['count'], 
                'avg_importance': row['avg_importance']
            } for row in cursor.fetchall()}
            
            # Memory type distribution in active pool
            cursor = conn.execute("""
                SELECT type, COUNT(*) as count, AVG(importance_score) as avg_importance
                FROM memories_enhanced 
                WHERE archive_status = 'active'
                GROUP BY type
                ORDER BY avg_importance DESC
            """)
            type_distribution = {row['type']: {
                'count': row['count'],
                'avg_importance': row['avg_importance']
            } for row in cursor.fetchall()}
            
            # CXD function distribution
            cursor = conn.execute("""
                SELECT cxd_function, COUNT(*) as count, AVG(importance_score) as avg_importance
                FROM memories_enhanced 
                WHERE archive_status = 'active' AND cxd_function IS NOT NULL
                GROUP BY cxd_function
            """)
            cxd_distribution = {row['cxd_function']: {
                'count': row['count'],
                'avg_importance': row['avg_importance']
            } for row in cursor.fetchall()}
            
            # Performance metrics
            avg_query_time = (self._total_query_time / self._query_count 
                            if self._query_count > 0 else 0.0)
            
            return {
                'status_stats': status_stats,
                'type_distribution': type_distribution,
                'cxd_distribution': cxd_distribution,
                'config': self.config.__dict__,
                'performance': {
                    'query_count': self._query_count,
                    'avg_query_time_ms': avg_query_time * 1000,
                    'cache_size': len(self._active_cache),
                    'cache_valid': self._cache_valid
                },
                'cache_timestamp': self._cache_timestamp.isoformat()
            }
    
    def maintain_pool(self, force: bool = False) -> Dict[str, int]:
        """
        Perform pool maintenance: ranking updates, archival, cleanup
        
        Args:
            force: Force maintenance even if not needed
            
        Returns:
            Dictionary with maintenance statistics
        """
        with self._lock:
            start_time = datetime.now()
            stats = {
                'updated_scores': 0,
                'archived_memories': 0,
                'pruned_memories': 0,
                'consolidated_groups': 0
            }
            
            if not force and not self._needs_maintenance():
                return stats
            
            with self._get_connection() as conn:
                # Update importance scores for all active memories
                stats['updated_scores'] = self._update_importance_scores(conn)
                
                # Archive low-importance memories
                stats['archived_memories'] = self._archive_stale_memories(conn)
                
                # Prune very old, low-importance memories
                stats['pruned_memories'] = self._prune_memories(conn)
                
                # Update memory consolidation
                stats['consolidated_groups'] = self._update_consolidation_groups(conn)
                
                conn.commit()
            
            # Refresh cache after maintenance
            self._cache_valid = False
            
            maintenance_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pool maintenance completed in {maintenance_time:.2f}s: {stats}")
            
            return stats
    
    def _calculate_importance_score(self, memory: Memory, cxd_function: Optional[str] = None,
                                   cxd_confidence: float = 0.0) -> float:
        """Calculate importance score using multi-factor algorithm"""
        
        # CXD classification component (40%)
        cxd_component = 0.0
        if cxd_function:
            cxd_weights = {'CONTROL': 0.8, 'CONTEXT': 1.0, 'DATA': 0.6}
            cxd_component = cxd_weights.get(cxd_function, 0.5) * cxd_confidence
        
        # Access frequency component (25%) - start with 0 for new memories
        access_frequency_component = 0.0
        
        # Recency component (20%) - new memories get full score
        recency_component = 1.0
        
        # Confidence component (10%)
        confidence_component = memory.confidence
        
        # Memory type component (5%)
        type_component = self._get_memory_type_weight(memory.type)
        
        # Calculate weighted importance score
        importance = (
            cxd_component * self.config.cxd_weight +
            access_frequency_component * self.config.access_frequency_weight +
            recency_component * self.config.recency_weight +
            confidence_component * self.config.confidence_weight +
            type_component * self.config.type_weight
        )
        
        return min(max(importance, 0.0), 1.0)
    
    def _get_memory_type_weight(self, memory_type: str) -> float:
        """Get importance weight for memory type"""
        weights = {
            'synthetic_wisdom': 1.0,
            'milestone': 0.9,
            'consciousness_evolution': 0.95,
            'reflection': 0.7,
            'interaction': 0.5,
            'project_info': 0.6
        }
        return weights.get(memory_type, 0.5)
    
    def _record_importance_calculation(self, conn, memory_id: int, importance_score: float):
        """Record importance calculation for audit trail"""
        conn.execute("""
            INSERT INTO importance_calculations 
            (memory_id, calculated_at, importance_score, calculation_version)
            VALUES (?, ?, ?, ?)
        """, (memory_id, datetime.now().isoformat(), importance_score, "1.0"))
    
    def _refresh_active_cache(self):
        """Refresh the active memory cache"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, content, type, confidence, importance_score, 
                       last_access_time, cxd_function, access_frequency
                FROM memories_enhanced 
                WHERE archive_status = 'active'
                ORDER BY importance_score DESC
                LIMIT ?
            """, (self.config.cache_size,))
            
            self._active_cache = {
                row['id']: dict(row) for row in cursor.fetchall()
            }
            
            self._cache_timestamp = datetime.now()
            self._cache_valid = True
    
    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh"""
        return (datetime.now() - self._cache_timestamp).total_seconds() > 300  # 5 minutes
    
    def _search_and_rank(self, query: str, limit: int, 
                        boost_factors: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Search and rank memories from active cache"""
        if not self._active_cache:
            return []
        
        query_lower = query.lower()
        scored_memories = []
        
        for memory_id, memory_data in self._active_cache.items():
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(memory_data, query_lower)
            
            # Apply boost factors if provided
            if boost_factors:
                for factor, weight in boost_factors.items():
                    if factor in memory_data.get('type', ''):
                        relevance_score *= weight
            
            # Combine with importance score
            final_score = (relevance_score * 0.6) + (memory_data['importance_score'] * 0.4)
            
            scored_memories.append((final_score, memory_data))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for score, memory in scored_memories[:limit]]
    
    def _calculate_relevance_score(self, memory_data: Dict[str, Any], query_lower: str) -> float:
        """Calculate relevance score for a memory against a query"""
        content_lower = memory_data['content'].lower()
        
        # Direct word matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        word_score = matches / len(query_words) if query_words else 0
        
        # Content length normalization
        length_penalty = min(len(content_lower) / 1000, 1.0)
        
        # Recent access boost
        try:
            last_access = datetime.fromisoformat(memory_data.get('last_access_time', ''))
            hours_since_access = (datetime.now() - last_access).total_seconds() / 3600
            recency_boost = max(0, 1.0 - (hours_since_access / 168))  # 1 week decay
        except:
            recency_boost = 0.0
        
        return (word_score * 0.7) + (recency_boost * 0.2) + ((1 - length_penalty) * 0.1)
    
    def _update_access_pattern(self, memory_id: int, query: str):
        """Update access pattern for a memory"""
        with self._get_connection() as conn:
            # Update last access time and access count
            conn.execute("""
                UPDATE memories_enhanced 
                SET last_access_time = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), memory_id))
            
            # Record access pattern
            conn.execute("""
                INSERT INTO memory_access_patterns 
                (memory_id, access_time, access_context)
                VALUES (?, ?, ?)
            """, (memory_id, datetime.now().isoformat(), query[:500]))
            
            conn.commit()
    
    def _maybe_trigger_maintenance(self):
        """Check if maintenance is needed and trigger if so"""
        if self._needs_maintenance():
            # Run maintenance in background thread to avoid blocking
            def run_maintenance():
                try:
                    self.maintain_pool()
                except Exception as e:
                    self.logger.error(f"Background maintenance failed: {e}")
            
            threading.Thread(target=run_maintenance, daemon=True).start()
    
    def _needs_maintenance(self) -> bool:
        """Check if pool maintenance is needed"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as active_count FROM memories_enhanced 
                WHERE archive_status = 'active'
            """)
            active_count = cursor.fetchone()['active_count']
            
            return active_count > self.config.max_pool_size
    
    def _update_importance_scores(self, conn) -> int:
        """Update importance scores for all active memories"""
        # This is a simplified version - full implementation would recalculate
        # based on current access patterns, CXD classification, etc.
        updated_count = 0
        # Implementation details would go here
        return updated_count
    
    def _archive_stale_memories(self, conn) -> int:
        """Archive stale memories based on thresholds"""
        stale_date = (datetime.now() - timedelta(days=self.config.stale_threshold_days)).isoformat()
        
        cursor = conn.execute("""
            UPDATE memories_enhanced 
            SET archive_status = 'archived'
            WHERE archive_status = 'active' 
            AND last_access_time < ?
            AND importance_score < ?
        """, (stale_date, self.config.archive_threshold))
        
        return cursor.rowcount
    
    def _prune_memories(self, conn) -> int:
        """Prune very old, low-importance memories"""
        prune_date = (datetime.now() - timedelta(days=self.config.stale_threshold_days * 3)).isoformat()
        
        cursor = conn.execute("""
            DELETE FROM memories_enhanced 
            WHERE archive_status = 'prune_candidate'
            AND last_access_time < ?
            AND importance_score < ?
        """, (prune_date, self.config.prune_threshold))
        
        return cursor.rowcount
    
    def _update_consolidation_groups(self, conn) -> int:
        """Update memory consolidation groups using MemoryConsolidator"""
        try:
            from .memory_consolidator import MemoryConsolidator
            
            # Create consolidator instance
            consolidator = MemoryConsolidator(self.db_path)
            
            # Run consolidation
            result = consolidator.consolidate_memories()
            
            return result.get('groups_created', 0) + result.get('groups_updated', 0)
            
        except ImportError:
            self.logger.warning("MemoryConsolidator not available, skipping consolidation")
            return 0
        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}")
            return 0
    
    def _update_performance_metrics(self, start_time: datetime):
        """Update performance tracking metrics"""
        query_time = (datetime.now() - start_time).total_seconds()
        self._query_count += 1
        self._total_query_time += query_time

# Utility functions
def create_active_memory_pool(db_path: str, config: Optional[ActiveMemoryConfig] = None) -> ActiveMemoryPool:
    """Create and initialize an active memory pool"""
    return ActiveMemoryPool(db_path, config)

if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        pool = create_active_memory_pool(db_path)
        status = pool.get_active_pool_status()
        
        print("Active Memory Pool Status:")
        print(f"  Active memories: {status['status_stats'].get('active', {}).get('count', 0)}")
        print(f"  Archived memories: {status['status_stats'].get('archived', {}).get('count', 0)}")
        print(f"  Cache size: {status['performance']['cache_size']}")
        print(f"  Avg query time: {status['performance']['avg_query_time_ms']:.2f}ms")
    else:
        print("Usage: python active_manager.py <database_path>")