#!/usr/bin/env python3
"""
Memory Consolidator - Advanced Memory Relationship Management
Intelligently merges related memories and prevents duplicates across the memory pool
"""

import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict
import json
import re

@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation"""
    similarity_threshold: float = 0.85
    content_similarity_weight: float = 0.6
    semantic_similarity_weight: float = 0.25
    temporal_proximity_weight: float = 0.15
    max_consolidation_group_size: int = 5
    consolidation_cooldown_hours: int = 24
    min_memory_age_hours: int = 1
    duplicate_threshold: float = 0.95
    
    # Content analysis settings
    min_content_length: int = 20
    max_content_length_ratio: float = 3.0
    keyword_overlap_threshold: float = 0.4

class MemoryConsolidator:
    """
    Advanced memory consolidation system for detecting and merging related memories
    
    Features:
    - Content similarity analysis using multiple algorithms
    - Semantic relationship detection
    - Temporal proximity consideration
    - Duplicate detection and prevention
    - Importance-aware consolidation
    - Audit trail for all consolidation operations
    """
    
    def __init__(self, db_path: str, config: Optional[ConsolidationConfig] = None):
        self.db_path = db_path
        self.config = config or ConsolidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize consolidation schema
        self._create_consolidation_schema()
        
        self.logger.info(f"MemoryConsolidator initialized with config: {self.config}")
    
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
    
    def _create_consolidation_schema(self):
        """Create schema for consolidation tracking"""
        with self._get_connection() as conn:
            # Consolidation groups table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated_at TEXT NOT NULL,
                    primary_memory_id INTEGER,
                    group_size INTEGER DEFAULT 1,
                    consolidation_reason TEXT,
                    similarity_score REAL,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (primary_memory_id) REFERENCES memories_enhanced(id)
                )
            """)
            
            # Memory group membership table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_group_membership (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    group_id INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    is_primary BOOLEAN DEFAULT FALSE,
                    consolidation_contribution REAL DEFAULT 0.0,
                    FOREIGN KEY (memory_id) REFERENCES memories_enhanced(id) ON DELETE CASCADE,
                    FOREIGN KEY (group_id) REFERENCES consolidation_groups(id) ON DELETE CASCADE,
                    UNIQUE(memory_id, group_id)
                )
            """)
            
            # Consolidation audit trail
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    operation_time TEXT NOT NULL,
                    group_id INTEGER,
                    memory_ids TEXT,
                    similarity_scores TEXT,
                    consolidation_details TEXT,
                    FOREIGN KEY (group_id) REFERENCES consolidation_groups(id)
                )
            """)
            
            # Add missing columns if they don't exist (for existing installations)
            missing_columns = [
                ("status", "TEXT DEFAULT 'active'"),
                ("group_size", "INTEGER DEFAULT 1"),
                ("consolidation_reason", "TEXT"),
                ("similarity_score", "REAL"),
                ("last_updated_at", "TEXT"),
                ("group_hash", "TEXT"),
                ("primary_memory_id", "INTEGER")
            ]
            
            for col_name, col_def in missing_columns:
                try:
                    conn.execute(f"ALTER TABLE consolidation_groups ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_group_membership_memory ON memory_group_membership(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_group_membership_group ON memory_group_membership(group_id)")
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_consolidation_groups_status ON consolidation_groups(status)")
            except sqlite3.OperationalError:
                pass  # Column doesn't exist yet
            
            conn.commit()
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, int]:
        """
        Perform comprehensive memory consolidation
        
        Args:
            force: Force consolidation even if cooldown hasn't expired
            
        Returns:
            Dictionary with consolidation statistics
        """
        start_time = datetime.now()
        stats = {
            'analyzed_memories': 0,
            'duplicates_found': 0,
            'groups_created': 0,
            'groups_updated': 0,
            'memories_consolidated': 0,
            'processing_time_ms': 0
        }
        
        try:
            with self._get_connection() as conn:
                # Check cooldown period
                if not force and not self._should_run_consolidation(conn):
                    self.logger.info("Consolidation skipped due to cooldown period")
                    return stats
                
                # Get eligible memories for consolidation
                eligible_memories = self._get_eligible_memories(conn)
                stats['analyzed_memories'] = len(eligible_memories)
                
                if not eligible_memories:
                    self.logger.info("No eligible memories found for consolidation")
                    return stats
                
                # Find duplicate memories
                duplicate_groups = self._find_duplicate_memories(eligible_memories)
                stats['duplicates_found'] = sum(len(group) - 1 for group in duplicate_groups)
                
                # Process duplicates first
                for duplicate_group in duplicate_groups:
                    self._consolidate_duplicate_group(conn, duplicate_group, stats)
                
                # Find similar memory groups
                similarity_groups = self._find_similar_memory_groups(eligible_memories)
                
                # Process similarity groups
                for group in similarity_groups:
                    self._consolidate_similarity_group(conn, group, stats)
                
                # Update consolidation timestamp
                self._update_last_consolidation_time(conn)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}")
            raise
        
        finally:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            stats['processing_time_ms'] = int(processing_time)
            
            self.logger.info(f"Memory consolidation completed: {stats}")
        
        return stats
    
    def _should_run_consolidation(self, conn) -> bool:
        """Check if consolidation should run based on cooldown period"""
        cursor = conn.execute("""
            SELECT MAX(operation_time) as last_consolidation
            FROM consolidation_audit 
            WHERE operation_type = 'full_consolidation'
        """)
        result = cursor.fetchone()
        
        if not result or not result['last_consolidation']:
            return True
        
        last_consolidation = datetime.fromisoformat(result['last_consolidation'])
        hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600
        
        return hours_since >= self.config.consolidation_cooldown_hours
    
    def _get_eligible_memories(self, conn) -> List[Dict[str, Any]]:
        """Get memories eligible for consolidation"""
        min_age = (datetime.now() - timedelta(hours=self.config.min_memory_age_hours)).isoformat()
        
        cursor = conn.execute("""
            SELECT id, content, type, confidence, importance_score, 
                   created_at, last_access_time, cxd_function, access_count
            FROM memories_enhanced 
            WHERE archive_status = 'active'
            AND created_at < ?
            AND LENGTH(content) >= ?
            AND id NOT IN (
                SELECT memory_id FROM memory_group_membership 
                WHERE group_id IN (
                    SELECT id FROM consolidation_groups 
                    WHERE status = 'active'
                    AND last_updated_at > ?
                )
            )
            ORDER BY importance_score DESC, created_at DESC
        """, (min_age, self.config.min_content_length, 
              (datetime.now() - timedelta(hours=self.config.consolidation_cooldown_hours)).isoformat()))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _find_duplicate_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of duplicate memories"""
        duplicate_groups = []
        processed_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1['id'] in processed_ids:
                continue
                
            duplicates = [memory1]
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2['id'] in processed_ids:
                    continue
                
                similarity = self._calculate_content_similarity(memory1['content'], memory2['content'])
                
                if similarity >= self.config.duplicate_threshold:
                    duplicates.append(memory2)
                    processed_ids.add(memory2['id'])
            
            if len(duplicates) > 1:
                duplicate_groups.append(duplicates)
                for mem in duplicates:
                    processed_ids.add(mem['id'])
        
        return duplicate_groups
    
    def _find_similar_memory_groups(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of similar memories using multiple similarity metrics"""
        similarity_groups = []
        processed_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1['id'] in processed_ids:
                continue
            
            similar_group = [memory1]
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2['id'] in processed_ids:
                    continue
                
                # Calculate comprehensive similarity score
                similarity = self._calculate_comprehensive_similarity(memory1, memory2)
                
                if similarity >= self.config.similarity_threshold:
                    similar_group.append(memory2)
                    
                    # Check group size limit
                    if len(similar_group) >= self.config.max_consolidation_group_size:
                        break
            
            if len(similar_group) > 1:
                similarity_groups.append(similar_group)
                for mem in similar_group:
                    processed_ids.add(mem['id'])
        
        return similarity_groups
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using multiple metrics"""
        if not content1 or not content2:
            return 0.0
        
        # Normalize content
        c1_norm = self._normalize_content(content1)
        c2_norm = self._normalize_content(content2)
        
        # Exact match check
        if c1_norm == c2_norm:
            return 1.0
        
        # Length ratio check
        len_ratio = min(len(c1_norm), len(c2_norm)) / max(len(c1_norm), len(c2_norm))
        if len_ratio < (1.0 / self.config.max_content_length_ratio):
            return 0.0
        
        # Jaccard similarity on words
        words1 = set(c1_norm.split())
        words2 = set(c2_norm.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Character-level similarity (for similar lengths)
        if len_ratio > 0.7:
            char_similarity = self._calculate_character_similarity(c1_norm, c2_norm)
            return (jaccard_similarity * 0.7) + (char_similarity * 0.3)
        
        return jaccard_similarity
    
    def _calculate_comprehensive_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate comprehensive similarity score using multiple factors"""
        
        # Content similarity
        content_sim = self._calculate_content_similarity(memory1['content'], memory2['content'])
        
        # Semantic similarity (based on type and CXD function)
        semantic_sim = self._calculate_semantic_similarity(memory1, memory2)
        
        # Temporal proximity
        temporal_sim = self._calculate_temporal_similarity(memory1, memory2)
        
        # Weighted combination
        comprehensive_score = (
            content_sim * self.config.content_similarity_weight +
            semantic_sim * self.config.semantic_similarity_weight +
            temporal_sim * self.config.temporal_proximity_weight
        )
        
        return min(comprehensive_score, 1.0)
    
    def _calculate_semantic_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate semantic similarity based on type and CXD function"""
        score = 0.0
        
        # Type similarity
        if memory1['type'] == memory2['type']:
            score += 0.5
        
        # CXD function similarity
        if (memory1.get('cxd_function') and memory2.get('cxd_function') and
            memory1['cxd_function'] == memory2['cxd_function']):
            score += 0.3
        
        # Confidence similarity
        conf_diff = abs(memory1['confidence'] - memory2['confidence'])
        conf_similarity = max(0, 1.0 - conf_diff)
        score += conf_similarity * 0.2
        
        return min(score, 1.0)
    
    def _calculate_temporal_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate temporal similarity based on creation and access times"""
        try:
            created1 = datetime.fromisoformat(memory1['created_at'])
            created2 = datetime.fromisoformat(memory2['created_at'])
            
            time_diff_hours = abs((created1 - created2).total_seconds()) / 3600
            
            # Memories created within 24 hours get high temporal similarity
            if time_diff_hours <= 24:
                return 1.0 - (time_diff_hours / 24)
            
            # Gradual decay over 7 days
            if time_diff_hours <= 168:  # 7 days
                return 0.5 * (1.0 - (time_diff_hours - 24) / 144)
            
            return 0.0
            
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity using Levenshtein-like approach"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap ratio
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def _consolidate_duplicate_group(self, conn, duplicate_group: List[Dict[str, Any]], stats: Dict[str, int]):
        """Consolidate a group of duplicate memories"""
        if len(duplicate_group) < 2:
            return
        
        # Choose primary memory (highest importance score, then most recent)
        primary_memory = max(duplicate_group, 
                           key=lambda m: (m['importance_score'], m['created_at']))
        
        # Create consolidation group
        group_hash = self._generate_group_hash(duplicate_group)
        group_id = self._create_consolidation_group(
            conn, group_hash, primary_memory['id'], len(duplicate_group),
            "duplicate_detection", 1.0
        )
        
        # Add all memories to the group
        for memory in duplicate_group:
            self._add_memory_to_group(
                conn, memory['id'], group_id, 
                memory['id'] == primary_memory['id']
            )
        
        # Mark non-primary memories as consolidated
        non_primary_ids = [m['id'] for m in duplicate_group if m['id'] != primary_memory['id']]
        if non_primary_ids:
            placeholders = ','.join(['?' for _ in non_primary_ids])
            conn.execute(f"""
                UPDATE memories_enhanced 
                SET archive_status = 'consolidated'
                WHERE id IN ({placeholders})
            """, non_primary_ids)
        
        # Record audit trail
        self._record_consolidation_audit(
            conn, "duplicate_consolidation", group_id,
            [m['id'] for m in duplicate_group], [1.0] * len(duplicate_group)
        )
        
        stats['groups_created'] += 1
        stats['memories_consolidated'] += len(duplicate_group) - 1
    
    def _consolidate_similarity_group(self, conn, similarity_group: List[Dict[str, Any]], stats: Dict[str, int]):
        """Consolidate a group of similar memories"""
        if len(similarity_group) < 2:
            return
        
        # Calculate average similarity for the group
        similarities = []
        for i, mem1 in enumerate(similarity_group):
            for mem2 in similarity_group[i+1:]:
                sim = self._calculate_comprehensive_similarity(mem1, mem2)
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Only consolidate if average similarity is above threshold
        if avg_similarity < self.config.similarity_threshold:
            return
        
        # Choose primary memory (highest importance * access_count)
        primary_memory = max(similarity_group, 
                           key=lambda m: m['importance_score'] * (m['access_count'] + 1))
        
        # Create consolidation group
        group_hash = self._generate_group_hash(similarity_group)
        group_id = self._create_consolidation_group(
            conn, group_hash, primary_memory['id'], len(similarity_group),
            "similarity_consolidation", avg_similarity
        )
        
        # Add all memories to the group
        for memory in similarity_group:
            contribution_score = self._calculate_contribution_score(memory, similarity_group)
            self._add_memory_to_group(
                conn, memory['id'], group_id, 
                memory['id'] == primary_memory['id'],
                contribution_score
            )
        
        # Record audit trail
        self._record_consolidation_audit(
            conn, "similarity_consolidation", group_id,
            [m['id'] for m in similarity_group], similarities
        )
        
        stats['groups_created'] += 1
        stats['memories_consolidated'] += len(similarity_group)
    
    def _generate_group_hash(self, memories: List[Dict[str, Any]]) -> str:
        """Generate a unique hash for a memory group"""
        memory_ids = sorted([str(m['id']) for m in memories])
        content = '|'.join(memory_ids)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_consolidation_group(self, conn, group_hash: str, primary_memory_id: int,
                                  group_size: int, reason: str, similarity_score: float) -> int:
        """Create a new consolidation group"""
        cursor = conn.execute("""
            INSERT INTO consolidation_groups 
            (group_hash, created_at, last_updated_at, primary_memory_id, 
             group_size, consolidation_reason, similarity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (group_hash, datetime.now().isoformat(), datetime.now().isoformat(),
              primary_memory_id, group_size, reason, similarity_score))
        
        return cursor.lastrowid
    
    def _add_memory_to_group(self, conn, memory_id: int, group_id: int, 
                           is_primary: bool, contribution: float = 0.0):
        """Add a memory to a consolidation group"""
        conn.execute("""
            INSERT OR REPLACE INTO memory_group_membership 
            (memory_id, group_id, added_at, is_primary, consolidation_contribution)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_id, group_id, datetime.now().isoformat(), is_primary, contribution))
    
    def _calculate_contribution_score(self, memory: Dict[str, Any], 
                                    group: List[Dict[str, Any]]) -> float:
        """Calculate how much a memory contributes to its consolidation group"""
        base_score = memory['importance_score']
        access_boost = min(memory['access_count'] / 10.0, 0.2)
        confidence_factor = memory['confidence']
        
        return min(base_score + access_boost, 1.0) * confidence_factor
    
    def _record_consolidation_audit(self, conn, operation_type: str, group_id: int,
                                  memory_ids: List[int], similarity_scores: List[float]):
        """Record consolidation operation in audit trail"""
        conn.execute("""
            INSERT INTO consolidation_audit 
            (operation_type, operation_time, group_id, memory_ids, 
             similarity_scores, consolidation_details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (operation_type, datetime.now().isoformat(), group_id,
              json.dumps(memory_ids), json.dumps(similarity_scores),
              json.dumps({'config': self.config.__dict__})))
    
    def _update_last_consolidation_time(self, conn):
        """Update the last consolidation time"""
        self._record_consolidation_audit(
            conn, "full_consolidation", None, [], []
        )
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status and statistics"""
        with self._get_connection() as conn:
            # Group statistics
            cursor = conn.execute("""
                SELECT COUNT(*) as total_groups, 
                       AVG(group_size) as avg_group_size,
                       MAX(group_size) as max_group_size,
                       AVG(similarity_score) as avg_similarity
                FROM consolidation_groups 
                WHERE status = 'active'
            """)
            group_stats = dict(cursor.fetchone())
            
            # Memory consolidation statistics
            cursor = conn.execute("""
                SELECT COUNT(*) as consolidated_memories
                FROM memories_enhanced 
                WHERE archive_status = 'consolidated'
            """)
            consolidated_count = cursor.fetchone()['consolidated_memories']
            
            # Recent consolidation activity
            cursor = conn.execute("""
                SELECT operation_type, COUNT(*) as count,
                       MAX(operation_time) as last_operation
                FROM consolidation_audit 
                WHERE operation_time > ?
                GROUP BY operation_type
            """, ((datetime.now() - timedelta(days=7)).isoformat(),))
            
            recent_activity = {row['operation_type']: {
                'count': row['count'],
                'last_operation': row['last_operation']
            } for row in cursor.fetchall()}
            
            return {
                'group_statistics': group_stats,
                'consolidated_memories': consolidated_count,
                'recent_activity': recent_activity,
                'configuration': self.config.__dict__
            }

# Utility functions
def create_memory_consolidator(db_path: str, config: Optional[ConsolidationConfig] = None) -> MemoryConsolidator:
    """Create and initialize a memory consolidator"""
    return MemoryConsolidator(db_path, config)

if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        consolidator = create_memory_consolidator(db_path)
        status = consolidator.get_consolidation_status()
        
        print("Memory Consolidation Status:")
        print(f"  Active groups: {status['group_statistics']['total_groups']}")
        print(f"  Consolidated memories: {status['consolidated_memories']}")
        print(f"  Average group size: {status['group_statistics']['avg_group_size']:.1f}")
        print(f"  Average similarity: {status['group_statistics']['avg_similarity']:.3f}")
        
        # Run consolidation
        print("\nRunning consolidation...")
        result = consolidator.consolidate_memories()
        print(f"  Analyzed: {result['analyzed_memories']} memories")
        print(f"  Created: {result['groups_created']} groups")
        print(f"  Consolidated: {result['memories_consolidated']} memories")
        print(f"  Processing time: {result['processing_time_ms']}ms")
    else:
        print("Usage: python memory_consolidator.py <database_path>")