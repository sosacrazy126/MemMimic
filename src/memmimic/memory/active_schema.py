#!/usr/bin/env python3
"""
Active Memory Management System - Database Schema Enhancements
Extends the existing memory.py schema with active memory management capabilities
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict


class ActiveMemorySchema:
    """Database schema enhancements for active memory management"""

    # Schema version for migration tracking
    SCHEMA_VERSION = "1.0.0"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

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

    def create_enhanced_schema(self) -> None:
        """Create enhanced schema for active memory management"""
        with self._get_connection() as conn:
            # Create schema version table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT
                )
            """
            )

            # Enhanced memories table with active memory fields
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories_enhanced (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    
                    -- Active Memory Management Fields
                    last_access_time TEXT,
                    importance_score REAL DEFAULT 0.0,
                    archive_status TEXT DEFAULT 'active' CHECK (archive_status IN ('active', 'archived', 'prune_candidate')),
                    consolidation_group_id INTEGER,
                    parent_memory_id INTEGER,
                    
                    -- CXD Classification Integration
                    cxd_function TEXT CHECK (cxd_function IN ('CONTROL', 'CONTEXT', 'DATA')),
                    cxd_confidence REAL DEFAULT 0.0,
                    
                    -- Memory Lifecycle Tracking
                    access_frequency REAL DEFAULT 0.0,
                    recency_score REAL DEFAULT 1.0,
                    temporal_decay REAL DEFAULT 1.0,
                    memory_type_weight REAL DEFAULT 0.0,
                    
                    -- Metadata
                    tags TEXT,  -- JSON array of tags
                    metadata TEXT,  -- JSON metadata
                    
                    FOREIGN KEY (parent_memory_id) REFERENCES memories_enhanced(id),
                    FOREIGN KEY (consolidation_group_id) REFERENCES consolidation_groups(id)
                )
            """
            )

            # Consolidation groups for related memories
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consolidation_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    summary_content TEXT,
                    group_importance REAL DEFAULT 0.0,
                    member_count INTEGER DEFAULT 0
                )
            """
            )

            # Memory access patterns for intelligent ranking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    access_time TEXT NOT NULL,
                    access_context TEXT,
                    query_similarity REAL DEFAULT 0.0,
                    
                    FOREIGN KEY (memory_id) REFERENCES memories_enhanced(id)
                )
            """
            )

            # Active memory pool configuration
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_memory_config (
                    id INTEGER PRIMARY KEY,
                    target_pool_size INTEGER DEFAULT 1000,
                    max_pool_size INTEGER DEFAULT 1500,
                    importance_threshold REAL DEFAULT 0.3,
                    stale_threshold_days INTEGER DEFAULT 30,
                    archive_threshold REAL DEFAULT 0.2,
                    prune_threshold REAL DEFAULT 0.1,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Memory importance calculation audit trail
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS importance_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    calculated_at TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    
                    -- Score breakdown for transparency
                    cxd_component REAL DEFAULT 0.0,
                    access_frequency_component REAL DEFAULT 0.0,
                    recency_component REAL DEFAULT 0.0,
                    confidence_component REAL DEFAULT 0.0,
                    type_weight_component REAL DEFAULT 0.0,
                    
                    calculation_version TEXT,
                    
                    FOREIGN KEY (memory_id) REFERENCES memories_enhanced(id)
                )
            """
            )

            self._create_indices(conn)
            self._insert_default_config(conn)
            conn.commit()

            # Record schema version
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_version (version, applied_at, description)
                VALUES (?, ?, ?)
            """,
                (
                    self.SCHEMA_VERSION,
                    datetime.now().isoformat(),
                    "Active Memory Management System schema",
                ),
            )
            conn.commit()

    def _create_indices(self, conn) -> None:
        """Create performance indices for active memory queries"""
        indices = [
            # Core memory indices
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_content ON memories_enhanced(content)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_type ON memories_enhanced(type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_confidence ON memories_enhanced(confidence)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_created_at ON memories_enhanced(created_at)",
            # Active memory management indices
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_importance ON memories_enhanced(importance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_archive_status ON memories_enhanced(archive_status)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_last_access ON memories_enhanced(last_access_time)",
            "CREATE INDEX IF NOT EXISTS idx_memories_enhanced_cxd_function ON memories_enhanced(cxd_function)",
            # Composite indices for common queries
            "CREATE INDEX IF NOT EXISTS idx_memories_active_important ON memories_enhanced(archive_status, importance_score DESC) WHERE archive_status = 'active'",
            "CREATE INDEX IF NOT EXISTS idx_memories_stale_candidates ON memories_enhanced(last_access_time, importance_score) WHERE archive_status = 'active'",
            # Access patterns indices
            "CREATE INDEX IF NOT EXISTS idx_access_patterns_memory ON memory_access_patterns(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_access_patterns_time ON memory_access_patterns(access_time)",
            # Consolidation indices
            "CREATE INDEX IF NOT EXISTS idx_consolidation_group ON memories_enhanced(consolidation_group_id)",
            "CREATE INDEX IF NOT EXISTS idx_parent_memory ON memories_enhanced(parent_memory_id)",
            # Importance calculation indices
            "CREATE INDEX IF NOT EXISTS idx_importance_calc_memory ON importance_calculations(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_importance_calc_time ON importance_calculations(calculated_at)",
        ]

        for index_sql in indices:
            conn.execute(index_sql)

    def _insert_default_config(self, conn) -> None:
        """Insert default active memory configuration"""
        conn.execute(
            """
            INSERT OR IGNORE INTO active_memory_config 
            (id, target_pool_size, max_pool_size, importance_threshold, 
             stale_threshold_days, archive_threshold, prune_threshold, updated_at)
            VALUES (1, 1000, 1500, 0.3, 30, 0.2, 0.1, ?)
        """,
            (datetime.now().isoformat(),),
        )

    def migrate_existing_memories(self) -> int:
        """Migrate memories from existing table to enhanced schema"""
        migrated_count = 0

        with self._get_connection() as conn:
            # Check if old memories table exists
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='memories'
            """
            )

            if cursor.fetchone():
                # Migrate existing memories
                cursor = conn.execute("SELECT * FROM memories")
                memories = cursor.fetchall()

                for memory in memories:
                    # Calculate initial importance score based on existing data
                    initial_importance = self._calculate_initial_importance(
                        memory["confidence"], memory["type"], memory["access_count"]
                    )

                    # Insert into enhanced table
                    conn.execute(
                        """
                        INSERT INTO memories_enhanced 
                        (content, type, confidence, created_at, access_count, 
                         last_access_time, importance_score, archive_status,
                         access_frequency, recency_score, temporal_decay, memory_type_weight)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 0.0, 1.0, 1.0, ?)
                    """,
                        (
                            memory["content"],
                            memory["type"],
                            memory["confidence"],
                            memory["created_at"],
                            memory["access_count"],
                            memory[
                                "created_at"
                            ],  # Use created_at as initial last_access
                            initial_importance,
                            self._get_memory_type_weight(memory["type"]),
                        ),
                    )
                    migrated_count += 1

                conn.commit()
                self.logger.info(
                    f"Migrated {migrated_count} memories to enhanced schema"
                )

        return migrated_count

    def _calculate_initial_importance(
        self, confidence: float, memory_type: str, access_count: int
    ) -> float:
        """Calculate initial importance score for migrated memories"""
        # Base score from confidence
        base_score = confidence * 0.4

        # Type weight bonus
        type_weights = {
            "synthetic_wisdom": 0.3,
            "milestone": 0.25,
            "reflection": 0.2,
            "interaction": 0.1,
            "project_info": 0.15,
        }
        type_bonus = type_weights.get(memory_type, 0.1)

        # Access count bonus (normalized)
        access_bonus = min(access_count / 10.0, 0.2)

        return min(base_score + type_bonus + access_bonus, 1.0)

    def _get_memory_type_weight(self, memory_type: str) -> float:
        """Get weight for memory type in importance calculation"""
        weights = {
            "synthetic_wisdom": 1.0,
            "milestone": 0.9,
            "reflection": 0.7,
            "interaction": 0.5,
            "project_info": 0.6,
            "consciousness_evolution": 0.95,
        }
        return weights.get(memory_type, 0.5)

    def check_enhanced_schema_exists(self) -> bool:
        """Check if enhanced schema exists in the database"""
        try:
            with self._get_connection() as conn:
                # Check if memories_enhanced table exists
                cursor = conn.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='memories_enhanced'
                """
                )
                return cursor.fetchone() is not None
        except Exception:
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the current schema"""
        with self._get_connection() as conn:
            # Get schema version
            cursor = conn.execute(
                "SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            )
            version_info = cursor.fetchone()

            # Get table info
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'memories%' OR name LIKE '%memory%'
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            # Get active memory config
            cursor = conn.execute("SELECT * FROM active_memory_config WHERE id = 1")
            config = cursor.fetchone()

            return {
                "schema_version": dict(version_info) if version_info else None,
                "tables": tables,
                "config": dict(config) if config else None,
            }


# Utility functions for schema management
def create_enhanced_memory_schema(db_path: str) -> ActiveMemorySchema:
    """Create and initialize enhanced memory schema"""
    schema = ActiveMemorySchema(db_path)
    schema.create_enhanced_schema()
    return schema


def migrate_memory_database(db_path: str, backup: bool = True) -> int:
    """Migrate existing memory database to enhanced schema"""
    import shutil

    if backup:
        backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to: {backup_path}")

    schema = ActiveMemorySchema(db_path)
    schema.create_enhanced_schema()
    migrated_count = schema.migrate_existing_memories()

    return migrated_count


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        print(f"Creating enhanced schema for: {db_path}")

        schema = create_enhanced_memory_schema(db_path)
        info = schema.get_schema_info()

        print("Schema Info:")
        print(
            f"  Version: {info['schema_version']['version'] if info['schema_version'] else 'None'}"
        )
        print(f"  Tables: {', '.join(info['tables'])}")
        print(f"  Config: {info['config']}")
    else:
        print("Usage: python active_schema.py <database_path>")

