"""
Database Initialization for Nervous System

Ensures proper database setup for enhanced triggers with memory storage.
"""

import asyncio
import sqlite3
import time
from typing import Optional
import logging

from ..memory.storage.amms_storage import AMMSStorage
from ..errors import get_error_logger, with_error_context

class DatabaseInitializer:
    """
    Database initialization for nervous system operation.
    
    Ensures all required tables and indexes are created properly,
    especially for in-memory databases used in testing.
    """
    
    def __init__(self, db_path: str = "memmimic.db"):
        self.db_path = db_path
        self.logger = get_error_logger("database_initializer")
        self._initialized = False
    
    async def initialize_database(self) -> bool:
        """
        Initialize database with all required schema.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        with with_error_context(
            operation="database_initialization",
            component="db_initializer",
            metadata={"db_path": self.db_path}
        ):
            try:
                start_time = time.perf_counter()
                
                # Create database connection and schema
                conn = sqlite3.connect(self.db_path)
                
                # Enable WAL mode for better performance (if not in-memory)
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL")
                
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys=ON")
                
                # Create memories table
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
                
                # Create indexes for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_created_at 
                    ON memories(created_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_importance 
                    ON memories(importance_score)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_content 
                    ON memories(content)
                """)
                
                # Create quality tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id INTEGER,
                        overall_score REAL,
                        dimensions TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories (id)
                    )
                """)
                
                # Create relationship tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_memory_id INTEGER,
                        target_memory_id INTEGER,
                        relationship_type TEXT,
                        strength REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                        FOREIGN KEY (target_memory_id) REFERENCES memories (id)
                    )
                """)
                
                # Commit changes
                conn.commit()
                conn.close()
                
                initialization_time = (time.perf_counter() - start_time) * 1000
                self._initialized = True
                
                self.logger.info(
                    f"Database initialized successfully in {initialization_time:.2f}ms",
                    extra={"initialization_time_ms": initialization_time, "db_path": self.db_path}
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                return False
    
    async def verify_database_integrity(self) -> bool:
        """
        Verify database integrity and schema completeness.
        
        Returns:
            bool: True if database is properly initialized
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if memories table exists and has correct schema
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='memories'
            """)
            
            if not cursor.fetchone():
                self.logger.error("memories table not found")
                conn.close()
                return False
            
            # Check table structure
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_columns = {
                'id': 'INTEGER',
                'content': 'TEXT',
                'metadata': 'TEXT',
                'importance_score': 'REAL',
                'created_at': 'TEXT',
                'updated_at': 'TEXT'
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in columns:
                    self.logger.error(f"Required column '{col_name}' missing from memories table")
                    conn.close()
                    return False
            
            conn.close()
            
            self.logger.info("Database integrity verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database integrity verification failed: {e}")
            return False
    
    async def create_test_data(self) -> bool:
        """
        Create test data for validation.
        
        Returns:
            bool: True if test data created successfully
        """
        try:
            # Initialize AMMS storage with proper database
            amms_storage = AMMSStorage(self.db_path)
            
            # Create test memories
            test_memories = [
                {
                    "content": "Machine learning algorithms for pattern recognition in neural networks",
                    "metadata": {"type": "technical", "importance": 0.8},
                    "importance_score": 0.8
                },
                {
                    "content": "User interface design principles for better user experience",
                    "metadata": {"type": "technical", "importance": 0.7},
                    "importance_score": 0.7
                },
                {
                    "content": "Database optimization techniques and indexing strategies",
                    "metadata": {"type": "technical", "importance": 0.9},
                    "importance_score": 0.9
                },
                {
                    "content": "Project completed successfully with all milestones achieved",
                    "metadata": {"type": "milestone", "importance": 0.95},
                    "importance_score": 0.95
                },
                {
                    "content": "Learned the importance of user feedback in product development cycles",
                    "metadata": {"type": "reflection", "importance": 0.6},
                    "importance_score": 0.6
                }
            ]
            
            # Store test memories
            from ..memory.storage.amms_storage import Memory
            from datetime import datetime
            
            conn = sqlite3.connect(self.db_path)
            
            for mem_data in test_memories:
                cursor = conn.execute("""
                    INSERT INTO memories (content, metadata, importance_score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    mem_data["content"],
                    str(mem_data["metadata"]),
                    mem_data["importance_score"],
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created {len(test_memories)} test memories for validation")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test data: {e}")
            return False

# Global database initializer instance
_global_db_initializer = None

async def get_initialized_database(db_path: str = "memmimic.db") -> DatabaseInitializer:
    """
    Get properly initialized database instance.
    
    Returns:
        DatabaseInitializer: Ready-to-use database initializer
    """
    global _global_db_initializer
    
    if _global_db_initializer is None or _global_db_initializer.db_path != db_path:
        _global_db_initializer = DatabaseInitializer(db_path)
        await _global_db_initializer.initialize_database()
    
    return _global_db_initializer

async def ensure_database_ready(db_path: str = "memmimic.db") -> bool:
    """
    Ensure database is properly initialized and ready for use.
    
    Returns:
        bool: True if database is ready
    """
    db_init = await get_initialized_database(db_path)
    return await db_init.verify_database_integrity()