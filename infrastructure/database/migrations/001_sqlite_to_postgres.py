#!/usr/bin/env python3
"""
MemMimic SQLite to PostgreSQL Migration Tool
Migrates data from SQLite AMMS storage to PostgreSQL sharded architecture
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import asyncpg
from asyncpg.pool import Pool
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles migration from SQLite to PostgreSQL with sharding"""

    def __init__(self, 
                 sqlite_path: str,
                 postgres_config: Dict[str, Any],
                 shard_configs: List[Dict[str, Any]]):
        self.sqlite_path = sqlite_path
        self.postgres_config = postgres_config
        self.shard_configs = shard_configs
        self.pg_pool: Optional[Pool] = None
        self.shard_pools: Dict[int, Pool] = {}

    async def initialize_connections(self):
        """Initialize PostgreSQL connections"""
        logger.info("Initializing PostgreSQL connections...")
        
        # Main coordinator connection
        self.pg_pool = await asyncpg.create_pool(**self.postgres_config)
        
        # Shard connections
        for shard_config in self.shard_configs:
            shard_id = shard_config["shard_id"]
            pool = await asyncpg.create_pool(**shard_config["connection"])
            self.shard_pools[shard_id] = pool
            
        logger.info(f"Connected to {len(self.shard_pools)} shards")

    def get_shard_key(self, memory_data: Dict) -> str:
        """Generate shard key based on memory content"""
        # Use memory type + first 100 chars of content for sharding
        content_hash = hashlib.md5(
            (memory_data.get("content", "") + memory_data.get("memory_type", ""))[:100].encode()
        ).hexdigest()[:10]
        return f"{memory_data.get('memory_type', 'unknown')}_{content_hash}"

    def get_shard_id(self, shard_key: str) -> int:
        """Determine which shard to use for a given key"""
        # Simple hash-based sharding
        hash_value = int(hashlib.md5(shard_key.encode()).hexdigest()[:8], 16)
        return (hash_value % len(self.shard_pools)) + 1

    async def migrate_memories(self):
        """Migrate memories from SQLite to PostgreSQL shards"""
        logger.info("Starting memory migration...")
        
        # Connect to SQLite
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()

        try:
            # Get all memories from SQLite
            cursor.execute("""
                SELECT * FROM memories 
                ORDER BY created_at
            """)
            
            memories = cursor.fetchall()
            logger.info(f"Found {len(memories)} memories to migrate")

            migrated_count = 0
            failed_count = 0

            for memory_row in memories:
                try:
                    # Convert SQLite row to dict
                    memory_data = {
                        "id": memory_row.get("id"),
                        "content": memory_row.get("content"),
                        "memory_type": memory_row.get("memory_type", "interaction"),
                        "importance_score": memory_row.get("importance_score", 0.5),
                        "metadata": json.loads(memory_row.get("metadata", "{}")),
                        "cxd_classification": memory_row.get("cxd_classification"),
                        "cxd_confidence": memory_row.get("cxd_confidence"),
                        "created_at": memory_row.get("created_at"),
                        "updated_at": memory_row.get("updated_at"),
                        "last_accessed": memory_row.get("last_accessed"),
                        "access_count": memory_row.get("access_count", 0),
                        "is_active": memory_row.get("is_active", True)
                    }

                    # Generate shard key and determine target shard
                    shard_key = self.get_shard_key(memory_data)
                    shard_id = self.get_shard_id(shard_key)
                    memory_data["shard_key"] = shard_key

                    # Insert into appropriate shard
                    shard_pool = self.shard_pools[shard_id]
                    
                    async with shard_pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO memories (
                                id, content, memory_type, importance_score, metadata,
                                cxd_classification, cxd_confidence, created_at, updated_at,
                                last_accessed, access_count, is_active, shard_key
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                            )
                        """, 
                            memory_data["id"],
                            memory_data["content"],
                            memory_data["memory_type"],
                            memory_data["importance_score"],
                            json.dumps(memory_data["metadata"]),
                            memory_data["cxd_classification"],
                            memory_data["cxd_confidence"],
                            memory_data["created_at"],
                            memory_data["updated_at"],
                            memory_data["last_accessed"],
                            memory_data["access_count"],
                            memory_data["is_active"],
                            memory_data["shard_key"]
                        )

                    migrated_count += 1
                    
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count} memories...")

                except Exception as e:
                    logger.error(f"Failed to migrate memory {memory_row.get('id')}: {e}")
                    failed_count += 1

            logger.info(f"Memory migration complete: {migrated_count} migrated, {failed_count} failed")

        finally:
            sqlite_conn.close()

    async def migrate_tales(self):
        """Migrate tales from file system to PostgreSQL"""
        logger.info("Starting tales migration...")
        
        tales_dir = Path("tales")
        if not tales_dir.exists():
            logger.warning("Tales directory not found, skipping tales migration")
            return

        migrated_count = 0
        failed_count = 0

        async with self.pg_pool.acquire() as conn:
            for tale_file in tales_dir.rglob("*.txt"):
                try:
                    # Extract category from path
                    relative_path = tale_file.relative_to(tales_dir)
                    category = "/".join(relative_path.parent.parts)
                    name = tale_file.stem

                    # Read content
                    content = tale_file.read_text(encoding="utf-8")

                    # Insert into PostgreSQL
                    await conn.execute("""
                        INSERT INTO tales (name, category, content, created_at, updated_at)
                        VALUES ($1, $2, $3, NOW(), NOW())
                        ON CONFLICT (name, category) DO UPDATE SET
                            content = EXCLUDED.content,
                            updated_at = NOW(),
                            version = tales.version + 1
                    """, name, category, content)

                    migrated_count += 1

                except Exception as e:
                    logger.error(f"Failed to migrate tale {tale_file}: {e}")
                    failed_count += 1

        logger.info(f"Tales migration complete: {migrated_count} migrated, {failed_count} failed")

    async def create_indexes(self):
        """Create performance indexes on migrated data"""
        logger.info("Creating performance indexes...")
        
        # Create indexes on all shards
        for shard_id, pool in self.shard_pools.items():
            async with pool.acquire() as conn:
                await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_shard_memories_type ON memories(memory_type)")
                await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_shard_memories_importance ON memories(importance_score DESC)")
                await conn.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_shard_memories_active ON memories(is_active) WHERE is_active = true")
                logger.info(f"Created indexes on shard {shard_id}")

    async def verify_migration(self):
        """Verify migration integrity"""
        logger.info("Verifying migration...")
        
        # Count records in SQLite
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        sqlite_count = cursor.fetchone()[0]
        sqlite_conn.close()

        # Count records across all shards
        total_pg_count = 0
        for shard_id, pool in self.shard_pools.items():
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM memories")
                total_pg_count += result
                logger.info(f"Shard {shard_id}: {result} memories")

        logger.info(f"SQLite total: {sqlite_count}")
        logger.info(f"PostgreSQL total: {total_pg_count}")
        
        if sqlite_count == total_pg_count:
            logger.info("✅ Migration verification successful!")
        else:
            logger.error("❌ Migration verification failed - record counts don't match")

    async def close_connections(self):
        """Close all database connections"""
        logger.info("Closing database connections...")
        
        if self.pg_pool:
            await self.pg_pool.close()
            
        for pool in self.shard_pools.values():
            await pool.close()

    async def migrate(self):
        """Run full migration process"""
        logger.info("Starting MemMimic SQLite to PostgreSQL migration...")
        
        try:
            await self.initialize_connections()
            await self.migrate_memories()
            await self.migrate_tales()
            await self.create_indexes()
            await self.verify_migration()
            logger.info("🎉 Migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            await self.close_connections()


async def main():
    """Main migration function"""
    
    # Configuration
    postgres_config = {
        "host": "localhost",
        "port": 5432,
        "user": "memmimic",
        "password": "your_password",
        "database": "memmimic",
        "min_size": 1,
        "max_size": 5
    }
    
    shard_configs = [
        {
            "shard_id": 1,
            "connection": {
                "host": "localhost", "port": 5432,
                "user": "memmimic", "password": "your_password",
                "database": "memmimic_shard_1", "min_size": 1, "max_size": 3
            }
        },
        {
            "shard_id": 2,
            "connection": {
                "host": "localhost", "port": 5432,
                "user": "memmimic", "password": "your_password", 
                "database": "memmimic_shard_2", "min_size": 1, "max_size": 3
            }
        },
        {
            "shard_id": 3,
            "connection": {
                "host": "localhost", "port": 5432,
                "user": "memmimic", "password": "your_password",
                "database": "memmimic_shard_3", "min_size": 1, "max_size": 3
            }
        },
        {
            "shard_id": 4,
            "connection": {
                "host": "localhost", "port": 5432,
                "user": "memmimic", "password": "your_password",
                "database": "memmimic_shard_4", "min_size": 1, "max_size": 3
            }
        }
    ]
    
    # Run migration
    migrator = DatabaseMigrator(
        sqlite_path="memmimic.db",
        postgres_config=postgres_config,
        shard_configs=shard_configs
    )
    
    await migrator.migrate()


if __name__ == "__main__":
    asyncio.run(main())