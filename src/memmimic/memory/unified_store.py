#!/usr/bin/env python3
"""
MemMimic Unified Memory Store
Bridges ActiveMemoryPool with existing MemoryStore interface for seamless integration
"""

import logging
from typing import Any, Dict, List, Optional

from ..config import MemMimicConfig, get_config
from .active_manager import ActiveMemoryConfig, ActiveMemoryPool
from .memory import Memory, MemoryStore


class UnifiedMemoryStore:
    """
    Unified memory store that provides backward compatibility with MemoryStore
    while leveraging the Advanced Memory Management System (AMMS) underneath.

    This class acts as a bridge between the legacy MemoryStore interface
    and the new ActiveMemoryPool system, ensuring seamless integration
    without breaking existing code.
    """

    def __init__(self, db_path: str = "memories.db", config_path: Optional[str] = None):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Load MemMimic configuration
        try:
            self.memmimic_config = get_config(config_path)
            self.logger.info("Loaded MemMimic configuration successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}, using defaults")
            self.memmimic_config = get_config()  # Get default config

        # Convert MemMimic config to ActiveMemoryConfig
        self.active_config = self._convert_config(self.memmimic_config)

        # Initialize the Active Memory Pool
        self.active_pool = ActiveMemoryPool(db_path, self.active_config)

        # Maintain reference to legacy store for compatibility if needed
        self.legacy_store = MemoryStore(db_path)

        # Track compatibility mode
        self._compatibility_mode = False

        self.logger.info(f"UnifiedMemoryStore initialized with AMMS (db: {db_path})")

    def _convert_config(self, memmimic_config: MemMimicConfig) -> ActiveMemoryConfig:
        """Convert MemMimicConfig to ActiveMemoryConfig format"""
        pool_config = memmimic_config.active_memory_pool
        cleanup_config = memmimic_config.cleanup_policies
        scoring_config = memmimic_config.scoring_weights

        return ActiveMemoryConfig(
            target_pool_size=pool_config.target_size,
            max_pool_size=pool_config.max_size,
            importance_threshold=pool_config.importance_threshold,
            stale_threshold_days=cleanup_config.stale_threshold_days,
            archive_threshold=cleanup_config.archive_threshold,
            prune_threshold=cleanup_config.prune_threshold,
            batch_size=pool_config.batch_size,
            max_query_time_ms=pool_config.max_query_time_ms,
            cache_size=pool_config.cache_size,
            cxd_weight=scoring_config.cxd_classification,
            access_frequency_weight=scoring_config.access_frequency,
            recency_weight=scoring_config.recency_temporal,
            confidence_weight=scoring_config.confidence_quality,
            type_weight=scoring_config.memory_type,
        )

    # === CORE MEMORY OPERATIONS (MemoryStore compatibility) ===

    def add(self, memory: Memory) -> int:
        """
        Add a memory to the active pool (compatible with MemoryStore.add)

        Args:
            memory: Memory object to add

        Returns:
            int: Memory ID of the added memory
        """
        try:
            # Use ActiveMemoryPool for enhanced storage
            memory_id = self.active_pool.add_memory(memory)
            self.logger.debug(f"Added memory {memory_id} via AMMS")
            return memory_id

        except Exception as e:
            self.logger.error(f"AMMS add failed: {e}, falling back to legacy")
            self._compatibility_mode = True
            return self.legacy_store.add(memory)

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search memories using AMMS intelligent ranking (compatible with MemoryStore.search)

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List[Memory]: List of relevant memories
        """
        try:
            if self._compatibility_mode:
                return self.legacy_store.search(query, limit)

            # Use ActiveMemoryPool for enhanced search
            results = self.active_pool.search_active_memories(query, limit)

            # Convert back to Memory objects for compatibility
            memories = []
            for result in results:
                memory = Memory(
                    content=result["content"],
                    memory_type=result["type"],
                    confidence=result["confidence"],
                )
                memory.created_at = result["created_at"]
                memory.access_count = result.get("access_count", 0)
                memory.id = result["id"]
                memories.append(memory)

            self.logger.debug(
                f"AMMS search returned {len(memories)} results for: {query}"
            )
            return memories

        except Exception as e:
            self.logger.error(f"AMMS search failed: {e}, falling back to legacy")
            self._compatibility_mode = True
            return self.legacy_store.search(query, limit)

    def get_recent(self, hours: int = 24) -> List[Memory]:
        """
        Get recent memories (compatible with MemoryStore.get_recent)

        Args:
            hours: Number of hours to look back

        Returns:
            List[Memory]: List of recent memories
        """
        try:
            if self._compatibility_mode:
                return self.legacy_store.get_recent(hours)

            # Use ActiveMemoryPool status to get recent memories
            status = self.active_pool.get_active_pool_status()
            recent_ids = status.get("recent_memory_ids", [])

            # Convert to Memory objects (simplified implementation)
            # In a full implementation, we'd fetch these from the enhanced schema
            memories = []
            with self.active_pool._get_connection() as conn:
                placeholders = ",".join(["?"] * len(recent_ids[:20]))  # Limit to 20
                if placeholders:
                    cursor = conn.execute(
                        f"""
                        SELECT * FROM memories_enhanced 
                        WHERE id IN ({placeholders})
                        ORDER BY created_at DESC
                    """,
                        recent_ids[:20],
                    )

                    for row in cursor.fetchall():
                        memory = self._row_to_memory(row)
                        memories.append(memory)

            return memories

        except Exception as e:
            self.logger.error(f"AMMS get_recent failed: {e}, falling back to legacy")
            self._compatibility_mode = True
            return self.legacy_store.get_recent(hours)

    def get_all(self) -> List[Memory]:
        """
        Get all memories (compatible with MemoryStore.get_all)
        Note: This returns only active memories for performance

        Returns:
            List[Memory]: List of all active memories
        """
        try:
            if self._compatibility_mode:
                return self.legacy_store.get_all()

            # Get all active memories from AMMS
            memories = []
            with self.active_pool._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM memories_enhanced 
                    WHERE archive_status = 'active'
                    ORDER BY created_at DESC
                """
                )

                for row in cursor.fetchall():
                    memory = self._row_to_memory(row)
                    memories.append(memory)

            self.logger.debug(f"Retrieved {len(memories)} active memories")
            return memories

        except Exception as e:
            self.logger.error(f"AMMS get_all failed: {e}, falling back to legacy")
            self._compatibility_mode = True
            return self.legacy_store.get_all()

    # === ENHANCED AMMS FEATURES ===

    def get_active_pool_status(self) -> Dict[str, Any]:
        """Get detailed status of the active memory pool"""
        try:
            return self.active_pool.get_active_pool_status()
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {"error": str(e), "compatibility_mode": self._compatibility_mode}

    def search_with_ranking(
        self,
        query: str,
        limit: int = 10,
        boost_factors: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with detailed ranking information

        Args:
            query: Search query
            limit: Maximum results
            boost_factors: Optional ranking boost factors

        Returns:
            List[Dict]: Detailed results with ranking scores
        """
        try:
            return self.active_pool.search_active_memories(query, limit, boost_factors)
        except Exception as e:
            self.logger.error(f"Enhanced search failed: {e}")
            # Fallback to basic search
            basic_results = self.search(query, limit)
            return [
                {
                    "content": m.content,
                    "type": m.type,
                    "confidence": m.confidence,
                    "id": getattr(m, "id", 0),
                    "score": 0.5,
                }
                for m in basic_results
            ]

    def trigger_maintenance(self) -> Dict[str, Any]:
        """Manually trigger memory pool maintenance"""
        try:
            return self.active_pool.perform_maintenance()
        except Exception as e:
            self.logger.error(f"Maintenance failed: {e}")
            return {"error": str(e)}

    def get_memory_importance(self, memory_id: int) -> float:
        """Get importance score for a specific memory"""
        try:
            with self.active_pool._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT importance_score FROM memories_enhanced WHERE id = ?",
                    (memory_id,),
                )
                row = cursor.fetchone()
                return row["importance_score"] if row else 0.0
        except Exception as e:
            self.logger.error(f"Failed to get importance for memory {memory_id}: {e}")
            return 0.0

    # === UTILITY METHODS ===

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object (enhanced schema compatible)"""
        memory = Memory(
            content=row["content"],
            memory_type=row["type"],
            confidence=row["confidence"],
        )
        memory.created_at = row["created_at"]
        memory.access_count = row.get("access_count", 0)
        memory.id = row["id"]

        # Add AMMS-specific attributes if available
        if "importance_score" in row.keys():
            memory.importance_score = row["importance_score"]
        if "archive_status" in row.keys():
            memory.archive_status = row["archive_status"]
        if "cxd_function" in row.keys():
            memory.cxd_function = row["cxd_function"]

        return memory

    def migrate_from_legacy(self) -> Dict[str, Any]:
        """
        Migrate memories from legacy MemoryStore to AMMS enhanced schema

        Returns:
            Dict: Migration results
        """
        try:
            self.logger.info("Starting migration from legacy MemoryStore to AMMS...")

            # Get all memories from legacy store
            legacy_memories = self.legacy_store.get_all()
            migrated_count = 0
            errors = []

            for memory in legacy_memories:
                try:
                    # Add to AMMS (which will calculate importance, etc.)
                    memory_id = self.active_pool.add_memory(memory)
                    migrated_count += 1

                    if migrated_count % 100 == 0:
                        self.logger.info(f"Migrated {migrated_count} memories...")

                except Exception as e:
                    errors.append(f"Memory {getattr(memory, 'id', 'unknown')}: {e}")

            result = {
                "total_legacy_memories": len(legacy_memories),
                "successfully_migrated": migrated_count,
                "errors": len(errors),
                "error_details": errors[:10],  # First 10 errors
            }

            self.logger.info(f"Migration completed: {result}")
            return result

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def enable_compatibility_mode(self, enable: bool = True):
        """Enable/disable compatibility mode (uses legacy MemoryStore)"""
        self._compatibility_mode = enable
        self.logger.info(f"Compatibility mode {'enabled' if enable else 'disabled'}")

    @property
    def is_amms_active(self) -> bool:
        """Check if AMMS is active (not in compatibility mode)"""
        return not self._compatibility_mode

    def __repr__(self):
        mode = "AMMS" if not self._compatibility_mode else "Legacy"
        return f"UnifiedMemoryStore(db='{self.db_path}', mode='{mode}')"
