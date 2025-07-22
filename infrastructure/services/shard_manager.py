"""
MemMimic Shard Manager
Handles database sharding, routing, and load balancing across PostgreSQL shards
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Shard configuration"""
    shard_id: int
    shard_name: str
    connection_string: str
    is_active: bool = True
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None


class ShardManager:
    """Manages database sharding and routing for MemMimic"""
    
    def __init__(self, coordinator_config: Dict[str, Any]):
        self.coordinator_config = coordinator_config
        self.coordinator_pool: Optional[Pool] = None
        self.shard_pools: Dict[int, Pool] = {}
        self.shard_configs: Dict[int, ShardConfig] = {}
        self.health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize shard manager and connections"""
        logger.info("Initializing shard manager...")
        
        # Connect to coordinator
        self.coordinator_pool = await asyncpg.create_pool(**self.coordinator_config)
        
        # Load shard configurations from coordinator
        await self._load_shard_configs()
        
        # Initialize shard connections
        await self._initialize_shard_pools()
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Shard manager initialized with {len(self.shard_pools)} active shards")
    
    async def _load_shard_configs(self):
        """Load shard configurations from coordinator database"""
        async with self.coordinator_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT shard_id, shard_name, connection_string, is_active
                FROM shard_config
                ORDER BY shard_id
            """)
            
            for row in rows:
                config = ShardConfig(
                    shard_id=row["shard_id"],
                    shard_name=row["shard_name"], 
                    connection_string=row["connection_string"],
                    is_active=row["is_active"]
                )
                self.shard_configs[config.shard_id] = config
                
        logger.info(f"Loaded {len(self.shard_configs)} shard configurations")
    
    async def _initialize_shard_pools(self):
        """Initialize connection pools for all active shards"""
        for shard_id, config in self.shard_configs.items():
            if not config.is_active:
                continue
                
            try:
                # Parse connection string into connection parameters
                # This is a simplified parser - in production, use proper URL parsing
                conn_params = self._parse_connection_string(config.connection_string)
                
                pool = await asyncpg.create_pool(**conn_params)
                self.shard_pools[shard_id] = pool
                
                # Test connection
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    
                config.health_status = "healthy"
                logger.info(f"Shard {shard_id} ({config.shard_name}) connected successfully")
                
            except Exception as e:
                logger.error(f"Failed to connect to shard {shard_id}: {e}")
                config.health_status = "unhealthy"
    
    def _parse_connection_string(self, conn_str: str) -> Dict[str, Any]:
        """Parse PostgreSQL connection string into parameters"""
        # Simplified parser - in production, use a proper URL parser
        # postgresql://user:password@host:port/database
        if not conn_str.startswith("postgresql://"):
            raise ValueError(f"Invalid connection string format: {conn_str}")
            
        # Remove protocol prefix
        conn_str = conn_str[13:]  # Remove 'postgresql://'
        
        # Split user:password@host:port/database
        if "@" in conn_str:
            auth_part, host_part = conn_str.split("@", 1)
            if ":" in auth_part:
                user, password = auth_part.split(":", 1)
            else:
                user, password = auth_part, None
        else:
            user, password = None, None
            host_part = conn_str
            
        if "/" in host_part:
            host_port, database = host_part.split("/", 1)
        else:
            host_port, database = host_part, "postgres"
            
        if ":" in host_port:
            host, port = host_port.split(":", 1)
            port = int(port)
        else:
            host, port = host_port, 5432
            
        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "min_size": 1,
            "max_size": 5
        }
    
    def get_shard_key(self, content: str, memory_type: str = "interaction") -> str:
        """Generate shard key for memory content"""
        # Create deterministic shard key based on content and type
        key_data = f"{memory_type}:{content[:100]}"  # First 100 chars
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get_shard_id(self, shard_key: str) -> int:
        """Determine target shard for a given shard key"""
        # Use consistent hashing
        hash_value = int(hashlib.md5(shard_key.encode()).hexdigest()[:8], 16)
        active_shards = [sid for sid, config in self.shard_configs.items() 
                        if config.is_active and config.health_status == "healthy"]
        
        if not active_shards:
            raise RuntimeError("No healthy shards available")
            
        return active_shards[hash_value % len(active_shards)]
    
    @asynccontextmanager
    async def get_shard_connection(self, shard_id: int):
        """Get connection to specific shard"""
        if shard_id not in self.shard_pools:
            raise ValueError(f"Shard {shard_id} not available")
            
        pool = self.shard_pools[shard_id]
        async with pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_coordinator_connection(self):
        """Get connection to coordinator database"""
        async with self.coordinator_pool.acquire() as conn:
            yield conn
    
    async def execute_on_shard(self, shard_id: int, query: str, *args) -> Any:
        """Execute query on specific shard"""
        async with self.get_shard_connection(shard_id) as conn:
            return await conn.execute(query, *args)
    
    async def fetch_from_shard(self, shard_id: int, query: str, *args) -> List[asyncpg.Record]:
        """Fetch results from specific shard"""
        async with self.get_shard_connection(shard_id) as conn:
            return await conn.fetch(query, *args)
    
    async def fetch_from_all_shards(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute query on all active shards and combine results"""
        results = []
        
        for shard_id in self.shard_pools:
            try:
                shard_results = await self.fetch_from_shard(shard_id, query, *args)
                results.extend(shard_results)
            except Exception as e:
                logger.error(f"Failed to execute query on shard {shard_id}: {e}")
                
        return results
    
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store memory in appropriate shard"""
        # Generate shard key
        shard_key = self.get_shard_key(
            memory_data["content"], 
            memory_data.get("memory_type", "interaction")
        )
        memory_data["shard_key"] = shard_key
        
        # Determine target shard
        shard_id = self.get_shard_id(shard_key)
        
        # Store in shard
        async with self.get_shard_connection(shard_id) as conn:
            memory_id = await conn.fetchval("""
                INSERT INTO memories (
                    content, memory_type, importance_score, metadata,
                    cxd_classification, cxd_confidence, shard_key
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """,
                memory_data["content"],
                memory_data.get("memory_type", "interaction"),
                memory_data.get("importance_score", 0.5),
                json.dumps(memory_data.get("metadata", {})),
                memory_data.get("cxd_classification"),
                memory_data.get("cxd_confidence"),
                shard_key
            )
            
        logger.debug(f"Stored memory {memory_id} in shard {shard_id}")
        return str(memory_id)
    
    async def search_memories(self, 
                            query: str,
                            memory_type: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories across all shards"""
        search_query = """
            SELECT id, content, memory_type, importance_score, metadata,
                   cxd_classification, cxd_confidence, created_at, shard_key
            FROM memories 
            WHERE is_active = true
        """
        
        params = []
        param_count = 0
        
        # Add content search
        if query:
            param_count += 1
            search_query += f" AND content ILIKE ${param_count}"
            params.append(f"%{query}%")
        
        # Add memory type filter
        if memory_type:
            param_count += 1
            search_query += f" AND memory_type = ${param_count}"
            params.append(memory_type)
        
        search_query += " ORDER BY importance_score DESC, created_at DESC"
        
        # Add limit per shard (we'll combine and re-limit later)
        search_query += f" LIMIT {limit * 2}"
        
        # Execute on all shards
        all_results = await self.fetch_from_all_shards(search_query, *params)
        
        # Convert to dictionaries and sort by importance
        results = []
        for row in all_results:
            result = {
                "id": str(row["id"]),
                "content": row["content"],
                "memory_type": row["memory_type"],
                "importance_score": row["importance_score"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "cxd_classification": row["cxd_classification"],
                "cxd_confidence": row["cxd_confidence"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "shard_key": row["shard_key"]
            }
            results.append(result)
        
        # Sort by importance and limit
        results.sort(key=lambda x: x["importance_score"], reverse=True)
        return results[:limit]
    
    async def _health_check_loop(self):
        """Periodic health check for all shards"""
        while True:
            try:
                await self._check_shard_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)  # Shorter retry interval on error
    
    async def _check_shard_health(self):
        """Check health of all shards"""
        for shard_id, config in self.shard_configs.items():
            if not config.is_active:
                continue
                
            try:
                if shard_id in self.shard_pools:
                    async with self.shard_pools[shard_id].acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    
                    old_status = config.health_status
                    config.health_status = "healthy"
                    config.last_health_check = datetime.now()
                    
                    if old_status != "healthy":
                        logger.info(f"Shard {shard_id} is now healthy")
                        
            except Exception as e:
                old_status = config.health_status
                config.health_status = "unhealthy"
                config.last_health_check = datetime.now()
                
                if old_status == "healthy":
                    logger.error(f"Shard {shard_id} became unhealthy: {e}")
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics"""
        stats = {
            "total_shards": len(self.shard_configs),
            "active_shards": sum(1 for c in self.shard_configs.values() if c.is_active),
            "healthy_shards": sum(1 for c in self.shard_configs.values() 
                                if c.health_status == "healthy"),
            "shard_status": {}
        }
        
        # Get memory count from each shard
        for shard_id, pool in self.shard_pools.items():
            try:
                async with pool.acquire() as conn:
                    memory_count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE is_active = true")
                    stats["shard_status"][shard_id] = {
                        "memory_count": memory_count,
                        "status": self.shard_configs[shard_id].health_status,
                        "last_check": self.shard_configs[shard_id].last_health_check.isoformat() 
                                    if self.shard_configs[shard_id].last_health_check else None
                    }
            except Exception as e:
                stats["shard_status"][shard_id] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return stats
    
    async def close(self):
        """Close all connections and cleanup"""
        logger.info("Shutting down shard manager...")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close shard pools
        for pool in self.shard_pools.values():
            await pool.close()
        
        # Close coordinator pool
        if self.coordinator_pool:
            await self.coordinator_pool.close()
            
        logger.info("Shard manager shutdown complete")


# Global shard manager instance
_shard_manager: Optional[ShardManager] = None


def get_shard_manager() -> ShardManager:
    """Get global shard manager instance"""
    if _shard_manager is None:
        raise RuntimeError("Shard manager not initialized")
    return _shard_manager


async def initialize_shard_manager(coordinator_config: Dict[str, Any]) -> ShardManager:
    """Initialize global shard manager"""
    global _shard_manager
    _shard_manager = ShardManager(coordinator_config)
    await _shard_manager.initialize()
    return _shard_manager