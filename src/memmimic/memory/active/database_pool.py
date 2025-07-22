"""
High-performance database connection pooling with transaction management.

Provides connection pooling, transaction coordination, and automatic failover
for optimal database performance and reliability in active memory management.
"""

import logging
import sqlite3
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue, Empty, Full
from typing import Any, Dict, List, Optional, Tuple
import weakref

from .interfaces import (
    DatabasePool, DatabasePoolError, ThreadSafeMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Wrapper for pooled database connections"""
    connection: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    is_healthy: bool = True
    transaction_active: bool = False
    connection_id: str = field(default_factory=lambda: str(id(object())))
    
    def mark_used(self):
        """Mark connection as recently used"""
        self.last_used = datetime.now()
        self.use_count += 1
    
    def age_seconds(self) -> float:
        """Get connection age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def idle_seconds(self) -> float:
        """Get idle time in seconds"""
        return (datetime.now() - self.last_used).total_seconds()


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pool"""
    # Pool sizing
    min_connections: int = 5
    max_connections: int = 20
    
    # Connection lifecycle
    max_connection_age_seconds: int = 3600  # 1 hour
    max_idle_seconds: int = 300  # 5 minutes
    connection_timeout_seconds: int = 30
    
    # Health and monitoring
    health_check_interval_seconds: int = 60
    enable_health_checks: bool = True
    
    # Performance
    enable_prepared_statements: bool = True
    enable_wal_mode: bool = True
    busy_timeout_ms: int = 30000
    
    # Transaction settings
    default_isolation_level: str = "DEFERRED"
    transaction_timeout_seconds: int = 30


class ConnectionPool(DatabasePool):
    """
    High-performance database connection pool with health monitoring.
    
    Features:
    - Dynamic pool sizing based on demand
    - Connection health monitoring and replacement
    - Transaction coordination and isolation
    - Automatic connection recycling
    - Performance metrics and monitoring
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str, config: Optional[ConnectionPoolConfig] = None):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database file
            config: Pool configuration (uses defaults if None)
        """
        self.db_path = db_path
        self.config = config or ConnectionPoolConfig()
        
        # Connection pool
        self._pool: Queue[PooledConnection] = Queue(maxsize=self.config.max_connections)
        self._all_connections: Dict[str, PooledConnection] = {}
        self._pool_lock = threading.RLock()
        
        # Performance metrics
        self._metrics = ThreadSafeMetrics()
        self._start_time = time.time()
        
        # Health monitoring
        self._health_thread = None
        self._stop_health_check = threading.Event()
        
        # Prepared statements cache
        self._prepared_statements: Dict[str, str] = {}
        
        # Initialize pool
        self._initialize_pool()
        self._start_health_monitoring()
        
        logger.info(f"ConnectionPool initialized: {self.config.min_connections}-{self.config.max_connections} connections")
    
    def get_connection(self) -> PooledConnection:
        """
        Get database connection from pool.
        
        Returns:
            PooledConnection instance
            
        Raises:
            DatabasePoolError: If unable to get connection within timeout
        """
        start_time = time.perf_counter()
        
        try:
            self._metrics.increment_counter('connection_requests')
            
            # Try to get connection from pool
            try:
                pooled_conn = self._pool.get(timeout=self.config.connection_timeout_seconds)
                
                # Validate connection health
                if self._is_connection_healthy(pooled_conn):
                    pooled_conn.mark_used()
                    self._metrics.increment_counter('connections_reused')
                    return pooled_conn
                else:
                    # Connection is unhealthy, replace it
                    self._close_connection(pooled_conn)
                    pooled_conn = self._create_new_connection()
                    
            except Empty:
                # Pool is empty, try to create new connection
                if len(self._all_connections) < self.config.max_connections:
                    pooled_conn = self._create_new_connection()
                else:
                    # Pool is at max capacity, wait for a connection
                    pooled_conn = self._pool.get(timeout=self.config.connection_timeout_seconds)
                    if not self._is_connection_healthy(pooled_conn):
                        self._close_connection(pooled_conn)
                        pooled_conn = self._create_new_connection()
            
            pooled_conn.mark_used()
            return pooled_conn
            
        except Exception as e:
            self._metrics.increment_counter('connection_errors')
            logger.error(f"Failed to get connection: {e}")
            raise DatabasePoolError(
                f"Failed to get database connection: {e}",
                error_code="CONNECTION_UNAVAILABLE",
                context={'db_path': self.db_path, 'pool_size': len(self._all_connections)}
            )
        
        finally:
            get_time = (time.perf_counter() - start_time) * 1000
            self._metrics.set_gauge('last_get_connection_time_ms', get_time)
    
    def return_connection(self, pooled_conn: PooledConnection) -> None:
        """
        Return connection to pool.
        
        Args:
            pooled_conn: PooledConnection to return
        """
        try:
            with self._pool_lock:
                # Check if connection should be recycled
                if (pooled_conn.age_seconds() > self.config.max_connection_age_seconds or
                    not self._is_connection_healthy(pooled_conn)):
                    
                    self._close_connection(pooled_conn)
                    
                    # Create replacement if below minimum
                    if len(self._all_connections) < self.config.min_connections:
                        replacement = self._create_new_connection()
                        self._pool.put_nowait(replacement)
                else:
                    # Return healthy connection to pool
                    try:
                        self._pool.put_nowait(pooled_conn)
                        self._metrics.increment_counter('connections_returned')
                    except Full:
                        # Pool is full, close excess connection
                        self._close_connection(pooled_conn)
                        
        except Exception as e:
            logger.error(f"Failed to return connection: {e}")
            # Try to close the connection to prevent leaks
            try:
                self._close_connection(pooled_conn)
            except Exception:
                pass
    
    @contextmanager
    def get_connection_context(self):
        """
        Context manager for automatic connection management.
        
        Yields:
            Database connection that is automatically returned to pool
        """
        pooled_conn = self.get_connection()
        try:
            yield pooled_conn.connection
        finally:
            self.return_connection(pooled_conn)
    
    def execute_query(self, query: str, params: tuple = ()) -> Any:
        """
        Execute query with automatic connection management.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Query result
        """
        with self.get_connection_context() as conn:
            try:
                cursor = conn.execute(query, params)
                if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
                    
            except Exception as e:
                conn.rollback()
                raise DatabasePoolError(
                    f"Query execution failed: {e}",
                    error_code="QUERY_EXECUTION_ERROR",
                    context={'query': query[:100], 'params': str(params)[:100]}
                )
    
    def execute_transaction(self, queries: List[Tuple[str, tuple]]) -> bool:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            True if transaction completed successfully
        """
        with self.get_connection_context() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")
                
                for query, params in queries:
                    conn.execute(query, params)
                
                conn.commit()
                self._metrics.increment_counter('transactions_committed')
                return True
                
            except Exception as e:
                conn.rollback()
                self._metrics.increment_counter('transactions_rolled_back')
                logger.error(f"Transaction failed: {e}")
                raise DatabasePoolError(
                    f"Transaction failed: {e}",
                    error_code="TRANSACTION_ERROR",
                    context={'query_count': len(queries)}
                )
    
    @contextmanager
    def transaction_context(self, isolation_level: Optional[str] = None):
        """
        Context manager for transactions with automatic rollback on error.
        
        Args:
            isolation_level: Transaction isolation level
            
        Yields:
            Database connection within transaction context
        """
        pooled_conn = self.get_connection()
        conn = pooled_conn.connection
        
        try:
            pooled_conn.transaction_active = True
            
            # Set isolation level if specified
            if isolation_level:
                conn.isolation_level = isolation_level
            
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.commit()
            self._metrics.increment_counter('transactions_committed')
            
        except Exception as e:
            conn.rollback()
            self._metrics.increment_counter('transactions_rolled_back')
            logger.error(f"Transaction context failed: {e}")
            raise
        
        finally:
            pooled_conn.transaction_active = False
            self.return_connection(pooled_conn)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dictionary containing pool metrics and health data
        """
        try:
            with self._pool_lock:
                metrics = self._metrics.get_all_metrics()
                
                # Pool state
                available_connections = self._pool.qsize()
                total_connections = len(self._all_connections)
                active_connections = total_connections - available_connections
                
                # Connection age analysis
                if self._all_connections:
                    connection_ages = [conn.age_seconds() for conn in self._all_connections.values()]
                    avg_age = sum(connection_ages) / len(connection_ages)
                    max_age = max(connection_ages)
                else:
                    avg_age = max_age = 0
                
                # Usage analysis
                total_requests = metrics.get('connection_requests', 0)
                connections_reused = metrics.get('connections_reused', 0)
                reuse_rate = connections_reused / total_requests if total_requests > 0 else 0.0
                
                uptime_seconds = time.time() - self._start_time
                
                return {
                    # Pool state
                    'total_connections': total_connections,
                    'available_connections': available_connections,
                    'active_connections': active_connections,
                    'min_connections': self.config.min_connections,
                    'max_connections': self.config.max_connections,
                    
                    # Pool utilization
                    'utilization': active_connections / self.config.max_connections,
                    'reuse_rate': reuse_rate,
                    
                    # Connection metrics
                    'avg_connection_age_seconds': avg_age,
                    'max_connection_age_seconds': max_age,
                    'connection_requests': total_requests,
                    'connections_created': metrics.get('connections_created', 0),
                    'connections_closed': metrics.get('connections_closed', 0),
                    'connections_reused': connections_reused,
                    'connection_errors': metrics.get('connection_errors', 0),
                    
                    # Transaction metrics
                    'transactions_committed': metrics.get('transactions_committed', 0),
                    'transactions_rolled_back': metrics.get('transactions_rolled_back', 0),
                    
                    # Performance
                    'last_get_connection_time_ms': metrics.get('last_get_connection_time_ms', 0),
                    'uptime_seconds': uptime_seconds,
                    
                    # Configuration
                    'db_path': self.db_path,
                    'health_checks_enabled': self.config.enable_health_checks,
                    'max_connection_age_seconds': self.config.max_connection_age_seconds,
                    'connection_timeout_seconds': self.config.connection_timeout_seconds,
                }
                
        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            raise DatabasePoolError(f"Failed to get pool stats: {e}")
    
    def health_check(self) -> bool:
        """
        Check pool health and connectivity.
        
        Returns:
            True if pool is healthy
        """
        try:
            stats = self.get_pool_stats()
            
            # Check basic connectivity
            with self.get_connection_context() as conn:
                conn.execute("SELECT 1").fetchone()
            
            # Health criteria
            healthy = (
                stats['total_connections'] >= self.config.min_connections and
                stats['connection_errors'] / max(stats['connection_requests'], 1) < 0.1 and  # <10% error rate
                stats['available_connections'] > 0
            )
            
            return healthy
            
        except Exception as e:
            logger.error(f"Pool health check failed: {e}")
            return False
    
    def _initialize_pool(self) -> None:
        """Initialize pool with minimum connections"""
        try:
            for _ in range(self.config.min_connections):
                pooled_conn = self._create_new_connection()
                self._pool.put_nowait(pooled_conn)
                
            logger.info(f"Pool initialized with {self.config.min_connections} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize pool: {e}")
            raise DatabasePoolError(f"Pool initialization failed: {e}")
    
    def _create_new_connection(self) -> PooledConnection:
        """Create new database connection"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.config.busy_timeout_ms / 1000.0,
                check_same_thread=False
            )
            
            # Configure connection
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA encoding = 'UTF-8'")
            
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode = WAL")
            
            # Set isolation level
            conn.isolation_level = self.config.default_isolation_level
            
            pooled_conn = PooledConnection(connection=conn)
            
            with self._pool_lock:
                self._all_connections[pooled_conn.connection_id] = pooled_conn
            
            self._metrics.increment_counter('connections_created')
            logger.debug(f"Created new connection: {pooled_conn.connection_id}")
            
            return pooled_conn
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise DatabasePoolError(f"Connection creation failed: {e}")
    
    def _close_connection(self, pooled_conn: PooledConnection) -> None:
        """Close and remove connection from pool"""
        try:
            conn_id = pooled_conn.connection_id
            
            # Close the actual connection
            pooled_conn.connection.close()
            
            # Remove from tracking
            with self._pool_lock:
                self._all_connections.pop(conn_id, None)
            
            self._metrics.increment_counter('connections_closed')
            logger.debug(f"Closed connection: {conn_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def _is_connection_healthy(self, pooled_conn: PooledConnection) -> bool:
        """Check if connection is healthy and usable"""
        try:
            # Check basic connectivity
            pooled_conn.connection.execute("SELECT 1").fetchone()
            
            # Check age and idle time
            if pooled_conn.age_seconds() > self.config.max_connection_age_seconds:
                return False
            
            if pooled_conn.idle_seconds() > self.config.max_idle_seconds:
                return False
            
            return pooled_conn.is_healthy
            
        except Exception:
            pooled_conn.is_healthy = False
            return False
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if not self.config.enable_health_checks:
            return
        
        def health_monitor():
            while not self._stop_health_check.wait(self.config.health_check_interval_seconds):
                try:
                    self._perform_health_maintenance()
                except Exception as e:
                    logger.error(f"Health monitoring failed: {e}")
        
        self._health_thread = threading.Thread(target=health_monitor, daemon=True)
        self._health_thread.start()
        logger.debug("Health monitoring thread started")
    
    def _perform_health_maintenance(self) -> None:
        """Perform periodic health maintenance"""
        with self._pool_lock:
            unhealthy_connections = []
            
            # Check all connections
            for conn_id, pooled_conn in self._all_connections.items():
                if not self._is_connection_healthy(pooled_conn):
                    unhealthy_connections.append(conn_id)
            
            # Close unhealthy connections
            for conn_id in unhealthy_connections:
                pooled_conn = self._all_connections.get(conn_id)
                if pooled_conn:
                    self._close_connection(pooled_conn)
            
            # Ensure minimum connections
            current_count = len(self._all_connections)
            if current_count < self.config.min_connections:
                needed = self.config.min_connections - current_count
                for _ in range(needed):
                    try:
                        new_conn = self._create_new_connection()
                        self._pool.put_nowait(new_conn)
                    except Exception as e:
                        logger.error(f"Failed to create replacement connection: {e}")
                        break
            
            if unhealthy_connections:
                logger.info(f"Health maintenance: replaced {len(unhealthy_connections)} connections")
    
    def shutdown(self) -> None:
        """Shutdown pool and close all connections"""
        try:
            # Stop health monitoring
            self._stop_health_check.set()
            if self._health_thread and self._health_thread.is_alive():
                self._health_thread.join(timeout=5.0)
            
            # Close all connections
            with self._pool_lock:
                for pooled_conn in list(self._all_connections.values()):
                    self._close_connection(pooled_conn)
                
                # Clear the pool queue
                while not self._pool.empty():
                    try:
                        self._pool.get_nowait()
                    except Empty:
                        break
            
            logger.info("Connection pool shutdown completed")
            
        except Exception as e:
            logger.error(f"Pool shutdown failed: {e}")
    
    def __del__(self):
        """Cleanup on garbage collection"""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup


def create_database_pool(db_path: str, config: Optional[ConnectionPoolConfig] = None) -> DatabasePool:
    """
    Factory function to create database connection pool.
    
    Args:
        db_path: Path to database file
        config: Pool configuration (uses defaults if None)
        
    Returns:
        DatabasePool instance
    """
    return ConnectionPool(db_path, config)