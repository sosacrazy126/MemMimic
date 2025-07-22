"""
Immutable Audit Logger - Core component for cryptographic audit logging.

Provides immutable audit trail logging with hash chain integrity verification
and cryptographic tamper detection for all MemMimic v2.0 operations.
"""

import hashlib
import json
import time
import uuid
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path

from ..errors import get_error_logger, handle_errors
from ..errors.exceptions import MemoryStorageError
from ..security.audit import SecurityEvent, SecurityEventType, SeverityLevel

logger = get_error_logger(__name__)


@dataclass
class AuditEntry:
    """Immutable audit entry with cryptographic verification."""
    
    # Core identification
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Operation details
    operation: str = ""
    component: str = ""
    memory_id: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Operation result and metadata
    operation_result: Dict[str, Any] = field(default_factory=dict)
    governance_status: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Security and compliance
    security_level: str = "standard"
    compliance_flags: List[str] = field(default_factory=list)
    
    # Cryptographic integrity
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    salt: Optional[str] = None
    
    def __post_init__(self):
        """Generate salt for cryptographic operations."""
        if not self.salt:
            self.salt = hashlib.sha256(f"{self.entry_id}:{time.time()}".encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary for hashing and storage."""
        data = asdict(self)
        # Convert datetime to ISO format for consistent hashing
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def get_canonical_data(self) -> Dict[str, Any]:
        """Get canonical data for hash calculation (excludes entry_hash)."""
        data = self.to_dict()
        # Remove entry_hash for canonical representation
        data.pop('entry_hash', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create AuditEntry from dictionary."""
        # Convert timestamp back to datetime if it's a string
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


class ImmutableAuditLogger:
    """
    Immutable audit logger with cryptographic verification.
    
    Features:
    - Immutable audit trails with hash chains
    - SHA-256 cryptographic verification
    - Performance-optimized <1ms logging overhead
    - Tamper detection and integrity validation
    - Secure persistent storage with SQLite
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_days: int = 90,
        enable_memory_buffer: bool = True,
        buffer_size: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize immutable audit logger.
        
        Args:
            db_path: Path to audit database (None for memory-only)
            retention_days: Retention period for audit entries
            enable_memory_buffer: Enable in-memory buffer for performance
            buffer_size: Size of memory buffer
            config: Additional configuration
        """
        self.db_path = Path(db_path) if db_path else None
        self.retention_days = retention_days
        self.config = config or {}
        
        # Hash chain management
        self._hash_chain: List[str] = []
        self._chain_lock = threading.RLock()
        self._verification_key = self._generate_verification_key()
        
        # Performance buffer
        self._enable_buffer = enable_memory_buffer
        self._buffer_size = buffer_size
        if enable_memory_buffer:
            self._memory_buffer: deque = deque(maxlen=buffer_size)
        
        # Performance metrics
        self._metrics = {
            'entries_logged': 0,
            'hash_verifications': 0,
            'integrity_checks': 0,
            'tamper_attempts': 0,
            'avg_log_time_ms': 0.0
        }
        
        # Initialize storage
        if self.db_path:
            self._init_database()
        
        # Background persistence for memory buffer
        self._persistence_thread = None
        if enable_memory_buffer and self.db_path:
            self._start_persistence_thread()
        
        logger.info(f"ImmutableAuditLogger initialized (db={db_path}, retention={retention_days}d)")
    
    def _generate_verification_key(self) -> str:
        """Generate cryptographic verification key for hash chains."""
        key_material = f"memmimic_audit_v2_{time.time()}_{uuid.uuid4()}"
        return hashlib.sha256(key_material.encode()).hexdigest()[:32]
    
    def _init_database(self) -> None:
        """Initialize audit database with security-optimized schema."""
        try:
            # Ensure directory exists
            if self.db_path:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Enable WAL mode for better concurrent performance
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                
                # Create audit entries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_entries (
                        entry_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        component TEXT NOT NULL,
                        memory_id TEXT,
                        user_context TEXT NOT NULL,
                        operation_result TEXT NOT NULL,
                        governance_status TEXT,
                        performance_metrics TEXT,
                        security_level TEXT NOT NULL DEFAULT 'standard',
                        compliance_flags TEXT,
                        previous_hash TEXT,
                        entry_hash TEXT NOT NULL UNIQUE,
                        salt TEXT NOT NULL,
                        created_at REAL NOT NULL DEFAULT (julianday('now'))
                    )
                """)
                
                # Create hash chain tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hash_chain (
                        chain_position INTEGER PRIMARY KEY,
                        entry_id TEXT NOT NULL,
                        entry_hash TEXT NOT NULL UNIQUE,
                        timestamp REAL NOT NULL,
                        verification_key_hash TEXT NOT NULL,
                        FOREIGN KEY (entry_id) REFERENCES audit_entries(entry_id)
                    )
                """)
                
                # Performance-optimized indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_entries(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_entries(operation, component)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_entries(memory_id) WHERE memory_id IS NOT NULL")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_hash ON audit_entries(entry_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chain_position ON hash_chain(chain_position)")
                
                # Load existing hash chain
                self._load_hash_chain(conn)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise
    
    def _load_hash_chain(self, conn: sqlite3.Connection) -> None:
        """Load existing hash chain from database."""
        try:
            cursor = conn.execute("""
                SELECT entry_hash FROM hash_chain 
                ORDER BY chain_position ASC
            """)
            
            with self._chain_lock:
                self._hash_chain = [row[0] for row in cursor.fetchall()]
            
            logger.debug(f"Loaded hash chain with {len(self._hash_chain)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load hash chain: {e}")
            self._hash_chain = []
    
    @handle_errors(MemoryStorageError)
    def log_audit_entry(
        self,
        operation: str,
        component: str,
        memory_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        operation_result: Optional[Dict[str, Any]] = None,
        governance_status: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        security_level: str = "standard",
        compliance_flags: Optional[List[str]] = None
    ) -> str:
        """
        Log immutable audit entry with cryptographic verification.
        
        Args:
            operation: Name of the operation being audited
            component: Component performing the operation
            memory_id: Associated memory ID if applicable
            user_context: User/session context information
            operation_result: Result of the operation
            governance_status: Governance validation status
            performance_metrics: Performance timing data
            security_level: Security classification level
            compliance_flags: Compliance requirements flags
            
        Returns:
            Entry ID of the logged audit entry
        """
        start_time = time.perf_counter()
        
        try:
            # Create audit entry
            entry = AuditEntry(
                operation=operation,
                component=component,
                memory_id=memory_id,
                user_context=user_context or {},
                operation_result=operation_result or {},
                governance_status=governance_status,
                performance_metrics=performance_metrics or {},
                security_level=security_level,
                compliance_flags=compliance_flags or []
            )
            
            # Generate cryptographic hash with chain integrity
            with self._chain_lock:
                entry.previous_hash = self._hash_chain[-1] if self._hash_chain else None
                entry.entry_hash = self._calculate_entry_hash(entry)
                self._hash_chain.append(entry.entry_hash)
            
            # Store entry
            if self._enable_buffer:
                self._memory_buffer.append(entry)
            
            if self.db_path and not self._enable_buffer:
                # Direct database storage for non-buffered mode
                self._store_entry_to_database(entry)
            
            # Update metrics
            log_time = (time.perf_counter() - start_time) * 1000
            self._metrics['entries_logged'] += 1
            self._update_avg_log_time(log_time)
            
            logger.debug(f"Audit entry logged: {entry.entry_id} ({log_time:.2f}ms)")
            return entry.entry_id
            
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
            raise
    
    def _calculate_entry_hash(self, entry: AuditEntry) -> str:
        """Calculate SHA-256 hash for audit entry with salt and verification key."""
        canonical_data = entry.get_canonical_data()
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        
        # Include salt and verification key for additional security
        hash_input = f"{canonical_json}:{entry.salt}:{self._verification_key}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def _store_entry_to_database(self, entry: AuditEntry) -> None:
        """Store audit entry to database."""
        if not self.db_path:
            return
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Insert audit entry
                conn.execute("""
                    INSERT INTO audit_entries (
                        entry_id, timestamp, operation, component, memory_id,
                        user_context, operation_result, governance_status,
                        performance_metrics, security_level, compliance_flags,
                        previous_hash, entry_hash, salt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.operation,
                    entry.component,
                    entry.memory_id,
                    json.dumps(entry.user_context),
                    json.dumps(entry.operation_result),
                    entry.governance_status,
                    json.dumps(entry.performance_metrics),
                    entry.security_level,
                    json.dumps(entry.compliance_flags),
                    entry.previous_hash,
                    entry.entry_hash,
                    entry.salt
                ))
                
                # Update hash chain table
                chain_position = len(self._hash_chain) - 1
                verification_key_hash = hashlib.sha256(self._verification_key.encode()).hexdigest()
                
                conn.execute("""
                    INSERT INTO hash_chain (
                        chain_position, entry_id, entry_hash, timestamp, verification_key_hash
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    chain_position,
                    entry.entry_id,
                    entry.entry_hash,
                    entry.timestamp.timestamp(),
                    verification_key_hash
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store audit entry to database: {e}")
            raise
    
    def _update_avg_log_time(self, log_time_ms: float) -> None:
        """Update average logging time metric."""
        current_avg = self._metrics['avg_log_time_ms']
        count = self._metrics['entries_logged']
        
        # Running average calculation
        self._metrics['avg_log_time_ms'] = ((current_avg * (count - 1)) + log_time_ms) / count
    
    def verify_entry_integrity(self, entry_id: str) -> bool:
        """
        Verify cryptographic integrity of a specific audit entry.
        
        Args:
            entry_id: ID of the entry to verify
            
        Returns:
            True if entry integrity is valid, False otherwise
        """
        start_time = time.perf_counter()
        
        try:
            entry = self._get_entry_by_id(entry_id)
            if not entry:
                return False
            
            # Recalculate hash and verify
            calculated_hash = self._calculate_entry_hash(entry)
            is_valid = calculated_hash == entry.entry_hash
            
            # Update metrics
            self._metrics['hash_verifications'] += 1
            if not is_valid:
                self._metrics['tamper_attempts'] += 1
                logger.warning(f"Hash verification failed for entry {entry_id}")
            
            verify_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Entry integrity verified: {entry_id} ({verify_time:.2f}ms, valid={is_valid})")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify entry integrity: {e}")
            return False
    
    def verify_hash_chain_integrity(self, start_position: int = 0) -> Dict[str, Any]:
        """
        Verify integrity of the entire hash chain.
        
        Args:
            start_position: Starting position in the chain (default: 0 for full chain)
            
        Returns:
            Dictionary with verification results and details
        """
        start_time = time.perf_counter()
        
        try:
            with self._chain_lock:
                chain_length = len(self._hash_chain)
                
                if start_position >= chain_length:
                    return {
                        'valid': True,
                        'entries_verified': 0,
                        'broken_links': [],
                        'verification_time_ms': 0.0
                    }
                
                broken_links = []
                entries_verified = 0
                
                # Verify hash chain links
                for i in range(max(1, start_position), chain_length):
                    current_hash = self._hash_chain[i]
                    expected_previous = self._hash_chain[i - 1]
                    
                    # Get entry and verify previous hash link
                    entry = self._get_entry_by_hash(current_hash)
                    if entry and entry.previous_hash != expected_previous:
                        broken_links.append({
                            'position': i,
                            'entry_id': entry.entry_id,
                            'expected_previous': expected_previous,
                            'actual_previous': entry.previous_hash
                        })
                    
                    entries_verified += 1
                
                verification_time = (time.perf_counter() - start_time) * 1000
                is_valid = len(broken_links) == 0
                
                # Update metrics
                self._metrics['integrity_checks'] += 1
                if not is_valid:
                    self._metrics['tamper_attempts'] += len(broken_links)
                
                result = {
                    'valid': is_valid,
                    'entries_verified': entries_verified,
                    'broken_links': broken_links,
                    'verification_time_ms': verification_time,
                    'chain_length': chain_length
                }
                
                logger.info(f"Hash chain integrity verified: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to verify hash chain integrity: {e}")
            return {
                'valid': False,
                'error': str(e),
                'entries_verified': 0,
                'broken_links': [],
                'verification_time_ms': 0.0
            }
    
    def _get_entry_by_id(self, entry_id: str) -> Optional[AuditEntry]:
        """Get audit entry by ID."""
        # Check memory buffer first
        if self._enable_buffer:
            for entry in reversed(self._memory_buffer):
                if entry.entry_id == entry_id:
                    return entry
        
        # Check database
        if self.db_path:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("""
                        SELECT * FROM audit_entries WHERE entry_id = ?
                    """, (entry_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_entry(row, cursor.description)
                        
            except Exception as e:
                logger.error(f"Failed to get entry by ID: {e}")
        
        return None
    
    def _get_entry_by_hash(self, entry_hash: str) -> Optional[AuditEntry]:
        """Get audit entry by hash."""
        # Check memory buffer first
        if self._enable_buffer:
            for entry in reversed(self._memory_buffer):
                if entry.entry_hash == entry_hash:
                    return entry
        
        # Check database
        if self.db_path:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("""
                        SELECT * FROM audit_entries WHERE entry_hash = ?
                    """, (entry_hash,))
                    
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_entry(row, cursor.description)
                        
            except Exception as e:
                logger.error(f"Failed to get entry by hash: {e}")
        
        return None
    
    def _row_to_entry(self, row: tuple, description: List[tuple]) -> AuditEntry:
        """Convert database row to AuditEntry."""
        # Create dict from row data
        columns = [col[0] for col in description]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        data['user_context'] = json.loads(data['user_context'])
        data['operation_result'] = json.loads(data['operation_result'])
        data['performance_metrics'] = json.loads(data['performance_metrics'])
        data['compliance_flags'] = json.loads(data['compliance_flags'])
        
        # Convert timestamp
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Remove database-only fields
        data.pop('created_at', None)
        
        return AuditEntry.from_dict(data)
    
    def _start_persistence_thread(self) -> None:
        """Start background thread for persisting memory buffer."""
        if self._persistence_thread and self._persistence_thread.is_alive():
            return
        
        def persistence_worker():
            while True:
                try:
                    time.sleep(5)  # Persist every 5 seconds
                    self._persist_buffer()
                except Exception as e:
                    logger.error(f"Persistence thread error: {e}")
        
        self._persistence_thread = threading.Thread(target=persistence_worker, daemon=True)
        self._persistence_thread.start()
        logger.debug("Audit persistence thread started")
    
    def _persist_buffer(self) -> None:
        """Persist memory buffer to database."""
        if not self._enable_buffer or not self.db_path or not self._memory_buffer:
            return
        
        try:
            # Get entries to persist (snapshot to avoid locking buffer too long)
            entries_to_persist = list(self._memory_buffer)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                for entry in entries_to_persist:
                    # Check if entry already exists
                    cursor = conn.execute("""
                        SELECT 1 FROM audit_entries WHERE entry_id = ?
                    """, (entry.entry_id,))
                    
                    if not cursor.fetchone():
                        self._store_entry_to_database_conn(conn, entry)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist buffer: {e}")
    
    def _store_entry_to_database_conn(self, conn: sqlite3.Connection, entry: AuditEntry) -> None:
        """Store entry to database using existing connection."""
        # Insert audit entry
        conn.execute("""
            INSERT INTO audit_entries (
                entry_id, timestamp, operation, component, memory_id,
                user_context, operation_result, governance_status,
                performance_metrics, security_level, compliance_flags,
                previous_hash, entry_hash, salt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.timestamp.isoformat(),
            entry.operation,
            entry.component,
            entry.memory_id,
            json.dumps(entry.user_context),
            json.dumps(entry.operation_result),
            entry.governance_status,
            json.dumps(entry.performance_metrics),
            entry.security_level,
            json.dumps(entry.compliance_flags),
            entry.previous_hash,
            entry.entry_hash,
            entry.salt
        ))
        
        # Update hash chain table
        with self._chain_lock:
            chain_position = len(self._hash_chain) - 1
            if entry.entry_hash in self._hash_chain:
                chain_position = self._hash_chain.index(entry.entry_hash)
        
        verification_key_hash = hashlib.sha256(self._verification_key.encode()).hexdigest()
        
        conn.execute("""
            INSERT OR IGNORE INTO hash_chain (
                chain_position, entry_id, entry_hash, timestamp, verification_key_hash
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            chain_position,
            entry.entry_id,
            entry.entry_hash,
            entry.timestamp.timestamp(),
            verification_key_hash
        ))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit logging performance metrics."""
        with self._chain_lock:
            return {
                **self._metrics,
                'hash_chain_length': len(self._hash_chain),
                'buffer_size': len(self._memory_buffer) if self._enable_buffer else 0,
                'buffer_enabled': self._enable_buffer,
                'database_enabled': self.db_path is not None
            }
    
    def cleanup_old_entries(self, older_than_days: Optional[int] = None) -> int:
        """
        Clean up old audit entries based on retention policy.
        
        Args:
            older_than_days: Override default retention period
            
        Returns:
            Number of entries cleaned up
        """
        retention_days = older_than_days or self.retention_days
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        cleaned_count = 0
        
        try:
            if self.db_path:
                with sqlite3.connect(str(self.db_path)) as conn:
                    # Clean old entries
                    cursor = conn.execute("""
                        DELETE FROM audit_entries 
                        WHERE created_at < julianday('now', '-{} days')
                    """.format(retention_days))
                    
                    cleaned_count = cursor.rowcount
                    conn.commit()
            
            # Clean memory buffer
            if self._enable_buffer and self._memory_buffer:
                initial_size = len(self._memory_buffer)
                cutoff_datetime = datetime.fromtimestamp(cutoff_time, timezone.utc)
                
                # Filter buffer, keeping only recent entries
                recent_entries = [
                    entry for entry in self._memory_buffer 
                    if entry.timestamp >= cutoff_datetime
                ]
                
                self._memory_buffer.clear()
                self._memory_buffer.extend(recent_entries)
                
                buffer_cleaned = initial_size - len(self._memory_buffer)
                cleaned_count += buffer_cleaned
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old audit entries (>{retention_days}d)")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")
            return 0