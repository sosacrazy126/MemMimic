"""
Audit Trail Manager - Comprehensive audit trail management and querying.

Provides high-level management interface for audit trails with advanced
querying capabilities, reporting, and integration with other audit components.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from ..errors import get_error_logger, handle_errors
from ..errors.exceptions import MemoryStorageError
from .immutable_logger import ImmutableAuditLogger, AuditEntry
from .cryptographic_verifier import CryptographicVerifier, HashChainVerifier
from .tamper_detector import TamperDetector, TamperAlert, TamperSeverity

logger = get_error_logger(__name__)


class QueryOperation(Enum):
    """Query operation types."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass
class AuditQueryFilter:
    """Filter condition for audit queries."""
    field: str
    operation: QueryOperation
    value: Any
    case_sensitive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary."""
        return {
            'field': self.field,
            'operation': self.operation.value,
            'value': self.value,
            'case_sensitive': self.case_sensitive
        }


@dataclass
class AuditQuery:
    """Comprehensive audit query specification."""
    
    # Basic filtering
    filters: List[AuditQueryFilter] = field(default_factory=list)
    
    # Time range filtering
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Specific entry filtering
    entry_ids: Optional[List[str]] = None
    memory_ids: Optional[List[str]] = None
    operations: Optional[List[str]] = None
    components: Optional[List[str]] = None
    
    # Result options
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"
    order_direction: str = "DESC"  # ASC or DESC
    
    # Include options
    include_metadata: bool = True
    include_evidence: bool = True
    include_verification_status: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            'filters': [f.to_dict() for f in self.filters],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'entry_ids': self.entry_ids,
            'memory_ids': self.memory_ids,
            'operations': self.operations,
            'components': self.components,
            'limit': self.limit,
            'offset': self.offset,
            'order_by': self.order_by,
            'order_direction': self.order_direction,
            'include_metadata': self.include_metadata,
            'include_evidence': self.include_evidence,
            'include_verification_status': self.include_verification_status
        }


@dataclass
class AuditQueryResult:
    """Result of audit query execution."""
    
    # Query results
    entries: List[AuditEntry]
    total_count: int
    query_time_ms: float
    
    # Query information
    query: AuditQuery
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Verification status (if requested)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    
    # Pagination info
    has_more: bool = False
    next_offset: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'entries': [entry.to_dict() for entry in self.entries],
            'total_count': self.total_count,
            'query_time_ms': self.query_time_ms,
            'query': self.query.to_dict(),
            'executed_at': self.executed_at.isoformat(),
            'verification_results': self.verification_results,
            'has_more': self.has_more,
            'next_offset': self.next_offset
        }


@dataclass
class AuditReport:
    """Comprehensive audit report."""
    
    # Report metadata
    report_id: str
    title: str
    description: str
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    # Generated timestamp
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Summary statistics
    total_entries: int = 0
    operations_summary: Dict[str, int] = field(default_factory=dict)
    components_summary: Dict[str, int] = field(default_factory=dict)
    
    # Security analysis
    tamper_alerts: List[TamperAlert] = field(default_factory=list)
    integrity_status: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance information
    compliance_status: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'report_id': self.report_id,
            'title': self.title,
            'description': self.description,
            'generated_at': self.generated_at.isoformat(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_entries': self.total_entries,
            'operations_summary': self.operations_summary,
            'components_summary': self.components_summary,
            'tamper_alerts': [alert.to_dict() for alert in self.tamper_alerts],
            'integrity_status': self.integrity_status,
            'performance_summary': self.performance_summary,
            'compliance_status': self.compliance_status
        }


class AuditTrailManager:
    """
    Comprehensive audit trail management system.
    
    Features:
    - Advanced audit trail querying
    - Automated report generation
    - Integration with tamper detection
    - Performance analytics
    - Compliance reporting
    """
    
    def __init__(
        self,
        audit_logger: ImmutableAuditLogger,
        crypto_verifier: CryptographicVerifier,
        hash_chain_verifier: HashChainVerifier,
        tamper_detector: TamperDetector,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audit trail manager.
        
        Args:
            audit_logger: Immutable audit logger instance
            crypto_verifier: Cryptographic verifier instance
            hash_chain_verifier: Hash chain verifier instance
            tamper_detector: Tamper detector instance
            config: Configuration options
        """
        self.audit_logger = audit_logger
        self.crypto_verifier = crypto_verifier
        self.hash_chain_verifier = hash_chain_verifier
        self.tamper_detector = tamper_detector
        self.config = config or {}
        
        # Query performance tracking
        self._query_metrics = {
            'total_queries': 0,
            'avg_query_time_ms': 0.0,
            'complex_queries': 0,
            'cache_hits': 0
        }
        
        # Simple query result cache
        self._query_cache = {}
        self._cache_ttl_seconds = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
        
        logger.info("AuditTrailManager initialized")
    
    @handle_errors(MemoryStorageError)
    def query_audit_trail(self, query: AuditQuery) -> AuditQueryResult:
        """
        Execute comprehensive audit trail query.
        
        Args:
            query: Audit query specification
            
        Returns:
            AuditQueryResult with matching entries and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._query_metrics['cache_hits'] += 1
                logger.debug(f"Query cache hit: {cache_key}")
                return cached_result
            
            # Execute query
            entries, total_count = self._execute_query(query)
            
            # Add verification status if requested
            verification_results = {}
            if query.include_verification_status:
                verification_results = self._verify_query_results(entries)
            
            # Calculate pagination info
            has_more = False
            next_offset = None
            if query.limit and len(entries) == query.limit:
                has_more = total_count > query.offset + query.limit
                if has_more:
                    next_offset = query.offset + query.limit
            
            query_time = (time.perf_counter() - start_time) * 1000
            
            # Create result
            result = AuditQueryResult(
                entries=entries,
                total_count=total_count,
                query_time_ms=query_time,
                query=query,
                verification_results=verification_results,
                has_more=has_more,
                next_offset=next_offset
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update metrics
            self._update_query_metrics(query_time, len(query.filters) > 3)
            
            logger.debug(
                f"Audit query executed: {len(entries)} entries, "
                f"{query_time:.2f}ms, total={total_count}"
            )
            
            return result
            
        except Exception as e:
            query_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Audit query failed: {e}")
            raise MemoryStorageError(f"Audit query execution failed: {e}") from e
    
    def _execute_query(self, query: AuditQuery) -> Tuple[List[AuditEntry], int]:
        """Execute the actual database query."""
        if not self.audit_logger.db_path:
            return self._execute_memory_query(query)
        
        try:
            with sqlite3.connect(str(self.audit_logger.db_path)) as conn:
                # Build SQL query
                sql_query, params = self._build_sql_query(query)
                
                # Execute count query for total
                count_sql = sql_query.replace("SELECT *", "SELECT COUNT(*)", 1)
                cursor = conn.execute(count_sql, params)
                total_count = cursor.fetchone()[0]
                
                # Execute main query with pagination
                if query.limit:
                    sql_query += f" LIMIT {query.limit}"
                if query.offset > 0:
                    sql_query += f" OFFSET {query.offset}"
                
                cursor = conn.execute(sql_query, params)
                rows = cursor.fetchall()
                
                # Convert rows to AuditEntry objects
                entries = []
                for row in rows:
                    entry = self._row_to_audit_entry(row, cursor.description)
                    entries.append(entry)
                
                return entries, total_count
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def _execute_memory_query(self, query: AuditQuery) -> Tuple[List[AuditEntry], int]:
        """Execute query against memory buffer."""
        if not hasattr(self.audit_logger, '_memory_buffer'):
            return [], 0
        
        # Get all entries from memory buffer
        all_entries = list(self.audit_logger._memory_buffer)
        
        # Apply filters
        filtered_entries = []
        for entry in all_entries:
            if self._entry_matches_query(entry, query):
                filtered_entries.append(entry)
        
        total_count = len(filtered_entries)
        
        # Apply sorting
        reverse = query.order_direction.upper() == "DESC"
        if query.order_by == "timestamp":
            filtered_entries.sort(key=lambda x: x.timestamp, reverse=reverse)
        else:
            # Add more sorting options as needed
            filtered_entries.sort(key=lambda x: getattr(x, query.order_by, ''), reverse=reverse)
        
        # Apply pagination
        start_idx = query.offset
        end_idx = start_idx + query.limit if query.limit else len(filtered_entries)
        entries = filtered_entries[start_idx:end_idx]
        
        return entries, total_count
    
    def _build_sql_query(self, query: AuditQuery) -> Tuple[str, List[Any]]:
        """Build SQL query from AuditQuery specification."""
        sql_parts = ["SELECT * FROM audit_entries"]
        where_conditions = []
        params = []
        
        # Time range filtering
        if query.start_time:
            where_conditions.append("timestamp >= ?")
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            where_conditions.append("timestamp <= ?")
            params.append(query.end_time.isoformat())
        
        # Specific field filtering
        if query.entry_ids:
            placeholders = ','.join(['?' for _ in query.entry_ids])
            where_conditions.append(f"entry_id IN ({placeholders})")
            params.extend(query.entry_ids)
        
        if query.memory_ids:
            placeholders = ','.join(['?' for _ in query.memory_ids])
            where_conditions.append(f"memory_id IN ({placeholders})")
            params.extend(query.memory_ids)
        
        if query.operations:
            placeholders = ','.join(['?' for _ in query.operations])
            where_conditions.append(f"operation IN ({placeholders})")
            params.extend(query.operations)
        
        if query.components:
            placeholders = ','.join(['?' for _ in query.components])
            where_conditions.append(f"component IN ({placeholders})")
            params.extend(query.components)
        
        # Custom filters
        for filter_obj in query.filters:
            condition, filter_params = self._build_filter_condition(filter_obj)
            if condition:
                where_conditions.append(condition)
                params.extend(filter_params)
        
        # Combine WHERE conditions
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add ORDER BY
        sql_parts.append(f"ORDER BY {query.order_by} {query.order_direction}")
        
        return " ".join(sql_parts), params
    
    def _build_filter_condition(self, filter_obj: AuditQueryFilter) -> Tuple[str, List[Any]]:
        """Build SQL condition from filter specification."""
        field = filter_obj.field
        op = filter_obj.operation
        value = filter_obj.value
        
        # Handle case sensitivity
        if isinstance(value, str) and not filter_obj.case_sensitive:
            field = f"LOWER({field})"
            if isinstance(value, str):
                value = value.lower()
        
        if op == QueryOperation.EQUALS:
            return f"{field} = ?", [value]
        elif op == QueryOperation.NOT_EQUALS:
            return f"{field} != ?", [value]
        elif op == QueryOperation.CONTAINS:
            return f"{field} LIKE ?", [f"%{value}%"]
        elif op == QueryOperation.STARTS_WITH:
            return f"{field} LIKE ?", [f"{value}%"]
        elif op == QueryOperation.ENDS_WITH:
            return f"{field} LIKE ?", [f"%{value}"]
        elif op == QueryOperation.GREATER_THAN:
            return f"{field} > ?", [value]
        elif op == QueryOperation.LESS_THAN:
            return f"{field} < ?", [value]
        elif op == QueryOperation.GREATER_EQUAL:
            return f"{field} >= ?", [value]
        elif op == QueryOperation.LESS_EQUAL:
            return f"{field} <= ?", [value]
        elif op == QueryOperation.IN:
            if isinstance(value, (list, tuple)):
                placeholders = ','.join(['?' for _ in value])
                return f"{field} IN ({placeholders})", list(value)
        elif op == QueryOperation.NOT_IN:
            if isinstance(value, (list, tuple)):
                placeholders = ','.join(['?' for _ in value])
                return f"{field} NOT IN ({placeholders})", list(value)
        elif op == QueryOperation.BETWEEN:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return f"{field} BETWEEN ? AND ?", list(value)
        elif op == QueryOperation.IS_NULL:
            return f"{field} IS NULL", []
        elif op == QueryOperation.IS_NOT_NULL:
            return f"{field} IS NOT NULL", []
        
        return "", []
    
    def _entry_matches_query(self, entry: AuditEntry, query: AuditQuery) -> bool:
        """Check if entry matches query criteria (for memory queries)."""
        # Time range check
        if query.start_time and entry.timestamp < query.start_time:
            return False
        if query.end_time and entry.timestamp > query.end_time:
            return False
        
        # Specific field checks
        if query.entry_ids and entry.entry_id not in query.entry_ids:
            return False
        if query.memory_ids and entry.memory_id not in query.memory_ids:
            return False
        if query.operations and entry.operation not in query.operations:
            return False
        if query.components and entry.component not in query.components:
            return False
        
        # Custom filters
        for filter_obj in query.filters:
            if not self._entry_matches_filter(entry, filter_obj):
                return False
        
        return True
    
    def _entry_matches_filter(self, entry: AuditEntry, filter_obj: AuditQueryFilter) -> bool:
        """Check if entry matches a specific filter."""
        field_value = getattr(entry, filter_obj.field, None)
        
        if field_value is None:
            return filter_obj.operation == QueryOperation.IS_NULL
        
        if filter_obj.operation == QueryOperation.IS_NULL:
            return False
        elif filter_obj.operation == QueryOperation.IS_NOT_NULL:
            return True
        
        # Handle string operations with case sensitivity
        if isinstance(field_value, str) and isinstance(filter_obj.value, str):
            if not filter_obj.case_sensitive:
                field_value = field_value.lower()
                compare_value = filter_obj.value.lower()
            else:
                compare_value = filter_obj.value
        else:
            compare_value = filter_obj.value
        
        # Apply operation
        if filter_obj.operation == QueryOperation.EQUALS:
            return field_value == compare_value
        elif filter_obj.operation == QueryOperation.NOT_EQUALS:
            return field_value != compare_value
        elif filter_obj.operation == QueryOperation.CONTAINS:
            return compare_value in field_value
        elif filter_obj.operation == QueryOperation.STARTS_WITH:
            return field_value.startswith(compare_value)
        elif filter_obj.operation == QueryOperation.ENDS_WITH:
            return field_value.endswith(compare_value)
        elif filter_obj.operation == QueryOperation.IN:
            return field_value in compare_value
        elif filter_obj.operation == QueryOperation.NOT_IN:
            return field_value not in compare_value
        # Add numeric comparisons as needed
        
        return False
    
    def _row_to_audit_entry(self, row: tuple, description: List[tuple]) -> AuditEntry:
        """Convert database row to AuditEntry object."""
        # Create dict from row data
        columns = [col[0] for col in description]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        json_fields = ['user_context', 'operation_result', 'performance_metrics', 'compliance_flags']
        for field in json_fields:
            if field in data and data[field]:
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    data[field] = {}
        
        # Convert timestamp
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Remove database-only fields
        data.pop('created_at', None)
        
        return AuditEntry.from_dict(data)
    
    def _verify_query_results(self, entries: List[AuditEntry]) -> Dict[str, Any]:
        """Verify integrity of query results."""
        verification_results = {
            'total_entries': len(entries),
            'verified_entries': 0,
            'failed_verifications': 0,
            'tamper_detected': False,
            'details': []
        }
        
        for entry in entries:
            try:
                # Verify individual entry
                is_valid = self.audit_logger.verify_entry_integrity(entry.entry_id)
                
                if is_valid:
                    verification_results['verified_entries'] += 1
                else:
                    verification_results['failed_verifications'] += 1
                    verification_results['tamper_detected'] = True
                    verification_results['details'].append({
                        'entry_id': entry.entry_id,
                        'status': 'failed',
                        'reason': 'Hash verification failed'
                    })
            
            except Exception as e:
                verification_results['failed_verifications'] += 1
                verification_results['details'].append({
                    'entry_id': entry.entry_id,
                    'status': 'error',
                    'reason': str(e)
                })
        
        return verification_results
    
    def generate_audit_report(
        self,
        start_time: datetime,
        end_time: datetime,
        title: str = "Audit Trail Report",
        include_performance: bool = True,
        include_security: bool = True,
        include_compliance: bool = True
    ) -> AuditReport:
        """
        Generate comprehensive audit report for specified time period.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            title: Report title
            include_performance: Include performance analysis
            include_security: Include security analysis
            include_compliance: Include compliance analysis
            
        Returns:
            AuditReport with comprehensive analysis
        """
        report_id = f"audit_report_{int(time.time())}"
        
        # Query audit entries for time period
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            order_by="timestamp",
            order_direction="ASC"
        )
        
        result = self.query_audit_trail(query)
        
        # Generate report
        report = AuditReport(
            report_id=report_id,
            title=title,
            description=f"Audit trail analysis from {start_time.isoformat()} to {end_time.isoformat()}",
            start_time=start_time,
            end_time=end_time,
            total_entries=result.total_count
        )
        
        # Analyze entries
        operations_count = {}
        components_count = {}
        
        for entry in result.entries:
            # Count operations
            operations_count[entry.operation] = operations_count.get(entry.operation, 0) + 1
            
            # Count components
            components_count[entry.component] = components_count.get(entry.component, 0) + 1
        
        report.operations_summary = operations_count
        report.components_summary = components_count
        
        # Security analysis
        if include_security:
            report.tamper_alerts = self.tamper_detector.get_recent_alerts(
                hours=int((end_time - start_time).total_seconds() / 3600)
            )
            
            # Verify hash chain integrity
            chain_verification = self.audit_logger.verify_hash_chain_integrity()
            report.integrity_status = {
                'hash_chain_valid': chain_verification['valid'],
                'entries_verified': chain_verification['entries_verified'],
                'broken_links': len(chain_verification['broken_links']),
                'verification_time_ms': chain_verification['verification_time_ms']
            }
        
        # Performance analysis
        if include_performance:
            logger_metrics = self.audit_logger.get_metrics()
            crypto_metrics = self.crypto_verifier.get_metrics()
            detector_metrics = self.tamper_detector.get_metrics()
            
            report.performance_summary = {
                'audit_logger': logger_metrics,
                'crypto_verifier': crypto_metrics,
                'tamper_detector': detector_metrics,
                'query_metrics': self._query_metrics
            }
        
        # Compliance analysis
        if include_compliance:
            report.compliance_status = self._analyze_compliance(result.entries)
        
        logger.info(f"Audit report generated: {report_id} ({result.total_count} entries)")
        return report
    
    def _analyze_compliance(self, entries: List[AuditEntry]) -> Dict[str, Any]:
        """Analyze entries for compliance requirements."""
        compliance_status = {
            'total_entries': len(entries),
            'compliance_flags_summary': {},
            'security_levels': {},
            'governance_status_summary': {},
            'retention_compliance': True
        }
        
        for entry in entries:
            # Analyze compliance flags
            for flag in entry.compliance_flags:
                compliance_status['compliance_flags_summary'][flag] = \
                    compliance_status['compliance_flags_summary'].get(flag, 0) + 1
            
            # Analyze security levels
            level = entry.security_level
            compliance_status['security_levels'][level] = \
                compliance_status['security_levels'].get(level, 0) + 1
            
            # Analyze governance status
            if entry.governance_status:
                compliance_status['governance_status_summary'][entry.governance_status] = \
                    compliance_status['governance_status_summary'].get(entry.governance_status, 0) + 1
        
        return compliance_status
    
    def _generate_cache_key(self, query: AuditQuery) -> str:
        """Generate cache key for query."""
        # Create a hash of query parameters
        import hashlib
        query_str = json.dumps(query.to_dict(), sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]
    
    def _get_cached_result(self, cache_key: str) -> Optional[AuditQueryResult]:
        """Get cached query result if still valid."""
        if cache_key in self._query_cache:
            cached_data, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_data
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: AuditQueryResult) -> None:
        """Cache query result."""
        # Limit cache size
        if len(self._query_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (result, time.time())
    
    def _update_query_metrics(self, query_time: float, is_complex: bool) -> None:
        """Update query performance metrics."""
        self._query_metrics['total_queries'] += 1
        
        if is_complex:
            self._query_metrics['complex_queries'] += 1
        
        # Update average query time
        current_avg = self._query_metrics['avg_query_time_ms']
        count = self._query_metrics['total_queries']
        self._query_metrics['avg_query_time_ms'] = ((current_avg * (count - 1)) + query_time) / count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit trail manager metrics."""
        return {
            'query_metrics': self._query_metrics,
            'cache_size': len(self._query_cache),
            'cache_ttl_seconds': self._cache_ttl_seconds,
            'audit_logger_metrics': self.audit_logger.get_metrics(),
            'crypto_verifier_metrics': self.crypto_verifier.get_metrics(),
            'tamper_detector_metrics': self.tamper_detector.get_metrics()
        }