"""
Security Audit and Logging System

Comprehensive security event logging, monitoring, and analysis.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    INPUT_VALIDATION = "input_validation"
    VALIDATION_FAILURE = "validation_failure"
    OUTPUT_SANITIZATION = "output_sanitization"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    MALFORMED_REQUEST = "malformed_request"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCESS_DENIED = "access_denied"
    FUNCTION_CALL = "function_call"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_ERROR = "system_error"


class SeverityLevel(Enum):
    """Security event severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: Union[SecurityEventType, str]
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    function_name: Optional[str] = None
    severity: Union[SeverityLevel, str] = SeverityLevel.LOW
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        """Process event after initialization."""
        # Ensure event_type is enum if string provided
        if isinstance(self.event_type, str):
            try:
                self.event_type = SecurityEventType(self.event_type)
            except ValueError:
                logger.warning(f"Unknown security event type: {self.event_type}")
        
        # Ensure severity is enum if string provided
        if isinstance(self.severity, str):
            try:
                self.severity = SeverityLevel(self.severity.upper())
            except ValueError:
                logger.warning(f"Unknown severity level: {self.severity}")
                self.severity = SeverityLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type.value if isinstance(self.event_type, SecurityEventType) else str(self.event_type),
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'component': self.component,
            'function_name': self.function_name,
            'severity': self.severity.value if isinstance(self.severity, SeverityLevel) else str(self.severity),
            'details': self.details,
            'metadata': self.metadata,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create SecurityEvent from dictionary."""
        return cls(
            event_type=data['event_type'],
            timestamp=data['timestamp'],
            component=data.get('component', 'unknown'),
            function_name=data.get('function_name'),
            severity=data.get('severity', 'LOW'),
            details=data.get('details', ''),
            metadata=data.get('metadata', {}),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            request_id=data.get('request_id')
        )


@dataclass
class SecurityMetrics:
    """Security metrics and statistics."""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_component: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_events: int = 0  # Events in last hour
    high_severity_events: int = 0
    blocked_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    
    def update_with_event(self, event: SecurityEvent) -> None:
        """Update metrics with new event."""
        self.total_events += 1
        
        event_type_str = event.event_type.value if isinstance(event.event_type, SecurityEventType) else str(event.event_type)
        severity_str = event.severity.value if isinstance(event.severity, SeverityLevel) else str(event.severity)
        
        self.events_by_type[event_type_str] += 1
        self.events_by_severity[severity_str] += 1
        self.events_by_component[event.component] += 1
        
        # Count recent events (last hour)
        if event.timestamp > time.time() - 3600:
            self.recent_events += 1
        
        # Count high severity events
        if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            self.high_severity_events += 1
        
        # Count blocked attempts
        if event.event_type in [
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventType.XSS_ATTEMPT,
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            SecurityEventType.RATE_LIMIT_EXCEEDED
        ]:
            self.blocked_attempts += 1


class SecurityAuditLogger:
    """
    Security audit logging system with event storage and analysis.
    
    Features:
    - Event logging with structured data
    - Metrics collection and analysis
    - Event filtering and querying
    - Automatic log rotation
    - Real-time threat detection
    """
    
    def __init__(self, log_file: Optional[str] = None, max_events_memory: int = 10000):
        """
        Initialize security audit logger.
        
        Args:
            log_file: Path to log file for persistent storage
            max_events_memory: Maximum events to keep in memory
        """
        self.log_file = Path(log_file) if log_file else None
        self.max_events_memory = max_events_memory
        
        # In-memory event storage (limited size for performance)
        self._events: deque = deque(maxlen=max_events_memory)
        self._metrics = SecurityMetrics()
        self._event_lock = threading.Lock()
        
        # Threat detection patterns
        self._threat_patterns = self._initialize_threat_patterns()
        
        # Setup file logging if specified
        if self.log_file:
            self._setup_file_logging()
        
        logger.info(f"SecurityAuditLogger initialized (log_file={log_file})")
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection patterns."""
        return {
            'repeated_failures': {
                'event_types': ['validation_failure', 'rate_limit_exceeded'],
                'threshold': 10,
                'window_seconds': 300,  # 5 minutes
                'severity': SeverityLevel.HIGH
            },
            'injection_attempts': {
                'event_types': ['sql_injection_attempt', 'xss_attempt', 'command_injection_attempt'],
                'threshold': 3,
                'window_seconds': 60,
                'severity': SeverityLevel.CRITICAL
            },
            'path_traversal_attempts': {
                'event_types': ['path_traversal_attempt'],
                'threshold': 5,
                'window_seconds': 300,
                'severity': SeverityLevel.HIGH
            }
        }
    
    def _setup_file_logging(self) -> None:
        """Setup file logging configuration."""
        try:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup file handler for security events
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            
            # JSON formatter for structured logs
            class SecurityFormatter(logging.Formatter):
                def format(self, record):
                    if hasattr(record, 'security_event'):
                        return json.dumps(record.security_event.to_dict(), ensure_ascii=False)
                    else:
                        return super().format(record)
            
            file_handler.setFormatter(SecurityFormatter())
            
            # Add handler to security logger
            security_logger = logging.getLogger('security_audit')
            security_logger.addHandler(file_handler)
            security_logger.setLevel(logging.INFO)
            
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """
        Log a security event.
        
        Args:
            event: SecurityEvent to log
        """
        with self._event_lock:
            # Add to in-memory storage
            self._events.append(event)
            
            # Update metrics
            self._metrics.update_with_event(event)
            
            # Log to file if configured
            if self.log_file:
                security_logger = logging.getLogger('security_audit')
                log_record = logging.LogRecord(
                    name='security_audit',
                    level=self._severity_to_log_level(event.severity),
                    pathname='',
                    lineno=0,
                    msg=f"Security event: {event.event_type.value if isinstance(event.event_type, SecurityEventType) else event.event_type}",
                    args=(),
                    exc_info=None
                )
                log_record.security_event = event
                security_logger.handle(log_record)
            
            # Check for threat patterns
            self._check_threat_patterns(event)
            
            # Log to application logger based on severity
            if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                logger.warning(f"Security event: {event.event_type.value if isinstance(event.event_type, SecurityEventType) else event.event_type} - {event.details}")
            elif event.severity == SeverityLevel.MEDIUM:
                logger.info(f"Security event: {event.event_type.value if isinstance(event.event_type, SecurityEventType) else event.event_type}")
    
    def _severity_to_log_level(self, severity: Union[SeverityLevel, str]) -> int:
        """Convert severity to logging level."""
        if isinstance(severity, str):
            severity = SeverityLevel(severity.upper())
        
        mapping = {
            SeverityLevel.LOW: logging.INFO,
            SeverityLevel.MEDIUM: logging.WARNING,
            SeverityLevel.HIGH: logging.ERROR,
            SeverityLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)
    
    def _check_threat_patterns(self, event: SecurityEvent) -> None:
        """Check event against threat detection patterns."""
        current_time = event.timestamp
        
        for pattern_name, pattern in self._threat_patterns.items():
            # Get events of relevant types within time window
            relevant_events = [
                e for e in self._events
                if (isinstance(e.event_type, SecurityEventType) and e.event_type.value in pattern['event_types']
                    or isinstance(e.event_type, str) and e.event_type in pattern['event_types'])
                and current_time - e.timestamp <= pattern['window_seconds']
            ]
            
            # Check if threshold exceeded
            if len(relevant_events) >= pattern['threshold']:
                threat_event = SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    component="threat_detection",
                    severity=pattern['severity'],
                    details=f"Threat pattern detected: {pattern_name}",
                    metadata={
                        'pattern_name': pattern_name,
                        'event_count': len(relevant_events),
                        'threshold': pattern['threshold'],
                        'window_seconds': pattern['window_seconds'],
                        'triggering_event': event.to_dict()
                    }
                )
                
                # Log threat detection (avoid infinite recursion by not checking patterns again)
                with self._event_lock:
                    self._events.append(threat_event)
                    self._metrics.update_with_event(threat_event)
                
                logger.critical(f"THREAT DETECTED: {pattern_name} - {len(relevant_events)} events in {pattern['window_seconds']} seconds")
    
    def get_events(self, event_type: Optional[Union[SecurityEventType, str]] = None,
                   severity: Optional[Union[SeverityLevel, str]] = None,
                   component: Optional[str] = None,
                   since: Optional[float] = None,
                   limit: Optional[int] = None) -> List[SecurityEvent]:
        """
        Query security events with filters.
        
        Args:
            event_type: Filter by event type
            severity: Filter by severity level
            component: Filter by component
            since: Filter events since timestamp
            limit: Maximum number of events to return
            
        Returns:
            List of matching SecurityEvent objects
        """
        with self._event_lock:
            events = list(self._events)
        
        # Apply filters
        if event_type:
            if isinstance(event_type, str):
                events = [e for e in events 
                         if (isinstance(e.event_type, SecurityEventType) and e.event_type.value == event_type)
                         or (isinstance(e.event_type, str) and e.event_type == event_type)]
            else:
                events = [e for e in events if e.event_type == event_type]
        
        if severity:
            if isinstance(severity, str):
                events = [e for e in events 
                         if (isinstance(e.severity, SeverityLevel) and e.severity.value == severity.upper())
                         or (isinstance(e.severity, str) and e.severity.upper() == severity.upper())]
            else:
                events = [e for e in events if e.severity == severity]
        
        if component:
            events = [e for e in events if e.component == component]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        with self._event_lock:
            return self._metrics
    
    def get_recent_activity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent security activity."""
        since = time.time() - (hours * 3600)
        recent_events = self.get_events(since=since)
        
        # Analyze recent events
        event_types = defaultdict(int)
        severities = defaultdict(int)
        components = defaultdict(int)
        hourly_counts = defaultdict(int)
        
        for event in recent_events:
            event_type_str = event.event_type.value if isinstance(event.event_type, SecurityEventType) else str(event.event_type)
            severity_str = event.severity.value if isinstance(event.severity, SeverityLevel) else str(event.severity)
            
            event_types[event_type_str] += 1
            severities[severity_str] += 1
            components[event.component] += 1
            
            # Group by hour
            hour_key = int(event.timestamp // 3600)
            hourly_counts[hour_key] += 1
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_types': dict(event_types),
            'severities': dict(severities),
            'components': dict(components),
            'hourly_distribution': dict(hourly_counts),
            'high_severity_count': severities.get('HIGH', 0) + severities.get('CRITICAL', 0)
        }
    
    def export_events(self, file_path: str, event_filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Export events to file.
        
        Args:
            file_path: Path to export file
            event_filter: Optional filter criteria
        """
        events = self.get_events(**(event_filter or {}))
        
        export_path = Path(file_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if export_path.suffix.lower() == '.json':
                with open(export_path, 'w') as f:
                    json.dump([event.to_dict() for event in events], f, indent=2, ensure_ascii=False)
            else:
                # CSV export
                import csv
                with open(export_path, 'w', newline='') as f:
                    if events:
                        fieldnames = events[0].to_dict().keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for event in events:
                            writer.writerow(event.to_dict())
            
            logger.info(f"Exported {len(events)} security events to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export events to {export_path}: {e}")
    
    def clear_old_events(self, older_than_days: int = 30) -> int:
        """
        Clear events older than specified days.
        
        Args:
            older_than_days: Remove events older than this many days
            
        Returns:
            Number of events removed
        """
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        
        with self._event_lock:
            initial_count = len(self._events)
            # Filter events, keeping only recent ones
            self._events = deque(
                (event for event in self._events if event.timestamp >= cutoff_time),
                maxlen=self.max_events_memory
            )
            final_count = len(self._events)
            removed_count = initial_count - final_count
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} security events older than {older_than_days} days")
        
        return removed_count


# Global audit logger instance
_global_audit_logger: Optional[SecurityAuditLogger] = None


def get_security_audit_logger() -> SecurityAuditLogger:
    """Get the global security audit logger."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = SecurityAuditLogger()
    return _global_audit_logger


def initialize_security_audit_logger(log_file: Optional[str] = None, 
                                    max_events_memory: int = 10000) -> SecurityAuditLogger:
    """Initialize the global security audit logger."""
    global _global_audit_logger
    _global_audit_logger = SecurityAuditLogger(log_file, max_events_memory)
    return _global_audit_logger


def log_security_event(event_type: Union[SecurityEventType, str],
                      component: str = "unknown",
                      severity: Union[SeverityLevel, str] = SeverityLevel.LOW,
                      details: str = "",
                      **metadata) -> None:
    """
    Quick function to log a security event.
    
    Args:
        event_type: Type of security event
        component: Component that generated the event
        severity: Severity level of the event
        details: Detailed description
        **metadata: Additional metadata
    """
    event = SecurityEvent(
        event_type=event_type,
        component=component,
        severity=severity,
        details=details,
        metadata=metadata
    )
    
    audit_logger = get_security_audit_logger()
    audit_logger.log_security_event(event)