"""
Enterprise Security Monitoring System for MemMimic

Provides comprehensive security monitoring, threat detection, and incident response
for authentication, API abuse, data access patterns, and security events.
"""

import time
import threading
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import ipaddress

from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHENTICATION_SUCCESS = "auth_success"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS_VIOLATION = "data_access_violation"
    API_ABUSE = "api_abuse"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class SecuritySeverity(Enum):
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: SecurityEventType
    severity: SecuritySeverity
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'endpoint': self.endpoint,
            'request_data': self.request_data,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SecurityIncident:
    """Security incident aggregating related events"""
    incident_id: str
    incident_type: str
    severity: SecuritySeverity
    events: List[SecurityEvent] = field(default_factory=list)
    status: str = "open"  # open, investigating, resolved
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    source_ips: Set[str] = field(default_factory=set)
    affected_endpoints: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'incident_id': self.incident_id,
            'incident_type': self.incident_type,
            'severity': self.severity.value,
            'status': self.status,
            'event_count': len(self.events),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'source_ips': list(self.source_ips),
            'affected_endpoints': list(self.affected_endpoints),
            'events': [event.to_dict() for event in self.events[-10:]]  # Last 10 events
        }


class ThreatDetector:
    """Advanced threat detection engine"""
    
    def __init__(self):
        # Rate limiting tracking
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Failed authentication tracking
        self.auth_failures: Dict[str, List[datetime]] = defaultdict(list)
        
        # Suspicious patterns
        self.suspicious_ips: Set[str] = set()
        self.known_attack_patterns = self._load_attack_patterns()
        
        # User behavior baseline
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Threat detector initialized")
    
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load known attack patterns"""
        return {
            'sql_injection': [
                r"union\s+select", r"drop\s+table", r"insert\s+into",
                r"'.*or.*'.*=.*'", r"1=1", r"1'1"
            ],
            'xss_patterns': [
                r"<script", r"javascript:", r"onload=", r"onerror=",
                r"alert\(", r"document\.cookie"
            ],
            'command_injection': [
                r";\s*cat", r";\s*ls", r";\s*pwd", r"&&", r"\|",
                r"rm\s+-rf", r"wget", r"curl"
            ],
            'path_traversal': [
                r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%252e%252e%252f"
            ]
        }
    
    def detect_rate_limit_abuse(self, source_ip: str, endpoint: str, 
                              window_seconds: int = 60, max_requests: int = 100) -> bool:
        """Detect rate limit abuse"""
        current_time = time.time()
        key = f"{source_ip}:{endpoint}"
        
        # Clean old requests
        self.request_counts[key] = deque(
            [req_time for req_time in self.request_counts[key] 
             if current_time - req_time <= window_seconds],
            maxlen=1000
        )
        
        # Add current request
        self.request_counts[key].append(current_time)
        
        return len(self.request_counts[key]) > max_requests
    
    def detect_brute_force_attack(self, source_ip: str, max_failures: int = 5, 
                                window_minutes: int = 15) -> bool:
        """Detect brute force authentication attacks"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=window_minutes)
        
        # Clean old failures
        self.auth_failures[source_ip] = [
            failure_time for failure_time in self.auth_failures[source_ip]
            if failure_time >= cutoff_time
        ]
        
        return len(self.auth_failures[source_ip]) >= max_failures
    
    def detect_injection_attempt(self, input_data: Any) -> Optional[str]:
        """Detect injection attempts in input data"""
        if not input_data:
            return None
        
        # Convert to string for pattern matching
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data).lower()
        else:
            input_str = str(input_data).lower()
        
        import re
        
        for attack_type, patterns in self.known_attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return attack_type
        
        return None
    
    def detect_anomalous_behavior(self, user_id: str, endpoint: str, 
                                request_size: int, response_time: float) -> bool:
        """Detect anomalous user behavior patterns"""
        if user_id not in self.user_baselines:
            # Initialize baseline
            self.user_baselines[user_id] = {
                'endpoints': defaultdict(int),
                'avg_request_size': 0,
                'avg_response_time': 0,
                'request_count': 0
            }
            return False
        
        baseline = self.user_baselines[user_id]
        
        # Check for unusual endpoint access
        if endpoint not in baseline['endpoints'] and baseline['request_count'] > 50:
            return True
        
        # Check for unusual request size
        if baseline['avg_request_size'] > 0:
            size_deviation = abs(request_size - baseline['avg_request_size']) / baseline['avg_request_size']
            if size_deviation > 5.0:  # 500% deviation
                return True
        
        # Update baseline
        baseline['endpoints'][endpoint] += 1
        baseline['request_count'] += 1
        baseline['avg_request_size'] = (
            (baseline['avg_request_size'] * (baseline['request_count'] - 1) + request_size)
            / baseline['request_count']
        )
        baseline['avg_response_time'] = (
            (baseline['avg_response_time'] * (baseline['request_count'] - 1) + response_time)
            / baseline['request_count']
        )
        
        return False
    
    def record_auth_failure(self, source_ip: str):
        """Record authentication failure"""
        self.auth_failures[source_ip].append(datetime.now())
    
    def is_suspicious_ip(self, source_ip: str) -> bool:
        """Check if IP is flagged as suspicious"""
        try:
            ip_obj = ipaddress.ip_address(source_ip)
            
            # Check for private/local addresses (less suspicious for internal use)
            if ip_obj.is_private or ip_obj.is_loopback:
                return False
            
            # Check known suspicious IPs
            return source_ip in self.suspicious_ips
            
        except ValueError:
            return True  # Invalid IP format is suspicious


class SecurityMonitor:
    """
    Enterprise security monitoring system for MemMimic.
    
    Provides comprehensive security event monitoring, threat detection,
    and incident response capabilities.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        
        # Event storage
        self.security_events: deque = deque(maxlen=10000)
        self.security_incidents: Dict[str, SecurityIncident] = {}
        
        # Threat detection
        self.threat_detector = ThreatDetector()
        
        # Metrics integration
        self.metrics_collector = get_metrics_collector()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SecurityEvent], None]] = []
        self.incident_callbacks: List[Callable[[SecurityIncident], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._processing_thread = None
        self._stop_processing = threading.Event()
        
        # Security statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'incidents_created': 0,
            'incidents_resolved': 0
        }
        
        self._start_background_processing()
        
        logger.info("Security Monitor initialized")
    
    def record_authentication_failure(self, source_ip: str, user_agent: str = None,
                                    endpoint: str = None, reason: str = "Invalid credentials"):
        """Record authentication failure event"""
        self.threat_detector.record_auth_failure(source_ip)
        
        # Check for brute force attack
        if self.threat_detector.detect_brute_force_attack(source_ip):
            severity = SecuritySeverity.HIGH
            message = f"Brute force attack detected from {source_ip}"
        else:
            severity = SecuritySeverity.MEDIUM
            message = f"Authentication failure from {source_ip}: {reason}"
        
        event = SecurityEvent(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=severity,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            message=message,
            metadata={'reason': reason}
        )
        
        self._record_event(event)
    
    def record_authentication_success(self, source_ip: str, user_id: str = None,
                                    user_agent: str = None, endpoint: str = None):
        """Record successful authentication event"""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            severity=SecuritySeverity.LOW,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            message=f"Successful authentication from {source_ip}",
            metadata={'user_id': user_id}
        )
        
        self._record_event(event)
    
    def record_rate_limit_exceeded(self, source_ip: str, endpoint: str, 
                                 request_count: int, limit: int, user_agent: str = None):
        """Record rate limit exceeded event"""
        severity = SecuritySeverity.HIGH if request_count > limit * 2 else SecuritySeverity.MEDIUM
        
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=severity,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            message=f"Rate limit exceeded from {source_ip}: {request_count}/{limit} requests",
            metadata={'request_count': request_count, 'limit': limit}
        )
        
        self._record_event(event)
    
    def record_api_request(self, source_ip: str, endpoint: str, request_data: Any = None,
                         user_agent: str = None, user_id: str = None, 
                         response_time: float = 0.0, request_size: int = 0):
        """Record and analyze API request for security issues"""
        
        # Check for rate limiting abuse
        if self.threat_detector.detect_rate_limit_abuse(source_ip, endpoint):
            self.record_rate_limit_exceeded(source_ip, endpoint, 100, 100, user_agent)
        
        # Check for injection attempts
        injection_type = self.threat_detector.detect_injection_attempt(request_data)
        if injection_type:
            event = SecurityEvent(
                event_type=SecurityEventType.INJECTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
                source_ip=source_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                request_data=self._sanitize_request_data(request_data),
                message=f"Potential {injection_type} attempt from {source_ip}",
                metadata={'injection_type': injection_type}
            )
            self._record_event(event)
        
        # Check for anomalous behavior
        if user_id and self.threat_detector.detect_anomalous_behavior(
            user_id, endpoint, request_size, response_time):
            event = SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=SecuritySeverity.MEDIUM,
                source_ip=source_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                message=f"Anomalous behavior detected for user {user_id}",
                metadata={'user_id': user_id, 'request_size': request_size}
            )
            self._record_event(event)
        
        # Check for suspicious IP
        if self.threat_detector.is_suspicious_ip(source_ip):
            event = SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=SecuritySeverity.MEDIUM,
                source_ip=source_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                message=f"Request from suspicious IP {source_ip}",
                metadata={'reason': 'suspicious_ip'}
            )
            self._record_event(event)
    
    def record_data_access_violation(self, source_ip: str, endpoint: str, 
                                   violation_type: str, user_id: str = None,
                                   requested_resource: str = None):
        """Record data access violation"""
        event = SecurityEvent(
            event_type=SecurityEventType.DATA_ACCESS_VIOLATION,
            severity=SecuritySeverity.HIGH,
            source_ip=source_ip,
            endpoint=endpoint,
            message=f"Data access violation: {violation_type}",
            metadata={
                'violation_type': violation_type,
                'user_id': user_id,
                'requested_resource': requested_resource
            }
        )
        
        self._record_event(event)
    
    def _record_event(self, event: SecurityEvent):
        """Record security event and update metrics"""
        with self._lock:
            self.security_events.append(event)
            
            # Update statistics
            self.stats['total_events'] += 1
            self.stats['events_by_type'][event.event_type.value] += 1
            self.stats['events_by_severity'][event.severity.value] += 1
            
            # Update metrics
            self.metrics_collector.increment_counter(f"memmimic_security_events_total")
            self.metrics_collector.increment_counter(
                f"memmimic_security_events_{event.event_type.value}_total"
            )
            self.metrics_collector.increment_counter(
                f"memmimic_security_events_{event.severity.value}_total"
            )
            
            # Trigger alerts for high/critical events
            if event.severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
                self._trigger_alert(event)
            
            # Check for incident correlation
            self._correlate_incident(event)
        
        logger.info(f"Security event recorded: {event.event_type.value} from {event.source_ip}")
    
    def _trigger_alert(self, event: SecurityEvent):
        """Trigger security alert"""
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"SECURITY ALERT: {event.message}")
    
    def _correlate_incident(self, event: SecurityEvent):
        """Correlate events into security incidents"""
        incident_key = self._generate_incident_key(event)
        
        if incident_key in self.security_incidents:
            # Add to existing incident
            incident = self.security_incidents[incident_key]
            incident.events.append(event)
            incident.last_seen = event.timestamp
            
            if event.source_ip:
                incident.source_ips.add(event.source_ip)
            if event.endpoint:
                incident.affected_endpoints.add(event.endpoint)
            
            # Escalate severity if needed
            if event.severity.value == SecuritySeverity.CRITICAL.value:
                incident.severity = SecuritySeverity.CRITICAL
            elif (event.severity.value == SecuritySeverity.HIGH.value and 
                  incident.severity.value != SecuritySeverity.CRITICAL.value):
                incident.severity = SecuritySeverity.HIGH
        else:
            # Create new incident
            incident = SecurityIncident(
                incident_id=incident_key,
                incident_type=event.event_type.value,
                severity=event.severity,
                events=[event],
                source_ips={event.source_ip} if event.source_ip else set(),
                affected_endpoints={event.endpoint} if event.endpoint else set()
            )
            
            self.security_incidents[incident_key] = incident
            self.stats['incidents_created'] += 1
            
            # Trigger incident callbacks
            for callback in self.incident_callbacks:
                try:
                    callback(incident)
                except Exception as e:
                    logger.error(f"Incident callback failed: {e}")
    
    def _generate_incident_key(self, event: SecurityEvent) -> str:
        """Generate incident correlation key"""
        # Correlate by source IP and event type within time window
        base_key = f"{event.source_ip}_{event.event_type.value}"
        time_window = int(event.timestamp.timestamp() / 300)  # 5-minute windows
        return f"{base_key}_{time_window}"
    
    def _sanitize_request_data(self, request_data: Any) -> Dict[str, Any]:
        """Sanitize request data for logging"""
        if not request_data:
            return {}
        
        # Convert to dict and remove sensitive data
        if isinstance(request_data, dict):
            sanitized = {}
            sensitive_keys = {'password', 'token', 'secret', 'key', 'auth', 'credential'}
            
            for key, value in request_data.items():
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, str) and len(value) > 1000:
                    sanitized[key] = value[:500] + "... [TRUNCATED]"
                else:
                    sanitized[key] = value
            
            return sanitized
        else:
            return {'data': str(request_data)[:500]}
    
    def add_alert_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def add_incident_callback(self, callback: Callable[[SecurityIncident], None]):
        """Add incident callback function"""
        self.incident_callbacks.append(callback)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary"""
        with self._lock:
            recent_events = [
                event for event in self.security_events
                if (datetime.now() - event.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            open_incidents = sum(1 for incident in self.security_incidents.values() 
                               if incident.status == "open")
            
            return {
                'total_events': self.stats['total_events'],
                'recent_events_1h': len(recent_events),
                'events_by_type': dict(self.stats['events_by_type']),
                'events_by_severity': dict(self.stats['events_by_severity']),
                'total_incidents': len(self.security_incidents),
                'open_incidents': open_incidents,
                'incidents_resolved': self.stats['incidents_resolved'],
                'monitoring_status': 'active'
            }
    
    def get_recent_events(self, hours: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]
        
        # Sort by timestamp (newest first) and limit
        recent_events.sort(key=lambda x: x.timestamp, reverse=True)
        return [event.to_dict() for event in recent_events[:limit]]
    
    def get_incidents(self, status: str = None) -> List[Dict[str, Any]]:
        """Get security incidents"""
        incidents = list(self.security_incidents.values())
        
        if status:
            incidents = [incident for incident in incidents if incident.status == status]
        
        incidents.sort(key=lambda x: x.last_seen, reverse=True)
        return [incident.to_dict() for incident in incidents]
    
    def resolve_incident(self, incident_id: str) -> bool:
        """Mark incident as resolved"""
        if incident_id in self.security_incidents:
            self.security_incidents[incident_id].status = "resolved"
            self.stats['incidents_resolved'] += 1
            logger.info(f"Security incident resolved: {incident_id}")
            return True
        return False
    
    def _start_background_processing(self):
        """Start background event processing"""
        def processing_worker():
            while not self._stop_processing.wait(60):  # Process every minute
                try:
                    self._cleanup_old_data()
                    self._analyze_threat_patterns()
                except Exception as e:
                    logger.error(f"Background security processing failed: {e}")
        
        self._processing_thread = threading.Thread(target=processing_worker, daemon=True)
        self._processing_thread.start()
        logger.info("Background security processing started")
    
    def _cleanup_old_data(self):
        """Clean up old security data"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Clean old events (handled by deque maxlen)
            
            # Clean resolved incidents older than 7 days
            incident_cutoff = datetime.now() - timedelta(days=7)
            expired_incidents = [
                incident_id for incident_id, incident in self.security_incidents.items()
                if incident.status == "resolved" and incident.last_seen < incident_cutoff
            ]
            
            for incident_id in expired_incidents:
                del self.security_incidents[incident_id]
    
    def _analyze_threat_patterns(self):
        """Analyze patterns in security events"""
        # This could be expanded with ML-based threat analysis
        with self._lock:
            recent_events = [
                event for event in self.security_events
                if (datetime.now() - event.timestamp).total_seconds() < 3600
            ]
            
            # Update threat intelligence
            if len(recent_events) > 100:  # High activity
                logger.warning(f"High security event volume: {len(recent_events)} events in last hour")
    
    def shutdown(self):
        """Shutdown security monitor"""
        try:
            self._stop_processing.set()
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
            
            logger.info("Security monitor shutdown completed")
            
        except Exception as e:
            logger.error(f"Security monitor shutdown failed: {e}")


def create_security_monitor(retention_hours: int = 24) -> SecurityMonitor:
    """
    Factory function to create security monitor.
    
    Args:
        retention_hours: Hours to retain security events
        
    Returns:
        SecurityMonitor instance
    """
    return SecurityMonitor(retention_hours=retention_hours)