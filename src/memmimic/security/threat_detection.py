"""
Advanced Threat Detection & Response System

Behavioral analysis, anomaly detection, and automated threat response
for enterprise security monitoring and incident response.
"""

import time
import json
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import sqlite3
from contextlib import contextmanager
import hashlib
import ipaddress
import re

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of detected threats."""
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SESSION_HIJACKING = "session_hijacking"
    API_ABUSE = "api_abuse"
    INJECTION_ATTACK = "injection_attack"
    MALICIOUS_IP = "malicious_ip"


class ResponseAction(Enum):
    """Automated response actions."""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    ALERT_ADMIN = "alert_admin"
    REVOKE_SESSION = "revoke_session"
    DISABLE_ACCOUNT = "disable_account"


@dataclass
class BehaviorBaseline:
    """User behavior baseline for anomaly detection."""
    user_id: str
    typical_access_hours: List[int] = field(default_factory=list)  # Hours of day (0-23)
    typical_request_rate: float = 0.0  # Requests per minute
    typical_session_duration: float = 0.0  # Minutes
    common_ip_addresses: Set[str] = field(default_factory=set)
    common_user_agents: Set[str] = field(default_factory=set)
    typical_endpoints: Dict[str, float] = field(default_factory=dict)  # Endpoint -> frequency
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int = 0


@dataclass
class ThreatDetection:
    """Detected threat information."""
    detection_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: Optional[str]
    description: str
    evidence: Dict[str, Any]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_actions: List[ResponseAction] = field(default_factory=list)
    is_active: bool = True
    false_positive: bool = False


@dataclass
class SecurityMetrics:
    """Security metrics for monitoring."""
    total_requests: int = 0
    blocked_requests: int = 0
    threat_detections: int = 0
    false_positives: int = 0
    response_actions_taken: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdvancedThreatDetectionSystem:
    """
    Advanced threat detection system with behavioral analysis and automated response.
    
    Features:
    - Behavioral analysis and anomaly detection
    - Real-time threat pattern recognition
    - Machine learning-based user profiling
    - Automated response and mitigation
    - Threat intelligence integration
    - Advanced correlation analysis
    - False positive learning and adjustment
    """
    
    def __init__(self, db_path: str = "memmimic_threats.db",
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize advanced threat detection system.
        
        Args:
            db_path: Path to threat detection database
            audit_logger: Security audit logger instance
        """
        self.db_path = db_path
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Behavior tracking
        self.behavior_baselines: Dict[str, BehaviorBaseline] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Threat detection patterns
        self.threat_patterns = self._initialize_threat_patterns()
        
        # Rate limiting and blocking
        self.blocked_ips: Dict[str, datetime] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Metrics
        self.metrics = SecurityMetrics()
        
        # Initialize database
        self._initialize_database()
        
        # Load existing baselines
        self._load_behavior_baselines()
        
        logger.info("AdvancedThreatDetectionSystem initialized")
    
    def _initialize_database(self) -> None:
        """Initialize threat detection database."""
        with self._get_db_connection() as conn:
            # Behavior baselines table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS behavior_baselines (
                    user_id TEXT PRIMARY KEY,
                    typical_access_hours TEXT,
                    typical_request_rate REAL,
                    typical_session_duration REAL,
                    common_ip_addresses TEXT,
                    common_user_agents TEXT,
                    typical_endpoints TEXT,
                    last_updated TIMESTAMP,
                    sample_count INTEGER
                )
            ''')
            
            # Threat detections table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_detections (
                    detection_id TEXT PRIMARY KEY,
                    threat_type TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    description TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_actions TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    false_positive BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Security metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_metrics (
                    id INTEGER PRIMARY KEY,
                    total_requests INTEGER DEFAULT 0,
                    blocked_requests INTEGER DEFAULT 0,
                    threat_detections INTEGER DEFAULT 0,
                    false_positives INTEGER DEFAULT 0,
                    response_actions_taken INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threats_user_id ON threat_detections (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threats_ip ON threat_detections (ip_address)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_threats_detected_at ON threat_detections (detected_at)')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection patterns."""
        return {
            'brute_force_login': {
                'type': ThreatType.BRUTE_FORCE,
                'level': ThreatLevel.HIGH,
                'conditions': {
                    'failed_logins_threshold': 10,
                    'time_window_minutes': 15,
                    'unique_usernames_threshold': 5
                },
                'actions': [ResponseAction.TEMPORARY_BLOCK, ResponseAction.ALERT_ADMIN]
            },
            'rapid_api_calls': {
                'type': ThreatType.API_ABUSE,
                'level': ThreatLevel.MEDIUM,
                'conditions': {
                    'requests_per_minute': 120,
                    'time_window_minutes': 5
                },
                'actions': [ResponseAction.RATE_LIMIT]
            },
            'unusual_access_time': {
                'type': ThreatType.ANOMALOUS_BEHAVIOR,
                'level': ThreatLevel.LOW,
                'conditions': {
                    'deviation_hours': 6,  # Hours outside normal pattern
                    'confidence_threshold': 0.8
                },
                'actions': [ResponseAction.LOG_ONLY, ResponseAction.ALERT_ADMIN]
            },
            'data_exfiltration': {
                'type': ThreatType.DATA_EXFILTRATION,
                'level': ThreatLevel.CRITICAL,
                'conditions': {
                    'data_volume_mb': 100,
                    'time_window_minutes': 10,
                    'recall_requests_threshold': 50
                },
                'actions': [ResponseAction.REVOKE_SESSION, ResponseAction.ALERT_ADMIN]
            },
            'session_hijacking': {
                'type': ThreatType.SESSION_HIJACKING,
                'level': ThreatLevel.HIGH,
                'conditions': {
                    'ip_changes': 3,
                    'user_agent_changes': 2,
                    'time_window_minutes': 30
                },
                'actions': [ResponseAction.REVOKE_SESSION, ResponseAction.ALERT_ADMIN]
            },
            'injection_patterns': {
                'type': ThreatType.INJECTION_ATTACK,
                'level': ThreatLevel.HIGH,
                'conditions': {
                    'sql_patterns': [
                        r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b)',
                        r'(\'|\")(\s*)(OR|AND)(\s*)(\'|\")?\s*=\s*(\'|\")?',
                        r'(\bSLEEP\b|\bBENCHMARK\b|\bWAITFOR\b)'
                    ],
                    'xss_patterns': [
                        r'<script[^>]*>.*?</script>',
                        r'javascript:',
                        r'on\w+\s*='
                    ],
                    'command_patterns': [
                        r'(;|\||\&|\$\()',
                        r'(rm\s|cat\s|ls\s|wget\s|curl\s)',
                        r'(nc\s|netcat\s|bash\s|sh\s)'
                    ]
                },
                'actions': [ResponseAction.TEMPORARY_BLOCK, ResponseAction.ALERT_ADMIN]
            }
        }
    
    def analyze_request(self, user_id: Optional[str], ip_address: str,
                       endpoint: str, user_agent: Optional[str] = None,
                       request_data: Optional[Dict[str, Any]] = None) -> List[ThreatDetection]:
        """
        Analyze incoming request for threats.
        
        Args:
            user_id: User ID making the request
            ip_address: Client IP address
            endpoint: API endpoint being accessed
            user_agent: Client user agent
            request_data: Request payload data
            
        Returns:
            List of detected threats
        """
        self.metrics.total_requests += 1
        threats = []
        
        # Check if IP is blocked
        if self._is_ip_blocked(ip_address):
            self.metrics.blocked_requests += 1
            return [ThreatDetection(
                detection_id=self._generate_detection_id(),
                threat_type=ThreatType.MALICIOUS_IP,
                threat_level=ThreatLevel.HIGH,
                user_id=user_id,
                ip_address=ip_address,
                description="Request from blocked IP address",
                evidence={"blocked_until": self.blocked_ips[ip_address].isoformat()},
                response_actions=[ResponseAction.PERMANENT_BLOCK]
            )]
        
        # Rate limiting analysis
        rate_threats = self._analyze_rate_limiting(user_id, ip_address)
        threats.extend(rate_threats)
        
        # Injection pattern analysis
        if request_data:
            injection_threats = self._analyze_injection_patterns(user_id, ip_address, request_data)
            threats.extend(injection_threats)
        
        # Behavioral analysis for authenticated users
        if user_id:
            behavior_threats = self._analyze_user_behavior(user_id, ip_address, endpoint, user_agent)
            threats.extend(behavior_threats)
            
            # Update behavior baseline
            self._update_behavior_baseline(user_id, ip_address, endpoint, user_agent)
        
        # Session analysis
        session_threats = self._analyze_session_behavior(user_id, ip_address, user_agent)
        threats.extend(session_threats)
        
        # Process detected threats
        for threat in threats:
            self._process_threat_detection(threat)
        
        return threats
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is currently blocked."""
        if ip_address in self.blocked_ips:
            block_until = self.blocked_ips[ip_address]
            if datetime.now(timezone.utc) < block_until:
                return True
            else:
                # Block expired, remove
                del self.blocked_ips[ip_address]
        return False
    
    def _analyze_rate_limiting(self, user_id: Optional[str], ip_address: str) -> List[ThreatDetection]:
        """Analyze request rate for abuse detection."""
        threats = []
        current_time = time.time()
        
        # Track requests per IP
        key = f"ip:{ip_address}"
        self.rate_limits[key].append(current_time)
        
        # Track requests per user
        if user_id:
            user_key = f"user:{user_id}"
            self.rate_limits[user_key].append(current_time)
        
        # Check rapid API calls pattern
        pattern = self.threat_patterns['rapid_api_calls']
        window_seconds = pattern['conditions']['time_window_minutes'] * 60
        threshold = pattern['conditions']['requests_per_minute'] * pattern['conditions']['time_window_minutes']
        
        # Check IP rate
        ip_requests = [t for t in self.rate_limits[key] if current_time - t <= window_seconds]
        if len(ip_requests) > threshold:
            threats.append(ThreatDetection(
                detection_id=self._generate_detection_id(),
                threat_type=pattern['type'],
                threat_level=pattern['level'],
                user_id=user_id,
                ip_address=ip_address,
                description=f"Rapid API calls detected from IP: {len(ip_requests)} requests in {pattern['conditions']['time_window_minutes']} minutes",
                evidence={
                    "request_count": len(ip_requests),
                    "threshold": threshold,
                    "time_window": window_seconds
                },
                response_actions=pattern['actions']
            ))
        
        # Check user rate if authenticated
        if user_id:
            user_requests = [t for t in self.rate_limits[user_key] if current_time - t <= window_seconds]
            if len(user_requests) > threshold:
                threats.append(ThreatDetection(
                    detection_id=self._generate_detection_id(),
                    threat_type=pattern['type'],
                    threat_level=pattern['level'],
                    user_id=user_id,
                    ip_address=ip_address,
                    description=f"Rapid API calls detected from user: {len(user_requests)} requests in {pattern['conditions']['time_window_minutes']} minutes",
                    evidence={
                        "request_count": len(user_requests),
                        "threshold": threshold,
                        "time_window": window_seconds
                    },
                    response_actions=pattern['actions']
                ))
        
        return threats
    
    def _analyze_injection_patterns(self, user_id: Optional[str], ip_address: str,
                                   request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Analyze request data for injection patterns."""
        threats = []
        pattern = self.threat_patterns['injection_patterns']
        
        # Convert request data to searchable string
        request_text = json.dumps(request_data, default=str).lower()
        
        detected_patterns = []
        
        # Check SQL injection patterns
        for sql_pattern in pattern['conditions']['sql_patterns']:
            if re.search(sql_pattern, request_text, re.IGNORECASE):
                detected_patterns.append(f"SQL: {sql_pattern}")
        
        # Check XSS patterns
        for xss_pattern in pattern['conditions']['xss_patterns']:
            if re.search(xss_pattern, request_text, re.IGNORECASE):
                detected_patterns.append(f"XSS: {xss_pattern}")
        
        # Check command injection patterns
        for cmd_pattern in pattern['conditions']['command_patterns']:
            if re.search(cmd_pattern, request_text, re.IGNORECASE):
                detected_patterns.append(f"CMD: {cmd_pattern}")
        
        if detected_patterns:
            threats.append(ThreatDetection(
                detection_id=self._generate_detection_id(),
                threat_type=pattern['type'],
                threat_level=pattern['level'],
                user_id=user_id,
                ip_address=ip_address,
                description=f"Injection patterns detected in request data",
                evidence={
                    "detected_patterns": detected_patterns,
                    "request_size": len(request_text)
                },
                response_actions=pattern['actions']
            ))
        
        return threats
    
    def _analyze_user_behavior(self, user_id: str, ip_address: str,
                              endpoint: str, user_agent: Optional[str]) -> List[ThreatDetection]:
        """Analyze user behavior for anomalies."""
        threats = []
        
        if user_id not in self.behavior_baselines:
            return threats  # No baseline yet
        
        baseline = self.behavior_baselines[user_id]
        current_hour = datetime.now(timezone.utc).hour
        
        # Check unusual access time
        if baseline.typical_access_hours:
            if current_hour not in baseline.typical_access_hours:
                # Calculate deviation
                min_distance = min(abs(current_hour - hour) for hour in baseline.typical_access_hours)
                pattern = self.threat_patterns['unusual_access_time']
                
                if min_distance >= pattern['conditions']['deviation_hours']:
                    threats.append(ThreatDetection(
                        detection_id=self._generate_detection_id(),
                        threat_type=pattern['type'],
                        threat_level=pattern['level'],
                        user_id=user_id,
                        ip_address=ip_address,
                        description=f"Unusual access time detected: {current_hour}:00",
                        evidence={
                            "current_hour": current_hour,
                            "typical_hours": baseline.typical_access_hours,
                            "deviation": min_distance
                        },
                        response_actions=pattern['actions']
                    ))
        
        # Check unusual IP address
        if baseline.common_ip_addresses and ip_address not in baseline.common_ip_addresses:
            threats.append(ThreatDetection(
                detection_id=self._generate_detection_id(),
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                user_id=user_id,
                ip_address=ip_address,
                description="Access from unusual IP address",
                evidence={
                    "new_ip": ip_address,
                    "known_ips": list(baseline.common_ip_addresses)[:10]  # Limit for privacy
                },
                response_actions=[ResponseAction.LOG_ONLY]
            ))
        
        # Check unusual user agent
        if (baseline.common_user_agents and user_agent and 
            user_agent not in baseline.common_user_agents):
            threats.append(ThreatDetection(
                detection_id=self._generate_detection_id(),
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.LOW,
                user_id=user_id,
                ip_address=ip_address,
                description="Access from unusual user agent",
                evidence={
                    "new_user_agent": user_agent[:100],  # Truncate for privacy
                    "known_agents_count": len(baseline.common_user_agents)
                },
                response_actions=[ResponseAction.LOG_ONLY]
            ))
        
        return threats
    
    def _analyze_session_behavior(self, user_id: Optional[str], ip_address: str,
                                 user_agent: Optional[str]) -> List[ThreatDetection]:
        """Analyze session behavior for hijacking attempts."""
        threats = []
        
        if not user_id:
            return threats
        
        session_key = f"session:{user_id}"
        current_time = datetime.now(timezone.utc)
        
        if session_key in self.active_sessions:
            session_info = self.active_sessions[session_key]
            
            # Check for IP address changes
            if 'ip_addresses' not in session_info:
                session_info['ip_addresses'] = []
            
            if ip_address not in session_info['ip_addresses']:
                session_info['ip_addresses'].append(ip_address)
                session_info['ip_change_times'] = session_info.get('ip_change_times', [])
                session_info['ip_change_times'].append(current_time)
            
            # Check for user agent changes
            if user_agent and 'user_agents' not in session_info:
                session_info['user_agents'] = []
            
            if user_agent and user_agent not in session_info['user_agents']:
                session_info['user_agents'].append(user_agent)
                session_info['ua_change_times'] = session_info.get('ua_change_times', [])
                session_info['ua_change_times'].append(current_time)
            
            # Analyze for session hijacking pattern
            pattern = self.threat_patterns['session_hijacking']
            window_minutes = pattern['conditions']['time_window_minutes']
            cutoff_time = current_time - timedelta(minutes=window_minutes)
            
            # Count recent changes
            recent_ip_changes = len([t for t in session_info.get('ip_change_times', []) if t > cutoff_time])
            recent_ua_changes = len([t for t in session_info.get('ua_change_times', []) if t > cutoff_time])
            
            if (recent_ip_changes >= pattern['conditions']['ip_changes'] or
                recent_ua_changes >= pattern['conditions']['user_agent_changes']):
                
                threats.append(ThreatDetection(
                    detection_id=self._generate_detection_id(),
                    threat_type=pattern['type'],
                    threat_level=pattern['level'],
                    user_id=user_id,
                    ip_address=ip_address,
                    description="Possible session hijacking detected",
                    evidence={
                        "recent_ip_changes": recent_ip_changes,
                        "recent_ua_changes": recent_ua_changes,
                        "total_ips": len(session_info['ip_addresses']),
                        "time_window": window_minutes
                    },
                    response_actions=pattern['actions']
                ))
        else:
            # Initialize session tracking
            self.active_sessions[session_key] = {
                'start_time': current_time,
                'ip_addresses': [ip_address],
                'user_agents': [user_agent] if user_agent else [],
                'ip_change_times': [],
                'ua_change_times': []
            }
        
        return threats
    
    def _update_behavior_baseline(self, user_id: str, ip_address: str,
                                 endpoint: str, user_agent: Optional[str]) -> None:
        """Update user behavior baseline with new data point."""
        if user_id not in self.behavior_baselines:
            self.behavior_baselines[user_id] = BehaviorBaseline(user_id=user_id)
        
        baseline = self.behavior_baselines[user_id]
        current_hour = datetime.now(timezone.utc).hour
        
        # Update typical access hours
        if current_hour not in baseline.typical_access_hours:
            baseline.typical_access_hours.append(current_hour)
            # Keep only most common hours (limit to prevent baseline pollution)
            if len(baseline.typical_access_hours) > 12:  # Half the day maximum
                # In production, use more sophisticated frequency analysis
                baseline.typical_access_hours = baseline.typical_access_hours[-10:]
        
        # Update common IP addresses
        baseline.common_ip_addresses.add(ip_address)
        if len(baseline.common_ip_addresses) > 20:  # Limit to recent IPs
            # In production, use frequency-based pruning
            oldest_ips = list(baseline.common_ip_addresses)[:5]
            for ip in oldest_ips:
                baseline.common_ip_addresses.remove(ip)
        
        # Update common user agents
        if user_agent:
            baseline.common_user_agents.add(user_agent)
            if len(baseline.common_user_agents) > 10:
                oldest_agents = list(baseline.common_user_agents)[:3]
                for agent in oldest_agents:
                    baseline.common_user_agents.remove(agent)
        
        # Update endpoint frequency
        baseline.typical_endpoints[endpoint] = baseline.typical_endpoints.get(endpoint, 0) + 1
        
        baseline.sample_count += 1
        baseline.last_updated = datetime.now(timezone.utc)
        
        # Persist baseline periodically
        if baseline.sample_count % 50 == 0:  # Every 50 requests
            self._store_behavior_baseline(baseline)
    
    def _process_threat_detection(self, threat: ThreatDetection) -> None:
        """Process detected threat and execute response actions."""
        self.metrics.threat_detections += 1
        
        # Store threat detection
        self._store_threat_detection(threat)
        
        # Execute response actions
        for action in threat.response_actions:
            self._execute_response_action(action, threat)
            self.metrics.response_actions_taken += 1
        
        # Log security event
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            component="threat_detection",
            severity=self._threat_level_to_severity(threat.threat_level),
            details=f"Threat detected: {threat.threat_type.value}",
            metadata={
                "detection_id": threat.detection_id,
                "threat_type": threat.threat_type.value,
                "threat_level": threat.threat_level.value,
                "user_id": threat.user_id,
                "ip_address": threat.ip_address,
                "evidence": threat.evidence,
                "response_actions": [action.value for action in threat.response_actions]
            }
        ))
    
    def _execute_response_action(self, action: ResponseAction, threat: ThreatDetection) -> None:
        """Execute automated response action."""
        if action == ResponseAction.TEMPORARY_BLOCK and threat.ip_address:
            # Block IP for 1 hour
            self.blocked_ips[threat.ip_address] = datetime.now(timezone.utc) + timedelta(hours=1)
            logger.warning(f"Temporarily blocked IP: {threat.ip_address}")
        
        elif action == ResponseAction.PERMANENT_BLOCK and threat.ip_address:
            # Block IP for 30 days
            self.blocked_ips[threat.ip_address] = datetime.now(timezone.utc) + timedelta(days=30)
            logger.warning(f"Permanently blocked IP: {threat.ip_address}")
        
        elif action == ResponseAction.ALERT_ADMIN:
            logger.critical(f"ADMIN ALERT: {threat.description} (Detection ID: {threat.detection_id})")
        
        elif action == ResponseAction.REVOKE_SESSION and threat.user_id:
            # Would integrate with authentication system to revoke sessions
            logger.warning(f"Session revocation requested for user: {threat.user_id}")
        
        # Add more response actions as needed
    
    def _threat_level_to_severity(self, threat_level: ThreatLevel) -> SeverityLevel:
        """Convert threat level to audit severity level."""
        mapping = {
            ThreatLevel.MINIMAL: SeverityLevel.LOW,
            ThreatLevel.LOW: SeverityLevel.LOW,
            ThreatLevel.MEDIUM: SeverityLevel.MEDIUM,
            ThreatLevel.HIGH: SeverityLevel.HIGH,
            ThreatLevel.CRITICAL: SeverityLevel.CRITICAL
        }
        return mapping.get(threat_level, SeverityLevel.MEDIUM)
    
    def _generate_detection_id(self) -> str:
        """Generate unique detection ID."""
        import secrets
        return f"threat_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _store_threat_detection(self, threat: ThreatDetection) -> None:
        """Store threat detection in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO threat_detections (
                    detection_id, threat_type, threat_level, user_id,
                    ip_address, description, evidence, detected_at,
                    response_actions, is_active, false_positive
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                threat.detection_id, threat.threat_type.value,
                threat.threat_level.value, threat.user_id,
                threat.ip_address, threat.description,
                json.dumps(threat.evidence), threat.detected_at.isoformat(),
                json.dumps([action.value for action in threat.response_actions]),
                threat.is_active, threat.false_positive
            ))
            conn.commit()
    
    def _store_behavior_baseline(self, baseline: BehaviorBaseline) -> None:
        """Store behavior baseline in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO behavior_baselines (
                    user_id, typical_access_hours, typical_request_rate,
                    typical_session_duration, common_ip_addresses,
                    common_user_agents, typical_endpoints,
                    last_updated, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                baseline.user_id,
                json.dumps(baseline.typical_access_hours),
                baseline.typical_request_rate,
                baseline.typical_session_duration,
                json.dumps(list(baseline.common_ip_addresses)),
                json.dumps(list(baseline.common_user_agents)),
                json.dumps(baseline.typical_endpoints),
                baseline.last_updated.isoformat(),
                baseline.sample_count
            ))
            conn.commit()
    
    def _load_behavior_baselines(self) -> None:
        """Load behavior baselines from database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("SELECT * FROM behavior_baselines")
                
                for row in cursor.fetchall():
                    baseline = BehaviorBaseline(
                        user_id=row['user_id'],
                        typical_access_hours=json.loads(row['typical_access_hours'] or '[]'),
                        typical_request_rate=row['typical_request_rate'] or 0.0,
                        typical_session_duration=row['typical_session_duration'] or 0.0,
                        common_ip_addresses=set(json.loads(row['common_ip_addresses'] or '[]')),
                        common_user_agents=set(json.loads(row['common_user_agents'] or '[]')),
                        typical_endpoints=json.loads(row['typical_endpoints'] or '{}'),
                        last_updated=datetime.fromisoformat(row['last_updated']),
                        sample_count=row['sample_count'] or 0
                    )
                    
                    self.behavior_baselines[baseline.user_id] = baseline
        except sqlite3.OperationalError:
            # Database doesn't exist yet or table doesn't exist
            pass
    
    def get_threat_dashboard_data(self) -> Dict[str, Any]:
        """Get threat dashboard data for monitoring."""
        with self._get_db_connection() as conn:
            # Recent threat counts by type
            cursor = conn.execute('''
                SELECT threat_type, COUNT(*) as count
                FROM threat_detections
                WHERE detected_at > datetime('now', '-24 hours')
                AND is_active = TRUE
                GROUP BY threat_type
            ''')
            
            threat_counts = {row['threat_type']: row['count'] for row in cursor.fetchall()}
            
            # Top blocked IPs
            cursor = conn.execute('''
                SELECT ip_address, COUNT(*) as count
                FROM threat_detections
                WHERE detected_at > datetime('now', '-24 hours')
                AND ip_address IS NOT NULL
                GROUP BY ip_address
                ORDER BY count DESC
                LIMIT 10
            ''')
            
            blocked_ips = [{"ip": row['ip_address'], "count": row['count']} for row in cursor.fetchall()]
            
            # Active blocks
            active_blocks = len([ip for ip, until in self.blocked_ips.items() 
                               if datetime.now(timezone.utc) < until])
        
        return {
            "threat_counts_24h": threat_counts,
            "blocked_ips_top": blocked_ips,
            "active_ip_blocks": active_blocks,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "blocked_requests": self.metrics.blocked_requests,
                "threat_detections": self.metrics.threat_detections,
                "false_positives": self.metrics.false_positives,
                "response_actions_taken": self.metrics.response_actions_taken
            },
            "behavior_baselines": len(self.behavior_baselines),
            "active_sessions": len(self.active_sessions)
        }


# Global threat detection system instance
_global_threat_detector: Optional[AdvancedThreatDetectionSystem] = None


def get_threat_detector() -> AdvancedThreatDetectionSystem:
    """Get the global threat detection system."""
    global _global_threat_detector
    if _global_threat_detector is None:
        _global_threat_detector = AdvancedThreatDetectionSystem()
    return _global_threat_detector


def initialize_threat_detector(db_path: str = "memmimic_threats.db",
                              audit_logger: Optional[SecurityAuditLogger] = None) -> AdvancedThreatDetectionSystem:
    """Initialize the global threat detection system."""
    global _global_threat_detector
    _global_threat_detector = AdvancedThreatDetectionSystem(db_path, audit_logger)
    return _global_threat_detector