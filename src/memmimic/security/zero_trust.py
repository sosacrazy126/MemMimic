"""
Zero-Trust Security Architecture

Context-aware access control, network segmentation, and continuous
verification for enterprise zero-trust security implementation.
"""

import json
import time
import ipaddress
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import sqlite3
from contextlib import contextmanager
import geoip2.database
import geoip2.errors
import user_agents

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel
from .authentication import User, Permission

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Zero-trust verification levels."""
    UNTRUSTED = "untrusted"
    LOW_TRUST = "low_trust"
    MEDIUM_TRUST = "medium_trust"
    HIGH_TRUST = "high_trust"
    VERIFIED = "verified"


class AccessDecision(Enum):
    """Access control decisions."""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    STEP_UP_AUTH = "step_up_auth"
    CONDITIONAL_ALLOW = "conditional_allow"


class RiskScore(Enum):
    """Risk assessment scores."""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class DeviceType(Enum):
    """Device types for trust evaluation."""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    MOBILE = "mobile"
    TABLET = "tablet"
    SERVER = "server"
    IOT = "iot"
    UNKNOWN = "unknown"


@dataclass
class NetworkSegment:
    """Network segment definition."""
    segment_id: str
    name: str
    cidr_blocks: List[str]
    trust_level: TrustLevel
    allowed_protocols: List[str] = field(default_factory=list)
    security_policies: List[str] = field(default_factory=list)
    monitoring_level: str = "standard"  # minimal, standard, enhanced
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Device:
    """Device trust profile."""
    device_id: str
    device_name: Optional[str]
    device_type: DeviceType
    os_family: Optional[str]
    os_version: Optional[str]
    user_agent: Optional[str]
    fingerprint: str  # Device fingerprint hash
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: int = 3  # 1-5 scale
    compliance_status: bool = False
    managed_device: bool = False
    certificate_enrolled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocationContext:
    """Geographic and network location context."""
    ip_address: str
    country: Optional[str] = None
    city: Optional[str] = None
    asn: Optional[int] = None
    isp: Optional[str] = None
    is_tor: bool = False
    is_proxy: bool = False
    is_vpn: bool = False
    network_segment: Optional[str] = None
    risk_score: int = 1  # 1-5 scale


@dataclass
class AccessContext:
    """Complete access request context."""
    user: User
    device: Device
    location: LocationContext
    resource: str
    action: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    additional_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRule:
    """Zero-trust policy rule."""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]  # Conditions to match
    action: AccessDecision
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessEvaluation:
    """Access evaluation result."""
    evaluation_id: str
    context: AccessContext
    decision: AccessDecision
    trust_score: float  # 0.0 to 1.0
    risk_score: int  # 1-5 scale
    matched_rules: List[str]
    reasoning: List[str]
    required_mitigations: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ZeroTrustError(Exception):
    """Zero-trust related errors."""
    pass


class ZeroTrustEngine:
    """
    Zero-Trust Security Architecture Engine.
    
    Features:
    - Context-aware access control with continuous verification
    - Device trust assessment and compliance checking
    - Network microsegmentation and policy enforcement
    - Risk-based authentication and authorization
    - Geographic and behavioral anomaly detection
    - Dynamic policy adaptation based on threat intelligence
    - Comprehensive audit and compliance reporting
    """
    
    def __init__(self, db_path: str = "memmimic_zerotrust.db",
                 geoip_db_path: Optional[str] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize Zero-Trust engine.
        
        Args:
            db_path: Path to zero-trust database
            geoip_db_path: Path to GeoIP database file
            audit_logger: Security audit logger instance
        """
        self.db_path = Path(db_path)
        self.geoip_db_path = geoip_db_path
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Initialize GeoIP database
        self.geoip_reader = None
        if geoip_db_path and Path(geoip_db_path).exists():
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning(f"Failed to initialize GeoIP database: {e}")
        
        # Network segments
        self.network_segments: Dict[str, NetworkSegment] = {}
        
        # Trust policies
        self.policy_rules: List[PolicyRule] = []
        
        # Device registry
        self.devices: Dict[str, Device] = {}
        
        # Initialize database
        self._initialize_database()
        
        # Initialize default network segments and policies
        self._initialize_default_configuration()
        
        # Load existing data
        self._load_zero_trust_data()
        
        logger.info("ZeroTrustEngine initialized")
    
    def _initialize_database(self) -> None:
        """Initialize zero-trust database."""
        with self._get_db_connection() as conn:
            # Network segments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS network_segments (
                    segment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cidr_blocks TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    allowed_protocols TEXT,
                    security_policies TEXT,
                    monitoring_level TEXT DEFAULT 'standard',
                    metadata TEXT
                )
            ''')
            
            # Devices table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT,
                    device_type TEXT NOT NULL,
                    os_family TEXT,
                    os_version TEXT,
                    user_agent TEXT,
                    fingerprint TEXT NOT NULL,
                    trust_level TEXT DEFAULT 'untrusted',
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    risk_score INTEGER DEFAULT 3,
                    compliance_status BOOLEAN DEFAULT FALSE,
                    managed_device BOOLEAN DEFAULT FALSE,
                    certificate_enrolled BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            ''')
            
            # Policy rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS policy_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    conditions TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER DEFAULT 100,
                    enabled BOOLEAN DEFAULT TRUE,
                    metadata TEXT
                )
            ''')
            
            # Access evaluations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    risk_score INTEGER NOT NULL,
                    matched_rules TEXT,
                    reasoning TEXT,
                    required_mitigations TEXT,
                    expires_at TIMESTAMP,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    location_data TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_devices_fingerprint ON devices (fingerprint)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_devices_trust_level ON devices (trust_level)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_user ON access_evaluations (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_device ON access_evaluations (device_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_decision ON access_evaluations (decision)')
            
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
    
    def _initialize_default_configuration(self) -> None:
        """Initialize default network segments and policies."""
        # Default network segments
        default_segments = [
            NetworkSegment(
                segment_id="corporate_lan",
                name="Corporate LAN",
                cidr_blocks=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
                trust_level=TrustLevel.MEDIUM_TRUST,
                allowed_protocols=["HTTPS", "SSH", "LDAPS"],
                security_policies=["corporate_baseline"],
                monitoring_level="standard"
            ),
            NetworkSegment(
                segment_id="guest_network",
                name="Guest Network",
                cidr_blocks=["192.168.100.0/24"],
                trust_level=TrustLevel.LOW_TRUST,
                allowed_protocols=["HTTPS"],
                security_policies=["guest_restricted"],
                monitoring_level="enhanced"
            ),
            NetworkSegment(
                segment_id="public_internet",
                name="Public Internet",
                cidr_blocks=["0.0.0.0/0"],
                trust_level=TrustLevel.UNTRUSTED,
                allowed_protocols=["HTTPS"],
                security_policies=["public_restricted"],
                monitoring_level="enhanced"
            )
        ]
        
        for segment in default_segments:
            self.network_segments[segment.segment_id] = segment
            self._store_network_segment(segment)
        
        # Default policy rules
        default_policies = [
            PolicyRule(
                rule_id="deny_untrusted_admin",
                name="Deny Admin Access from Untrusted Devices",
                description="Block administrative actions from untrusted devices",
                conditions={
                    "device_trust_level": ["untrusted", "low_trust"],
                    "required_permissions": ["SYSTEM_ADMIN", "MANAGE_USERS"]
                },
                action=AccessDecision.DENY,
                priority=10
            ),
            PolicyRule(
                rule_id="mfa_high_risk",
                name="Require MFA for High Risk Access",
                description="Require additional authentication for high-risk scenarios",
                conditions={
                    "risk_score": {"min": 4},
                    "OR": [
                        {"location_anomaly": True},
                        {"device_trust_level": ["untrusted", "low_trust"]},
                        {"time_anomaly": True}
                    ]
                },
                action=AccessDecision.STEP_UP_AUTH,
                priority=20
            ),
            PolicyRule(
                rule_id="allow_verified_corporate",
                name="Allow Verified Corporate Access",
                description="Allow access from verified devices on corporate network",
                conditions={
                    "device_trust_level": ["high_trust", "verified"],
                    "network_segment": "corporate_lan",
                    "device_compliance": True
                },
                action=AccessDecision.ALLOW,
                priority=100
            ),
            PolicyRule(
                rule_id="challenge_anomalous_location",
                name="Challenge Anomalous Locations",
                description="Challenge access from unusual geographic locations",
                conditions={
                    "location_anomaly": True,
                    "NOT": {"device_trust_level": "verified"}
                },
                action=AccessDecision.CHALLENGE,
                priority=30
            )
        ]
        
        for policy in default_policies:
            self.policy_rules.append(policy)
            self._store_policy_rule(policy)
        
        logger.info(f"Initialized {len(default_segments)} network segments and {len(default_policies)} policy rules")
    
    def _load_zero_trust_data(self) -> None:
        """Load existing zero-trust data from database."""
        try:
            with self._get_db_connection() as conn:
                # Load network segments
                cursor = conn.execute("SELECT * FROM network_segments")
                for row in cursor.fetchall():
                    segment = NetworkSegment(
                        segment_id=row['segment_id'],
                        name=row['name'],
                        cidr_blocks=json.loads(row['cidr_blocks']),
                        trust_level=TrustLevel(row['trust_level']),
                        allowed_protocols=json.loads(row['allowed_protocols'] or '[]'),
                        security_policies=json.loads(row['security_policies'] or '[]'),
                        monitoring_level=row['monitoring_level'] or "standard",
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    self.network_segments[segment.segment_id] = segment
                
                # Load devices
                cursor = conn.execute("SELECT * FROM devices")
                for row in cursor.fetchall():
                    device = Device(
                        device_id=row['device_id'],
                        device_name=row['device_name'],
                        device_type=DeviceType(row['device_type']),
                        os_family=row['os_family'],
                        os_version=row['os_version'],
                        user_agent=row['user_agent'],
                        fingerprint=row['fingerprint'],
                        trust_level=TrustLevel(row['trust_level']),
                        last_seen=datetime.fromisoformat(row['last_seen']),
                        risk_score=row['risk_score'] or 3,
                        compliance_status=bool(row['compliance_status']),
                        managed_device=bool(row['managed_device']),
                        certificate_enrolled=bool(row['certificate_enrolled']),
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    self.devices[device.device_id] = device
                
                # Load policy rules
                self.policy_rules = []
                cursor = conn.execute("SELECT * FROM policy_rules ORDER BY priority ASC")
                for row in cursor.fetchall():
                    rule = PolicyRule(
                        rule_id=row['rule_id'],
                        name=row['name'],
                        description=row['description'] or "",
                        conditions=json.loads(row['conditions']),
                        action=AccessDecision(row['action']),
                        priority=row['priority'] or 100,
                        enabled=bool(row['enabled']),
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    self.policy_rules.append(rule)
                    
        except sqlite3.OperationalError:
            # Database doesn't exist yet
            pass
    
    def evaluate_access(self, user: User, device_info: Dict[str, Any],
                       ip_address: str, resource: str, action: str,
                       session_id: Optional[str] = None) -> AccessEvaluation:
        """
        Evaluate access request using zero-trust principles.
        
        Args:
            user: User requesting access
            device_info: Device information (user_agent, fingerprint, etc.)
            ip_address: Source IP address
            resource: Resource being accessed
            action: Action being performed
            session_id: Current session ID
            
        Returns:
            Access evaluation result
        """
        # Create or update device profile
        device = self._get_or_create_device(device_info, ip_address)
        
        # Get location context
        location = self._analyze_location_context(ip_address)
        
        # Create access context
        context = AccessContext(
            user=user,
            device=device,
            location=location,
            resource=resource,
            action=action,
            session_id=session_id
        )
        
        # Calculate trust and risk scores
        trust_score = self._calculate_trust_score(context)
        risk_score = self._calculate_risk_score(context)
        
        # Evaluate against policies
        decision, matched_rules, reasoning, mitigations = self._evaluate_policies(context, trust_score, risk_score)
        
        # Create evaluation result
        evaluation_id = f"eval_{int(time.time())}_{hashlib.md5(f'{user.user_id}{device.device_id}{resource}'.encode()).hexdigest()[:8]}"
        
        # Set expiration for conditional decisions
        expires_at = None
        if decision in [AccessDecision.CONDITIONAL_ALLOW]:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        evaluation = AccessEvaluation(
            evaluation_id=evaluation_id,
            context=context,
            decision=decision,
            trust_score=trust_score,
            risk_score=risk_score.value,
            matched_rules=matched_rules,
            reasoning=reasoning,
            required_mitigations=mitigations,
            expires_at=expires_at
        )
        
        # Store evaluation
        self._store_access_evaluation(evaluation)
        
        # Log access decision
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.ACCESS_CONTROL,
            component="zero_trust",
            severity=self._decision_to_severity(decision),
            details=f"Zero-trust access decision: {decision.value}",
            metadata={
                "user_id": user.user_id,
                "device_id": device.device_id,
                "resource": resource,
                "action": action,
                "decision": decision.value,
                "trust_score": trust_score,
                "risk_score": risk_score.value,
                "ip_address": ip_address
            }
        ))
        
        return evaluation
    
    def _get_or_create_device(self, device_info: Dict[str, Any], ip_address: str) -> Device:
        """Get or create device profile."""
        # Create device fingerprint
        fingerprint_data = {
            "user_agent": device_info.get("user_agent", ""),
            "screen_resolution": device_info.get("screen_resolution", ""),
            "timezone": device_info.get("timezone", ""),
            "language": device_info.get("language", ""),
            "plugins": device_info.get("plugins", [])
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        
        # Check if device exists
        if fingerprint in [d.fingerprint for d in self.devices.values()]:
            device = next(d for d in self.devices.values() if d.fingerprint == fingerprint)
            # Update last seen
            device.last_seen = datetime.now(timezone.utc)
            self._store_device(device)
            return device
        
        # Create new device
        device_id = f"device_{int(time.time())}_{fingerprint[:8]}"
        
        # Parse user agent
        device_type = DeviceType.UNKNOWN
        os_family = None
        os_version = None
        
        user_agent_str = device_info.get("user_agent", "")
        if user_agent_str:
            try:
                ua = user_agents.parse(user_agent_str)
                if ua.is_mobile:
                    device_type = DeviceType.MOBILE
                elif ua.is_tablet:
                    device_type = DeviceType.TABLET
                elif ua.is_pc:
                    device_type = DeviceType.DESKTOP
                
                os_family = ua.os.family
                os_version = ua.os.version_string
            except Exception:
                pass
        
        device = Device(
            device_id=device_id,
            device_name=device_info.get("device_name"),
            device_type=device_type,
            os_family=os_family,
            os_version=os_version,
            user_agent=user_agent_str,
            fingerprint=fingerprint,
            trust_level=TrustLevel.UNTRUSTED,
            risk_score=4,  # New devices start with higher risk
            compliance_status=False,
            managed_device=False,
            certificate_enrolled=False,
            metadata={
                "first_seen_ip": ip_address,
                "creation_method": "automatic"
            }
        )
        
        self.devices[device_id] = device
        self._store_device(device)
        
        # Log new device registration
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="zero_trust",
            severity=SeverityLevel.MEDIUM,
            details=f"New device registered: {device_id}",
            metadata={
                "device_id": device_id,
                "device_type": device_type.value,
                "fingerprint": fingerprint[:16],
                "ip_address": ip_address
            }
        ))
        
        return device
    
    def _analyze_location_context(self, ip_address: str) -> LocationContext:
        """Analyze geographic and network location context."""
        location = LocationContext(ip_address=ip_address)
        
        try:
            # Determine network segment
            ip_obj = ipaddress.ip_address(ip_address)
            for segment_id, segment in self.network_segments.items():
                for cidr in segment.cidr_blocks:
                    network = ipaddress.ip_network(cidr, strict=False)
                    if ip_obj in network:
                        location.network_segment = segment_id
                        break
                if location.network_segment:
                    break
            
            # GeoIP analysis
            if self.geoip_reader:
                try:
                    response = self.geoip_reader.city(ip_address)
                    location.country = response.country.iso_code
                    location.city = response.city.name
                    
                    # ASN information
                    asn_response = self.geoip_reader.asn(ip_address)
                    location.asn = asn_response.autonomous_system_number
                    location.isp = asn_response.autonomous_system_organization
                    
                except geoip2.errors.AddressNotFoundError:
                    pass
        
        except Exception as e:
            logger.warning(f"Error analyzing location context: {e}")
        
        # Risk assessment
        location.risk_score = self._calculate_location_risk(location)
        
        return location
    
    def _calculate_location_risk(self, location: LocationContext) -> int:
        """Calculate risk score for location context."""
        risk_score = 1
        
        # High-risk countries (example - would be configurable)
        high_risk_countries = ["CN", "RU", "KP", "IR"]
        if location.country in high_risk_countries:
            risk_score += 2
        
        # Tor/Proxy/VPN detection (would integrate with threat intelligence)
        if location.is_tor or location.is_proxy:
            risk_score += 3
        elif location.is_vpn:
            risk_score += 1
        
        # Unknown or public networks
        if not location.network_segment or location.network_segment == "public_internet":
            risk_score += 1
        
        return min(risk_score, 5)
    
    def _calculate_trust_score(self, context: AccessContext) -> float:
        """Calculate overall trust score (0.0 to 1.0)."""
        scores = []
        
        # User trust factors
        user_score = 0.5  # Base score
        if context.user.mfa_enabled:
            user_score += 0.2
        if context.user.role.value in ["admin", "system_admin"]:
            user_score += 0.1
        scores.append(min(user_score, 1.0))
        
        # Device trust factors
        device_trust_values = {
            TrustLevel.VERIFIED: 1.0,
            TrustLevel.HIGH_TRUST: 0.8,
            TrustLevel.MEDIUM_TRUST: 0.6,
            TrustLevel.LOW_TRUST: 0.3,
            TrustLevel.UNTRUSTED: 0.1
        }
        device_score = device_trust_values[context.device.trust_level]
        
        if context.device.managed_device:
            device_score += 0.1
        if context.device.certificate_enrolled:
            device_score += 0.1
        if context.device.compliance_status:
            device_score += 0.1
        
        scores.append(min(device_score, 1.0))
        
        # Location trust factors
        location_score = 0.5  # Base score
        if context.location.network_segment in ["corporate_lan"]:
            location_score += 0.3
        elif context.location.network_segment in ["guest_network"]:
            location_score += 0.1
        else:
            location_score -= 0.2
        
        # Reduce score for high-risk locations
        location_score -= (context.location.risk_score - 1) * 0.1
        
        scores.append(max(location_score, 0.0))
        
        # Calculate weighted average
        weights = [0.3, 0.5, 0.2]  # User, Device, Location
        trust_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return max(0.0, min(trust_score, 1.0))
    
    def _calculate_risk_score(self, context: AccessContext) -> RiskScore:
        """Calculate overall risk score."""
        risk_factors = 0
        
        # Device risk
        risk_factors += context.device.risk_score
        
        # Location risk
        risk_factors += context.location.risk_score
        
        # Time-based risk (accessing during unusual hours)
        current_hour = datetime.now(timezone.utc).hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            risk_factors += 1
        
        # Resource sensitivity
        sensitive_resources = ["admin", "config", "users", "security"]
        if any(resource in context.resource.lower() for resource in sensitive_resources):
            risk_factors += 2
        
        # Action sensitivity
        sensitive_actions = ["delete", "modify", "admin", "config"]
        if any(action in context.action.lower() for action in sensitive_actions):
            risk_factors += 1
        
        # Convert to RiskScore enum
        if risk_factors <= 2:
            return RiskScore.MINIMAL
        elif risk_factors <= 4:
            return RiskScore.LOW
        elif risk_factors <= 6:
            return RiskScore.MEDIUM
        elif risk_factors <= 8:
            return RiskScore.HIGH
        else:
            return RiskScore.CRITICAL
    
    def _evaluate_policies(self, context: AccessContext, trust_score: float,
                          risk_score: RiskScore) -> Tuple[AccessDecision, List[str], List[str], List[str]]:
        """Evaluate access against zero-trust policies."""
        matched_rules = []
        reasoning = []
        mitigations = []
        
        # Default decision is DENY (zero-trust principle)
        decision = AccessDecision.DENY
        
        # Evaluate rules in priority order
        for rule in sorted(self.policy_rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue
            
            if self._evaluate_rule_conditions(rule, context, trust_score, risk_score):
                matched_rules.append(rule.rule_id)
                reasoning.append(f"Matched rule: {rule.name}")
                decision = rule.action
                
                # Add mitigations based on rule
                if rule.action == AccessDecision.STEP_UP_AUTH:
                    mitigations.append("Additional authentication required")
                elif rule.action == AccessDecision.CHALLENGE:
                    mitigations.append("User verification challenge required")
                elif rule.action == AccessDecision.CONDITIONAL_ALLOW:
                    mitigations.append("Enhanced monitoring enabled")
                
                # First matching rule wins (highest priority)
                break
        
        if not matched_rules:
            reasoning.append("No matching policy rules - default DENY")
        
        return decision, matched_rules, reasoning, mitigations
    
    def _evaluate_rule_conditions(self, rule: PolicyRule, context: AccessContext,
                                 trust_score: float, risk_score: RiskScore) -> bool:
        """Evaluate if rule conditions match the context."""
        conditions = rule.conditions
        
        try:
            return self._evaluate_condition_tree(conditions, context, trust_score, risk_score)
        except Exception as e:
            logger.warning(f"Error evaluating rule {rule.rule_id}: {e}")
            return False
    
    def _evaluate_condition_tree(self, conditions: Dict[str, Any], context: AccessContext,
                                trust_score: float, risk_score: RiskScore) -> bool:
        """Recursively evaluate condition tree."""
        # Handle logical operators
        if "AND" in conditions:
            return all(self._evaluate_condition_tree(cond, context, trust_score, risk_score) 
                      for cond in conditions["AND"])
        
        if "OR" in conditions:
            return any(self._evaluate_condition_tree(cond, context, trust_score, risk_score) 
                      for cond in conditions["OR"])
        
        if "NOT" in conditions:
            return not self._evaluate_condition_tree(conditions["NOT"], context, trust_score, risk_score)
        
        # Evaluate individual conditions
        for condition_key, condition_value in conditions.items():
            if not self._evaluate_single_condition(condition_key, condition_value, context, trust_score, risk_score):
                return False
        
        return True
    
    def _evaluate_single_condition(self, condition_key: str, condition_value: Any,
                                  context: AccessContext, trust_score: float, risk_score: RiskScore) -> bool:
        """Evaluate a single condition."""
        if condition_key == "device_trust_level":
            if isinstance(condition_value, list):
                return context.device.trust_level.value in condition_value
            return context.device.trust_level.value == condition_value
        
        elif condition_key == "network_segment":
            return context.location.network_segment == condition_value
        
        elif condition_key == "device_compliance":
            return context.device.compliance_status == condition_value
        
        elif condition_key == "managed_device":
            return context.device.managed_device == condition_value
        
        elif condition_key == "risk_score":
            if isinstance(condition_value, dict):
                min_risk = condition_value.get("min", 1)
                max_risk = condition_value.get("max", 5)
                return min_risk <= risk_score.value <= max_risk
            return risk_score.value == condition_value
        
        elif condition_key == "trust_score":
            if isinstance(condition_value, dict):
                min_trust = condition_value.get("min", 0.0)
                max_trust = condition_value.get("max", 1.0)
                return min_trust <= trust_score <= max_trust
            return trust_score == condition_value
        
        elif condition_key == "required_permissions":
            # Would integrate with permission system
            return True  # Simplified for now
        
        elif condition_key == "location_anomaly":
            # Simplified location anomaly detection
            return context.location.risk_score >= 3
        
        elif condition_key == "time_anomaly":
            # Simplified time anomaly detection
            current_hour = datetime.now(timezone.utc).hour
            return current_hour < 6 or current_hour > 22
        
        return False
    
    def _decision_to_severity(self, decision: AccessDecision) -> SeverityLevel:
        """Convert access decision to audit severity level."""
        mapping = {
            AccessDecision.DENY: SeverityLevel.HIGH,
            AccessDecision.STEP_UP_AUTH: SeverityLevel.MEDIUM,
            AccessDecision.CHALLENGE: SeverityLevel.MEDIUM,
            AccessDecision.CONDITIONAL_ALLOW: SeverityLevel.LOW,
            AccessDecision.ALLOW: SeverityLevel.LOW
        }
        return mapping.get(decision, SeverityLevel.MEDIUM)
    
    def _store_network_segment(self, segment: NetworkSegment) -> None:
        """Store network segment in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO network_segments (
                    segment_id, name, cidr_blocks, trust_level,
                    allowed_protocols, security_policies, monitoring_level, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                segment.segment_id, segment.name, json.dumps(segment.cidr_blocks),
                segment.trust_level.value, json.dumps(segment.allowed_protocols),
                json.dumps(segment.security_policies), segment.monitoring_level,
                json.dumps(segment.metadata)
            ))
            conn.commit()
    
    def _store_device(self, device: Device) -> None:
        """Store device in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO devices (
                    device_id, device_name, device_type, os_family, os_version,
                    user_agent, fingerprint, trust_level, last_seen, risk_score,
                    compliance_status, managed_device, certificate_enrolled, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                device.device_id, device.device_name, device.device_type.value,
                device.os_family, device.os_version, device.user_agent,
                device.fingerprint, device.trust_level.value,
                device.last_seen.isoformat(), device.risk_score,
                device.compliance_status, device.managed_device,
                device.certificate_enrolled, json.dumps(device.metadata)
            ))
            conn.commit()
    
    def _store_policy_rule(self, rule: PolicyRule) -> None:
        """Store policy rule in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO policy_rules (
                    rule_id, name, description, conditions, action,
                    priority, enabled, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id, rule.name, rule.description,
                json.dumps(rule.conditions), rule.action.value,
                rule.priority, rule.enabled, json.dumps(rule.metadata)
            ))
            conn.commit()
    
    def _store_access_evaluation(self, evaluation: AccessEvaluation) -> None:
        """Store access evaluation in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO access_evaluations (
                    evaluation_id, user_id, device_id, resource, action,
                    decision, trust_score, risk_score, matched_rules,
                    reasoning, required_mitigations, expires_at, evaluated_at,
                    ip_address, location_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evaluation.evaluation_id, evaluation.context.user.user_id,
                evaluation.context.device.device_id, evaluation.context.resource,
                evaluation.context.action, evaluation.decision.value,
                evaluation.trust_score, evaluation.risk_score,
                json.dumps(evaluation.matched_rules), json.dumps(evaluation.reasoning),
                json.dumps(evaluation.required_mitigations),
                evaluation.expires_at.isoformat() if evaluation.expires_at else None,
                evaluation.evaluated_at.isoformat(),
                evaluation.context.location.ip_address,
                json.dumps({
                    "country": evaluation.context.location.country,
                    "city": evaluation.context.location.city,
                    "network_segment": evaluation.context.location.network_segment,
                    "risk_score": evaluation.context.location.risk_score
                })
            ))
            conn.commit()
    
    def update_device_trust(self, device_id: str, trust_level: TrustLevel,
                           compliance_status: bool = None,
                           managed_device: bool = None) -> bool:
        """Update device trust level and compliance status."""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        old_trust_level = device.trust_level
        
        device.trust_level = trust_level
        if compliance_status is not None:
            device.compliance_status = compliance_status
        if managed_device is not None:
            device.managed_device = managed_device
        
        # Adjust risk score based on trust level
        trust_to_risk = {
            TrustLevel.VERIFIED: 1,
            TrustLevel.HIGH_TRUST: 2,
            TrustLevel.MEDIUM_TRUST: 3,
            TrustLevel.LOW_TRUST: 4,
            TrustLevel.UNTRUSTED: 5
        }
        device.risk_score = trust_to_risk[trust_level]
        
        self._store_device(device)
        
        # Log trust level change
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            component="zero_trust",
            severity=SeverityLevel.MEDIUM,
            details=f"Device trust level updated: {device_id}",
            metadata={
                "device_id": device_id,
                "old_trust_level": old_trust_level.value,
                "new_trust_level": trust_level.value,
                "compliance_status": compliance_status,
                "managed_device": managed_device
            }
        ))
        
        return True
    
    def get_zero_trust_dashboard_data(self) -> Dict[str, Any]:
        """Get zero-trust dashboard data for monitoring."""
        with self._get_db_connection() as conn:
            dashboard_data = {
                "devices": {
                    "total": len(self.devices),
                    "by_trust_level": {},
                    "by_type": {},
                    "compliance_rate": 0
                },
                "access_decisions": {},
                "network_segments": len(self.network_segments),
                "policy_rules": len([r for r in self.policy_rules if r.enabled]),
                "recent_evaluations": []
            }
            
            # Device statistics
            trust_counts = {}
            type_counts = {}
            compliant_devices = 0
            
            for device in self.devices.values():
                trust_level = device.trust_level.value
                device_type = device.device_type.value
                
                trust_counts[trust_level] = trust_counts.get(trust_level, 0) + 1
                type_counts[device_type] = type_counts.get(device_type, 0) + 1
                
                if device.compliance_status:
                    compliant_devices += 1
            
            dashboard_data["devices"]["by_trust_level"] = trust_counts
            dashboard_data["devices"]["by_type"] = type_counts
            if len(self.devices) > 0:
                dashboard_data["devices"]["compliance_rate"] = round((compliant_devices / len(self.devices)) * 100, 1)
            
            # Access decision statistics (last 24 hours)
            cursor = conn.execute('''
                SELECT decision, COUNT(*) as count
                FROM access_evaluations
                WHERE evaluated_at > datetime('now', '-24 hours')
                GROUP BY decision
            ''')
            
            for row in cursor.fetchall():
                dashboard_data["access_decisions"][row['decision']] = row['count']
            
            # Recent evaluations
            cursor = conn.execute('''
                SELECT evaluation_id, user_id, device_id, resource, decision,
                       trust_score, risk_score, evaluated_at
                FROM access_evaluations
                ORDER BY evaluated_at DESC
                LIMIT 10
            ''')
            
            for row in cursor.fetchall():
                dashboard_data["recent_evaluations"].append({
                    "evaluation_id": row['evaluation_id'],
                    "user_id": row['user_id'],
                    "device_id": row['device_id'],
                    "resource": row['resource'],
                    "decision": row['decision'],
                    "trust_score": row['trust_score'],
                    "risk_score": row['risk_score'],
                    "evaluated_at": row['evaluated_at']
                })
        
        return dashboard_data


# Global zero-trust engine instance
_global_zerotrust_engine: Optional[ZeroTrustEngine] = None


def get_zerotrust_engine() -> ZeroTrustEngine:
    """Get the global zero-trust engine."""
    global _global_zerotrust_engine
    if _global_zerotrust_engine is None:
        _global_zerotrust_engine = ZeroTrustEngine()
    return _global_zerotrust_engine


def initialize_zerotrust_engine(db_path: str = "memmimic_zerotrust.db",
                               geoip_db_path: Optional[str] = None,
                               audit_logger: Optional[SecurityAuditLogger] = None) -> ZeroTrustEngine:
    """Initialize the global zero-trust engine."""
    global _global_zerotrust_engine
    _global_zerotrust_engine = ZeroTrustEngine(db_path, geoip_db_path, audit_logger)
    return _global_zerotrust_engine