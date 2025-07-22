"""
Tamper Detector - Real-time tamper detection and alerting system.

Provides continuous monitoring and automated detection of tampering attempts
with configurable alerting and response mechanisms.
"""

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..errors import get_error_logger, handle_errors
from .cryptographic_verifier import VerificationResult, VerificationStatus, HashChainVerificationResult

logger = get_error_logger(__name__)


class TamperSeverity(Enum):
    """Tamper detection severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TamperType(Enum):
    """Types of tampering detected."""
    HASH_MISMATCH = "hash_mismatch"
    CHAIN_BREAK = "chain_break"
    MISSING_ENTRY = "missing_entry"
    TEMPORAL_VIOLATION = "temporal_violation"
    SIGNATURE_INVALID = "signature_invalid"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BULK_MODIFICATION = "bulk_modification"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class TamperAlert:
    """Tamper detection alert with detailed information."""
    
    alert_id: str = field(default_factory=lambda: f"alert_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Tamper details
    tamper_type: TamperType = TamperType.HASH_MISMATCH
    severity: TamperSeverity = TamperSeverity.MEDIUM
    description: str = ""
    
    # Affected entries
    affected_entries: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    component: str = "audit_system"
    detection_method: str = "automatic"
    confidence_score: float = 1.0  # 0.0 to 1.0
    
    # Response tracking
    acknowledged: bool = False
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'tamper_type': self.tamper_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'affected_entries': self.affected_entries,
            'evidence': self.evidence,
            'component': self.component,
            'detection_method': self.detection_method,
            'confidence_score': self.confidence_score,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'response_actions': self.response_actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TamperAlert':
        """Create TamperAlert from dictionary."""
        # Convert enum fields
        if isinstance(data.get('tamper_type'), str):
            data['tamper_type'] = TamperType(data['tamper_type'])
        if isinstance(data.get('severity'), str):
            data['severity'] = TamperSeverity(data['severity'])
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


@dataclass
class TamperPattern:
    """Pattern definition for tamper detection."""
    
    name: str
    tamper_types: List[TamperType]
    threshold: int  # Number of events to trigger
    time_window_seconds: int
    severity: TamperSeverity
    description: str = ""
    enabled: bool = True
    
    # Advanced pattern matching
    require_same_entry: bool = False  # Must be same entry
    require_same_component: bool = False  # Must be same component
    escalation_multiplier: float = 2.0  # Severity escalation factor


class TamperDetector:
    """
    Real-time tamper detection system with pattern-based alerting.
    
    Features:
    - Real-time hash verification monitoring
    - Pattern-based tamper detection
    - Automated alerting and escalation
    - Configurable response actions
    - Performance-optimized detection (<1ms overhead)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tamper detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Alert storage and management
        self._alerts: deque = deque(maxlen=self.config.get('max_alerts', 10000))
        self._alert_callbacks: List[Callable[[TamperAlert], None]] = []
        self._alerts_lock = threading.RLock()
        
        # Detection patterns
        self._patterns = self._initialize_patterns()
        
        # Event tracking for pattern detection
        self._event_history: deque = deque(maxlen=self.config.get('event_history_size', 5000))
        self._event_lock = threading.RLock()
        
        # Performance metrics
        self._metrics = {
            'total_detections': 0,
            'alerts_generated': 0,
            'pattern_matches': 0,
            'false_positives': 0,
            'avg_detection_time_ms': 0.0
        }
        
        # Component tracking
        self._component_status: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'last_tamper': None,
            'tamper_count': 0,
            'status': 'healthy'
        })
        
        # Automatic cleanup thread
        self._start_cleanup_thread()
        
        logger.info("TamperDetector initialized with {} patterns".format(len(self._patterns)))
    
    def _initialize_patterns(self) -> Dict[str, TamperPattern]:
        """Initialize tamper detection patterns."""
        default_patterns = {
            'hash_verification_failures': TamperPattern(
                name='hash_verification_failures',
                tamper_types=[TamperType.HASH_MISMATCH],
                threshold=3,
                time_window_seconds=300,  # 5 minutes
                severity=TamperSeverity.HIGH,
                description="Multiple hash verification failures detected",
                require_same_entry=False
            ),
            'chain_break_cluster': TamperPattern(
                name='chain_break_cluster',
                tamper_types=[TamperType.CHAIN_BREAK],
                threshold=2,
                time_window_seconds=60,
                severity=TamperSeverity.CRITICAL,
                description="Hash chain integrity compromised",
                require_same_component=True
            ),
            'bulk_tampering': TamperPattern(
                name='bulk_tampering',
                tamper_types=[TamperType.HASH_MISMATCH, TamperType.MISSING_ENTRY],
                threshold=10,
                time_window_seconds=600,  # 10 minutes
                severity=TamperSeverity.CRITICAL,
                description="Bulk tampering attempt detected",
                escalation_multiplier=3.0
            ),
            'temporal_anomalies': TamperPattern(
                name='temporal_anomalies',
                tamper_types=[TamperType.TEMPORAL_VIOLATION],
                threshold=5,
                time_window_seconds=300,
                severity=TamperSeverity.MEDIUM,
                description="Temporal ordering violations detected"
            ),
            'signature_failures': TamperPattern(
                name='signature_failures',
                tamper_types=[TamperType.SIGNATURE_INVALID],
                threshold=2,
                time_window_seconds=120,
                severity=TamperSeverity.HIGH,
                description="Multiple signature verification failures"
            )
        }
        
        # Allow custom patterns from config
        custom_patterns = self.config.get('custom_patterns', {})
        for name, pattern_data in custom_patterns.items():
            try:
                # Convert string enums to actual enums
                tamper_types = [TamperType(t) for t in pattern_data.get('tamper_types', [])]
                severity = TamperSeverity(pattern_data.get('severity', 'medium'))
                
                default_patterns[name] = TamperPattern(
                    name=name,
                    tamper_types=tamper_types,
                    threshold=pattern_data.get('threshold', 5),
                    time_window_seconds=pattern_data.get('time_window_seconds', 300),
                    severity=severity,
                    description=pattern_data.get('description', f"Custom pattern: {name}"),
                    enabled=pattern_data.get('enabled', True)
                )
            except Exception as e:
                logger.warning(f"Failed to load custom pattern {name}: {e}")
        
        return default_patterns
    
    def detect_from_verification_result(self, result: VerificationResult) -> Optional[TamperAlert]:
        """
        Process verification result and detect tampering.
        
        Args:
            result: Verification result to analyze
            
        Returns:
            TamperAlert if tampering detected, None otherwise
        """
        start_time = time.perf_counter()
        
        try:
            # Record event for pattern analysis
            self._record_verification_event(result)
            
            alert = None
            
            # Direct tamper detection from verification result
            if not result.is_valid():
                alert = self._create_alert_from_verification(result)
            
            # Check for pattern matches
            pattern_alert = self._check_patterns()
            if pattern_alert and (not alert or pattern_alert.severity.value > alert.severity.value):
                alert = pattern_alert
            
            # Generate alert if tampering detected
            if alert:
                self._generate_alert(alert)
            
            # Update metrics
            detection_time = (time.perf_counter() - start_time) * 1000
            self._update_detection_metrics(detection_time, alert is not None)
            
            return alert
            
        except Exception as e:
            logger.error(f"Tamper detection error: {e}")
            return None
    
    def detect_from_chain_verification(self, result: HashChainVerificationResult) -> List[TamperAlert]:
        """
        Process hash chain verification result and detect tampering.
        
        Args:
            result: Hash chain verification result
            
        Returns:
            List of TamperAlert objects for detected issues
        """
        alerts = []
        
        try:
            if not result.valid and result.broken_links:
                for broken_link in result.broken_links:
                    alert = TamperAlert(
                        tamper_type=TamperType.CHAIN_BREAK,
                        severity=self._determine_chain_break_severity(broken_link, result),
                        description=f"Hash chain break detected: {broken_link.get('details', 'Unknown')}",
                        affected_entries=[broken_link.get('entry_id', 'unknown')],
                        evidence={
                            'chain_verification_result': result.to_dict(),
                            'broken_link': broken_link,
                            'integrity_percentage': result.integrity_percentage()
                        },
                        component='hash_chain_verifier',
                        confidence_score=min(1.0, len(result.broken_links) / max(1, result.total_entries))
                    )
                    
                    alerts.append(alert)
                    self._generate_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Chain tamper detection error: {e}")
            return alerts
    
    def _record_verification_event(self, result: VerificationResult) -> None:
        """Record verification event for pattern analysis."""
        with self._event_lock:
            event = {
                'timestamp': result.timestamp,
                'entry_id': result.entry_id,
                'status': result.status,
                'verification_time': result.verification_time_ms,
                'component': result.metadata.get('component', 'unknown')
            }
            
            self._event_history.append(event)
    
    def _create_alert_from_verification(self, result: VerificationResult) -> TamperAlert:
        """Create tamper alert from verification result."""
        tamper_type_mapping = {
            VerificationStatus.TAMPERED: TamperType.HASH_MISMATCH,
            VerificationStatus.MISSING: TamperType.MISSING_ENTRY,
            VerificationStatus.INVALID: TamperType.SIGNATURE_INVALID,
            VerificationStatus.ERROR: TamperType.SUSPICIOUS_PATTERN
        }
        
        tamper_type = tamper_type_mapping.get(result.status, TamperType.HASH_MISMATCH)
        severity = self._determine_severity_from_verification(result)
        
        return TamperAlert(
            tamper_type=tamper_type,
            severity=severity,
            description=f"Verification failed: {result.details}",
            affected_entries=[result.entry_id],
            evidence={
                'verification_result': result.to_dict(),
                'verification_time_ms': result.verification_time_ms
            },
            component='cryptographic_verifier',
            confidence_score=0.9  # High confidence from crypto verification
        )
    
    def _determine_severity_from_verification(self, result: VerificationResult) -> TamperSeverity:
        """Determine severity level from verification result."""
        if result.status == VerificationStatus.TAMPERED:
            return TamperSeverity.HIGH
        elif result.status == VerificationStatus.MISSING:
            return TamperSeverity.MEDIUM
        elif result.status == VerificationStatus.INVALID:
            return TamperSeverity.HIGH
        else:
            return TamperSeverity.LOW
    
    def _determine_chain_break_severity(
        self,
        broken_link: Dict[str, Any],
        result: HashChainVerificationResult
    ) -> TamperSeverity:
        """Determine severity of chain break based on context."""
        integrity_percentage = result.integrity_percentage()
        
        if integrity_percentage < 50:
            return TamperSeverity.CRITICAL
        elif integrity_percentage < 80:
            return TamperSeverity.HIGH
        elif integrity_percentage < 95:
            return TamperSeverity.MEDIUM
        else:
            return TamperSeverity.LOW
    
    def _check_patterns(self) -> Optional[TamperAlert]:
        """Check for pattern matches in recent events."""
        current_time = datetime.now(timezone.utc)
        
        for pattern_name, pattern in self._patterns.items():
            if not pattern.enabled:
                continue
            
            # Get events within time window
            window_start = current_time - timedelta(seconds=pattern.time_window_seconds)
            
            with self._event_lock:
                relevant_events = [
                    event for event in self._event_history
                    if (event['timestamp'] >= window_start and
                        any(event['status'].name.lower() == tt.value.replace('_', '_') 
                            for tt in pattern.tamper_types))
                ]
            
            # Apply pattern-specific filters
            if pattern.require_same_entry:
                # Group by entry_id and check if any entry exceeds threshold
                entry_groups = defaultdict(list)
                for event in relevant_events:
                    entry_groups[event['entry_id']].append(event)
                
                for entry_id, events in entry_groups.items():
                    if len(events) >= pattern.threshold:
                        return self._create_pattern_alert(pattern, events, entry_id)
            
            elif pattern.require_same_component:
                # Group by component
                component_groups = defaultdict(list)
                for event in relevant_events:
                    component_groups[event['component']].append(event)
                
                for component, events in component_groups.items():
                    if len(events) >= pattern.threshold:
                        return self._create_pattern_alert(pattern, events, component=component)
            
            else:
                # Check total count across all events
                if len(relevant_events) >= pattern.threshold:
                    return self._create_pattern_alert(pattern, relevant_events)
        
        return None
    
    def _create_pattern_alert(
        self,
        pattern: TamperPattern,
        events: List[Dict[str, Any]],
        entry_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> TamperAlert:
        """Create alert from pattern match."""
        affected_entries = list(set(event['entry_id'] for event in events))
        
        # Calculate escalated severity if applicable
        severity = pattern.severity
        if len(events) > pattern.threshold * pattern.escalation_multiplier:
            severity_levels = list(TamperSeverity)
            current_index = severity_levels.index(severity)
            if current_index < len(severity_levels) - 1:
                severity = severity_levels[current_index + 1]
        
        description = f"{pattern.description} ({len(events)} events in {pattern.time_window_seconds}s)"
        
        return TamperAlert(
            tamper_type=pattern.tamper_types[0],  # Primary tamper type
            severity=severity,
            description=description,
            affected_entries=affected_entries,
            evidence={
                'pattern_name': pattern.name,
                'event_count': len(events),
                'threshold': pattern.threshold,
                'time_window_seconds': pattern.time_window_seconds,
                'events': events[-10:]  # Last 10 events for evidence
            },
            component=component or 'pattern_detector',
            detection_method='pattern_matching',
            confidence_score=min(1.0, len(events) / pattern.threshold)
        )
    
    def _generate_alert(self, alert: TamperAlert) -> None:
        """Generate and process tamper alert."""
        with self._alerts_lock:
            self._alerts.append(alert)
            self._metrics['alerts_generated'] += 1
        
        # Update component status
        for entry_id in alert.affected_entries:
            component = alert.component
            self._component_status[component]['last_tamper'] = alert.timestamp
            self._component_status[component]['tamper_count'] += 1
            
            # Update component health status
            if alert.severity in [TamperSeverity.HIGH, TamperSeverity.CRITICAL]:
                self._component_status[component]['status'] = 'compromised'
            elif alert.severity == TamperSeverity.MEDIUM:
                self._component_status[component]['status'] = 'degraded'
        
        # Execute alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Log alert
        logger.warning(
            f"TAMPER ALERT: {alert.tamper_type.value} - {alert.description} "
            f"(severity={alert.severity.value}, confidence={alert.confidence_score:.2f})"
        )
    
    def register_alert_callback(self, callback: Callable[[TamperAlert], None]) -> None:
        """Register callback function for alert notifications."""
        self._alert_callbacks.append(callback)
        logger.info(f"Alert callback registered: {callback.__name__}")
    
    def get_recent_alerts(
        self,
        severity: Optional[TamperSeverity] = None,
        hours: int = 24,
        limit: Optional[int] = None
    ) -> List[TamperAlert]:
        """
        Get recent tamper alerts with optional filtering.
        
        Args:
            severity: Filter by severity level
            hours: Hours to look back
            limit: Maximum number of alerts to return
            
        Returns:
            List of matching TamperAlert objects
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._alerts_lock:
            alerts = [
                alert for alert in self._alerts
                if alert.timestamp >= cutoff_time
            ]
        
        # Apply severity filter
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, response_action: str = "") -> bool:
        """
        Acknowledge a tamper alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            response_action: Description of response action taken
            
        Returns:
            True if alert was found and acknowledged, False otherwise
        """
        with self._alerts_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    if response_action:
                        alert.response_actions.append(response_action)
                    
                    logger.info(f"Alert acknowledged: {alert_id} - {response_action}")
                    return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """
        Mark a tamper alert as resolved.
        
        Args:
            alert_id: Alert ID to resolve
            resolution: Description of resolution
            
        Returns:
            True if alert was found and resolved, False otherwise
        """
        with self._alerts_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.acknowledged = True
                    if resolution:
                        alert.response_actions.append(f"RESOLVED: {resolution}")
                    
                    logger.info(f"Alert resolved: {alert_id} - {resolution}")
                    return True
        
        return False
    
    def get_component_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all monitored components."""
        return dict(self._component_status)
    
    def _update_detection_metrics(self, detection_time: float, alert_generated: bool) -> None:
        """Update tamper detection performance metrics."""
        self._metrics['total_detections'] += 1
        
        if alert_generated:
            self._metrics['pattern_matches'] += 1
        
        # Update average detection time
        current_avg = self._metrics['avg_detection_time_ms']
        count = self._metrics['total_detections']
        self._metrics['avg_detection_time_ms'] = ((current_avg * (count - 1)) + detection_time) / count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tamper detection performance metrics."""
        with self._alerts_lock:
            unresolved_alerts = sum(1 for alert in self._alerts if not alert.resolved)
            critical_alerts = sum(
                1 for alert in self._alerts 
                if alert.severity == TamperSeverity.CRITICAL and not alert.resolved
            )
        
        return {
            **self._metrics,
            'active_patterns': len([p for p in self._patterns.values() if p.enabled]),
            'total_patterns': len(self._patterns),
            'unresolved_alerts': unresolved_alerts,
            'critical_alerts': critical_alerts,
            'event_history_size': len(self._event_history),
            'component_count': len(self._component_status)
        }
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for cleaning up old events and alerts."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.debug("Tamper detector cleanup thread started")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old events and resolved alerts."""
        try:
            retention_hours = self.config.get('data_retention_hours', 168)  # 7 days default
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
            
            # Clean up resolved alerts
            with self._alerts_lock:
                initial_alert_count = len(self._alerts)
                alerts_to_keep = [
                    alert for alert in self._alerts
                    if not alert.resolved or alert.timestamp >= cutoff_time
                ]
                
                self._alerts.clear()
                self._alerts.extend(alerts_to_keep)
                
                cleaned_alerts = initial_alert_count - len(self._alerts)
            
            # Clean up old events
            with self._event_lock:
                initial_event_count = len(self._event_history)
                events_to_keep = [
                    event for event in self._event_history
                    if event['timestamp'] >= cutoff_time
                ]
                
                self._event_history.clear()
                self._event_history.extend(events_to_keep)
                
                cleaned_events = initial_event_count - len(self._event_history)
            
            if cleaned_alerts > 0 or cleaned_events > 0:
                logger.info(f"Cleaned up {cleaned_alerts} old alerts and {cleaned_events} old events")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")