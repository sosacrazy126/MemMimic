"""
Intelligent Alerting System for MemMimic v2.0

Threshold-based alerting and notification system with intelligent deduplication,
escalation policies, and automated remediation actions.
"""

import time
import threading
import json
import smtplib
import requests
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..errors import get_error_logger
from .monitor import PerformanceMonitor, ThresholdViolation, AlertSeverity, get_performance_monitor

logger = get_error_logger("telemetry.alerts")


class AlertChannel(Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    SLACK = "slack"
    TEAMS = "teams"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert instance with full lifecycle tracking"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    source_violation: Optional[ThresholdViolation]
    status: AlertStatus = AlertStatus.NEW
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notification_count: int = 0
    last_notification_at: Optional[datetime] = None
    escalation_level: int = 0
    suppression_reason: Optional[str] = None
    remediation_action: Optional[str] = None
    remediation_status: Optional[str] = None


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_type: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: Set[AlertSeverity] = field(default_factory=lambda: set(AlertSeverity))
    rate_limit_seconds: int = 60  # Minimum time between notifications
    last_notification: Optional[datetime] = None


@dataclass
class EscalationPolicy:
    """Alert escalation policy"""
    name: str
    severity_levels: Set[AlertSeverity]
    escalation_steps: List[Dict[str, Any]]  # List of escalation actions
    enabled: bool = True
    max_escalations: int = 3
    escalation_interval_minutes: int = 30


class AlertingSystem:
    """
    Intelligent Alerting System with advanced features.
    
    Features:
    - Multi-channel notification delivery
    - Intelligent alert deduplication
    - Escalation policies and workflows
    - Rate limiting and suppression
    - Automated remediation actions
    - Alert lifecycle management
    - Performance impact monitoring
    - Integration with external systems
    """
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None, config: Optional[Dict[str, Any]] = None):
        self.monitor = monitor or get_performance_monitor()
        self.config = config or {}
        
        # Configuration
        self.alert_retention_hours = self.config.get('alert_retention_hours', 72)
        self.deduplication_window_seconds = self.config.get('deduplication_window_seconds', 300)  # 5 minutes
        self.auto_resolve_after_minutes = self.config.get('auto_resolve_after_minutes', 60)
        self.max_alerts_per_minute = self.config.get('max_alerts_per_minute', 10)
        
        # Alert storage and tracking
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._alert_counter = 0
        self._rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Notification channels
        self._channels: Dict[str, NotificationChannel] = {}
        
        # Escalation policies
        self._escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Deduplication tracking
        self._dedup_keys: Dict[str, str] = {}  # Maps dedup key to alert ID
        self._dedup_cleanup_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._processing_thread = None
        self._stop_processing = threading.Event()
        
        # Performance tracking
        self._alert_metrics = {
            'alerts_created': 0,
            'alerts_resolved': 0,
            'notifications_sent': 0,
            'notifications_failed': 0,
            'escalations_triggered': 0,
            'alerts_suppressed': 0
        }
        
        # Initialize default channels and policies
        self._initialize_default_configuration()
        
        # Start background processing
        self._start_background_processing()
        
        logger.info("AlertingSystem initialized")
    
    def _initialize_default_configuration(self):
        """Initialize default notification channels and escalation policies"""
        # Default log channel (always enabled)
        self.add_notification_channel(
            "default_log",
            AlertChannel.LOG,
            config={},
            severity_filter={AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY}
        )
        
        # Default console channel for immediate visibility
        self.add_notification_channel(
            "default_console",
            AlertChannel.CONSOLE,
            config={},
            severity_filter={AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY},
            rate_limit_seconds=30
        )
        
        # Default escalation policy for critical issues
        self.add_escalation_policy(EscalationPolicy(
            name="critical_performance",
            severity_levels={AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY},
            escalation_steps=[
                {"action": "notify_console", "delay_minutes": 0},
                {"action": "notify_all_channels", "delay_minutes": 15},
                {"action": "trigger_remediation", "delay_minutes": 30}
            ],
            escalation_interval_minutes=15,
            max_escalations=3
        ))
        
        logger.info("Default alerting configuration initialized")
    
    def add_notification_channel(
        self,
        name: str,
        channel_type: AlertChannel,
        config: Dict[str, Any],
        severity_filter: Optional[Set[AlertSeverity]] = None,
        rate_limit_seconds: int = 60
    ) -> None:
        """Add notification channel"""
        if severity_filter is None:
            severity_filter = set(AlertSeverity)
        
        channel = NotificationChannel(
            channel_type=channel_type,
            config=config,
            severity_filter=severity_filter,
            rate_limit_seconds=rate_limit_seconds
        )
        
        with self._lock:
            self._channels[name] = channel
        
        logger.info(f"Added notification channel: {name} ({channel_type.value})")
    
    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add escalation policy"""
        with self._lock:
            self._escalation_policies[policy.name] = policy
        
        logger.info(f"Added escalation policy: {policy.name}")
    
    def create_alert_from_violation(self, violation: ThresholdViolation) -> Alert:
        """Create alert from threshold violation"""
        # Generate alert ID
        alert_id = f"alert_{int(time.time() * 1000)}_{self._alert_counter}"
        self._alert_counter += 1
        
        # Check for deduplication
        dedup_key = f"{violation.threshold.name}_{violation.threshold.operation}"
        
        with self._lock:
            # Check if we already have an active alert for this violation type
            if dedup_key in self._dedup_keys:
                existing_alert_id = self._dedup_keys[dedup_key]
                if existing_alert_id in self._alerts:
                    existing_alert = self._alerts[existing_alert_id]
                    if existing_alert.status not in [AlertStatus.RESOLVED]:
                        # Update existing alert instead of creating new one
                        existing_alert.updated_at = datetime.now()
                        existing_alert.notification_count += 1
                        existing_alert.metadata['violation_count'] = violation.violation_count
                        logger.debug(f"Updated existing alert {existing_alert_id} for dedup key {dedup_key}")
                        return existing_alert
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                title=f"Performance Threshold Violation: {violation.threshold.name}",
                message=violation.message,
                severity=violation.threshold.severity,
                source_violation=violation,
                tags={
                    violation.threshold.operation,
                    violation.threshold.threshold_type.value,
                    f"severity_{violation.threshold.severity.value}"
                },
                metadata={
                    'threshold_name': violation.threshold.name,
                    'operation': violation.threshold.operation,
                    'metric': violation.threshold.metric,
                    'current_value': violation.current_value,
                    'threshold_value': violation.threshold.threshold_value,
                    'violation_count': violation.violation_count,
                    'threshold_type': violation.threshold.threshold_type.value
                }
            )
            
            # Store alert
            self._alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._dedup_keys[dedup_key] = alert_id
            self._alert_metrics['alerts_created'] += 1
            
            logger.info(f"Created alert {alert_id}: {alert.title}")
            return alert
    
    def create_custom_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create custom alert"""
        alert_id = f"custom_alert_{int(time.time() * 1000)}_{self._alert_counter}"
        self._alert_counter += 1
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source_violation=None,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._alert_metrics['alerts_created'] += 1
        
        logger.info(f"Created custom alert {alert_id}: {title}")
        return alert
    
    def process_alert(self, alert: Alert) -> None:
        """Process alert through notification and escalation workflows"""
        current_time = datetime.now()
        
        # Check rate limiting
        if not self._check_rate_limit(alert):
            logger.debug(f"Alert {alert.id} rate limited")
            return
        
        # Send notifications
        self._send_notifications(alert)
        
        # Update alert tracking
        with self._lock:
            alert.last_notification_at = current_time
            alert.notification_count += 1
        
        # Check for escalation
        self._check_escalation(alert)
        
        # Log processing
        logger.info(f"Processed alert {alert.id} - notifications: {alert.notification_count}, escalation level: {alert.escalation_level}")
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert is within rate limits"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        rate_key = f"alerts_{alert.severity.value}"
        rate_queue = self._rate_limiter[rate_key]
        
        # Remove old entries
        while rate_queue and rate_queue[0] < minute_ago:
            rate_queue.popleft()
        
        # Check rate limit
        if len(rate_queue) >= self.max_alerts_per_minute:
            self._alert_metrics['alerts_suppressed'] += 1
            return False
        
        # Add current alert to rate tracker
        rate_queue.append(current_time)
        return True
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications through all appropriate channels"""
        notifications_sent = 0
        notifications_failed = 0
        
        with self._lock:
            for channel_name, channel in self._channels.items():
                if not channel.enabled:
                    continue
                
                # Check severity filter
                if alert.severity not in channel.severity_filter:
                    continue
                
                # Check rate limiting for channel
                if (channel.last_notification and 
                    (datetime.now() - channel.last_notification).total_seconds() < channel.rate_limit_seconds):
                    continue
                
                try:
                    success = self._send_notification_to_channel(alert, channel)
                    if success:
                        notifications_sent += 1
                        channel.last_notification = datetime.now()
                    else:
                        notifications_failed += 1
                
                except Exception as e:
                    logger.error(f"Failed to send notification to {channel_name}: {e}")
                    notifications_failed += 1
        
        # Update metrics
        self._alert_metrics['notifications_sent'] += notifications_sent
        self._alert_metrics['notifications_failed'] += notifications_failed
        
        if notifications_sent == 0 and notifications_failed > 0:
            logger.warning(f"All notifications failed for alert {alert.id}")
    
    def _send_notification_to_channel(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification to specific channel"""
        try:
            if channel.channel_type == AlertChannel.LOG:
                return self._send_log_notification(alert, channel)
            elif channel.channel_type == AlertChannel.CONSOLE:
                return self._send_console_notification(alert, channel)
            elif channel.channel_type == AlertChannel.EMAIL:
                return self._send_email_notification(alert, channel)
            elif channel.channel_type == AlertChannel.WEBHOOK:
                return self._send_webhook_notification(alert, channel)
            elif channel.channel_type == AlertChannel.SLACK:
                return self._send_slack_notification(alert, channel)
            else:
                logger.warning(f"Unsupported channel type: {channel.channel_type}")
                return False
        
        except Exception as e:
            logger.error(f"Channel notification failed ({channel.channel_type}): {e}")
            return False
    
    def _send_log_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification to log"""
        severity_map = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical
        }
        
        log_func = severity_map.get(alert.severity, logger.info)
        log_func(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
        return True
    
    def _send_console_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification to console"""
        severity_symbol = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }.get(alert.severity, "ℹ️")
        
        print(f"\n{severity_symbol} MEMMIMIC ALERT [{alert.severity.value.upper()}]")
        print(f"Title: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Time: {alert.created_at.isoformat()}")
        if alert.metadata:
            print(f"Details: {json.dumps(alert.metadata, indent=2)}")
        print("-" * 50)
        
        return True
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email notification"""
        config = channel.config
        
        if not all(key in config for key in ['smtp_server', 'smtp_port', 'username', 'password', 'to_emails']):
            logger.error("Email channel missing required configuration")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', config['username'])
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[MemMimic Alert] {alert.title}"
            
            body = f"""
Alert Details:
- Title: {alert.title}
- Severity: {alert.severity.value.upper()}
- Message: {alert.message}
- Created: {alert.created_at.isoformat()}
- Alert ID: {alert.id}

Additional Information:
{json.dumps(alert.metadata, indent=2)}

This is an automated alert from MemMimic v2.0 Telemetry System.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls', True):
                server.starttls()
            server.login(config['username'], config['password'])
            text = msg.as_string()
            server.sendmail(config['username'], config['to_emails'], text)
            server.quit()
            
            return True
        
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook notification"""
        config = channel.config
        
        if 'url' not in config:
            logger.error("Webhook channel missing URL configuration")
            return False
        
        try:
            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'created_at': alert.created_at.isoformat(),
                'tags': list(alert.tags),
                'metadata': alert.metadata
            }
            
            headers = config.get('headers', {'Content-Type': 'application/json'})
            timeout = config.get('timeout', 10)
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
    
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack notification"""
        config = channel.config
        
        if 'webhook_url' not in config:
            logger.error("Slack channel missing webhook_url configuration")
            return False
        
        try:
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "good"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Alert ID", "value": alert.id, "short": True},
                            {"title": "Created", "value": alert.created_at.isoformat(), "short": True}
                        ],
                        "timestamp": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    def _check_escalation(self, alert: Alert) -> None:
        """Check and trigger escalation if needed"""
        # Find applicable escalation policies
        applicable_policies = [
            policy for policy in self._escalation_policies.values()
            if policy.enabled and alert.severity in policy.severity_levels
        ]
        
        if not applicable_policies:
            return
        
        current_time = datetime.now()
        
        for policy in applicable_policies:
            # Check if escalation is due
            time_since_creation = (current_time - alert.created_at).total_seconds() / 60  # minutes
            
            escalation_due_time = alert.escalation_level * policy.escalation_interval_minutes
            
            if (time_since_creation >= escalation_due_time and 
                alert.escalation_level < policy.max_escalations and
                alert.status not in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED]):
                
                self._trigger_escalation(alert, policy)
    
    def _trigger_escalation(self, alert: Alert, policy: EscalationPolicy) -> None:
        """Trigger escalation action"""
        if alert.escalation_level >= len(policy.escalation_steps):
            return
        
        escalation_step = policy.escalation_steps[alert.escalation_level]
        
        try:
            action = escalation_step.get('action', 'notify_all_channels')
            
            if action == 'notify_console':
                self._send_console_notification(alert, NotificationChannel(AlertChannel.CONSOLE, {}))
            elif action == 'notify_all_channels':
                self._send_notifications(alert)
            elif action == 'trigger_remediation':
                self._trigger_remediation_action(alert)
            
            # Update escalation level
            with self._lock:
                alert.escalation_level += 1
                alert.updated_at = datetime.now()
                alert.status = AlertStatus.ESCALATED
            
            self._alert_metrics['escalations_triggered'] += 1
            
            logger.warning(f"Escalated alert {alert.id} to level {alert.escalation_level} using policy {policy.name}")
        
        except Exception as e:
            logger.error(f"Escalation failed for alert {alert.id}: {e}")
    
    def _trigger_remediation_action(self, alert: Alert) -> None:
        """Trigger automated remediation action"""
        if not alert.source_violation:
            return
        
        threshold = alert.source_violation.threshold
        
        # Simple remediation actions based on threshold type
        if threshold.operation == "system_resources":
            if "memory" in threshold.metric:
                alert.remediation_action = "memory_cleanup_suggested"
                alert.remediation_status = "manual_intervention_required"
                logger.info(f"Remediation suggested for {alert.id}: memory cleanup required")
            elif "cpu" in threshold.metric:
                alert.remediation_action = "cpu_optimization_suggested" 
                alert.remediation_status = "manual_intervention_required"
                logger.info(f"Remediation suggested for {alert.id}: CPU optimization required")
        
        elif "telemetry_overhead" in threshold.operation:
            alert.remediation_action = "telemetry_optimization_needed"
            alert.remediation_status = "automatic_optimization_attempted"
            # Could trigger actual optimization here
            logger.info(f"Remediation triggered for {alert.id}: telemetry optimization")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge alert"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                alert.updated_at = datetime.now()
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve alert"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                alert.updated_at = datetime.now()
                
                # Clean up deduplication tracking
                if alert.source_violation:
                    dedup_key = f"{alert.source_violation.threshold.name}_{alert.source_violation.threshold.operation}"
                    if dedup_key in self._dedup_keys and self._dedup_keys[dedup_key] == alert_id:
                        del self._dedup_keys[dedup_key]
                
                self._alert_metrics['alerts_resolved'] += 1
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False
    
    def get_active_alerts(self, severity_filter: Optional[Set[AlertSeverity]] = None) -> List[Alert]:
        """Get currently active alerts"""
        active_statuses = {AlertStatus.NEW, AlertStatus.ACKNOWLEDGED, AlertStatus.IN_PROGRESS, AlertStatus.ESCALATED}
        
        with self._lock:
            active_alerts = [
                alert for alert in self._alerts.values()
                if alert.status in active_statuses
            ]
            
            if severity_filter:
                active_alerts = [alert for alert in active_alerts if alert.severity in severity_filter]
        
        return sorted(active_alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary"""
        current_time = datetime.now()
        
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            summary = {
                'timestamp': current_time.isoformat(),
                'active_alerts': len(active_alerts),
                'alerts_by_severity': {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                'alerts_by_status': {
                    status.value: len([a for a in self._alerts.values() if a.status == status])
                    for status in AlertStatus
                },
                'notification_channels': len(self._channels),
                'escalation_policies': len(self._escalation_policies),
                'metrics': self._alert_metrics.copy(),
                'recent_alerts': [
                    {
                        'id': alert.id,
                        'title': alert.title,
                        'severity': alert.severity.value,
                        'status': alert.status.value,
                        'created_at': alert.created_at.isoformat(),
                        'escalation_level': alert.escalation_level
                    }
                    for alert in active_alerts[:10]  # Most recent 10
                ]
            }
        
        return summary
    
    def _start_background_processing(self):
        """Start background alert processing"""
        def processing_worker():
            while not self._stop_processing.wait(30):  # Check every 30 seconds
                try:
                    current_time = datetime.now()
                    
                    # Check for threshold violations from monitor
                    violations = self.monitor.check_thresholds()
                    for violation in violations:
                        alert = self.create_alert_from_violation(violation)
                        self.process_alert(alert)
                    
                    # Auto-resolve alerts that are no longer relevant
                    self._auto_resolve_alerts(current_time)
                    
                    # Clean up old data
                    self._cleanup_old_data(current_time)
                
                except Exception as e:
                    logger.error(f"Background alert processing failed: {e}")
        
        self._processing_thread = threading.Thread(target=processing_worker, daemon=True)
        self._processing_thread.start()
        logger.info("Background alert processing started")
    
    def _auto_resolve_alerts(self, current_time: datetime):
        """Auto-resolve alerts that are no longer relevant"""
        resolution_threshold = timedelta(minutes=self.auto_resolve_after_minutes)
        
        with self._lock:
            alerts_to_resolve = []
            
            for alert in self._alerts.values():
                if (alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED] and
                    current_time - alert.created_at > resolution_threshold):
                    
                    # Check if the underlying issue is resolved
                    if alert.source_violation:
                        # Get current metric value
                        current_value = self.monitor._get_current_metric_value(alert.source_violation.threshold)
                        if current_value is not None:
                            threshold = alert.source_violation.threshold
                            is_still_violated = False
                            
                            if threshold.threshold_type.value in ['performance', 'resource']:
                                is_still_violated = current_value > threshold.threshold_value
                            elif threshold.threshold_type.value == 'success_rate':
                                is_still_violated = current_value < threshold.threshold_value
                            
                            if not is_still_violated:
                                alerts_to_resolve.append(alert.id)
            
            # Resolve alerts
            for alert_id in alerts_to_resolve:
                self.resolve_alert(alert_id, "auto_resolution")
    
    def _cleanup_old_data(self, current_time: datetime):
        """Clean up old alerts and data"""
        retention_threshold = current_time - timedelta(hours=self.alert_retention_hours)
        
        with self._lock:
            # Remove old resolved alerts
            alerts_to_remove = [
                alert_id for alert_id, alert in self._alerts.items()
                if (alert.status == AlertStatus.RESOLVED and 
                    alert.resolved_at and 
                    alert.resolved_at < retention_threshold)
            ]
            
            for alert_id in alerts_to_remove:
                del self._alerts[alert_id]
            
            # Clean up deduplication keys
            dedup_cleanup_threshold = time.time() - self.deduplication_window_seconds
            if current_time.timestamp() - self._dedup_cleanup_time > 300:  # Clean every 5 minutes
                stale_keys = [
                    key for key, alert_id in self._dedup_keys.items()
                    if alert_id not in self._alerts
                ]
                for key in stale_keys:
                    del self._dedup_keys[key]
                
                self._dedup_cleanup_time = current_time.timestamp()
            
            if alerts_to_remove:
                logger.debug(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def shutdown(self):
        """Shutdown alerting system"""
        try:
            self._stop_processing.set()
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
            
            logger.info("AlertingSystem shutdown completed")
            
        except Exception as e:
            logger.error(f"AlertingSystem shutdown failed: {e}")


# Global alerting system instance
_global_alerting: Optional[AlertingSystem] = None
_alerting_lock = threading.Lock()


def get_alerting_system() -> AlertingSystem:
    """Get global alerting system instance"""
    global _global_alerting
    
    if _global_alerting is None:
        with _alerting_lock:
            if _global_alerting is None:
                _global_alerting = AlertingSystem()
    
    return _global_alerting