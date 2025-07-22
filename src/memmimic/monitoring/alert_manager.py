"""
Enterprise Alert Management System for MemMimic

Provides intelligent alerting, notification routing, escalation policies,
and alert correlation to minimize noise while ensuring critical issues are addressed.
"""

import time
import threading
import logging
import smtplib
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    source: str  # Component that generated the alert
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'labels': self.labels,
            'annotations': self.annotations,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: str  # Condition expression
    severity: AlertSeverity
    threshold: float
    duration_seconds: int = 0  # How long condition must be true
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    # Notification settings
    notify_channels: List[str] = field(default_factory=list)
    escalation_minutes: int = 30
    max_notifications: int = 5


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    channel_type: str  # email, webhook, slack, etc.
    config: Dict[str, Any]
    enabled: bool = True


class AlertEvaluator:
    """Evaluates metric values against alert rules"""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        
    def evaluate_rule(self, rule: AlertRule) -> Optional[Alert]:
        """Evaluate an alert rule against current metrics"""
        try:
            # Parse condition (simplified - could be extended with full expression parser)
            if '>' in rule.condition:
                metric_name, threshold_str = rule.condition.split('>')
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())
                operator = 'gt'
            elif '<' in rule.condition:
                metric_name, threshold_str = rule.condition.split('<')
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())
                operator = 'lt'
            else:
                logger.warning(f"Unsupported alert condition: {rule.condition}")
                return None
            
            # Get current metric value
            current_value = self._get_metric_value(metric_name)
            if current_value is None:
                return None
            
            # Evaluate condition
            condition_met = False
            if operator == 'gt' and current_value > threshold:
                condition_met = True
            elif operator == 'lt' and current_value < threshold:
                condition_met = True
            
            if condition_met:
                # Create alert
                alert_id = f"{rule.rule_id}_{int(time.time())}"
                
                return Alert(
                    alert_id=alert_id,
                    name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.name}: {metric_name} is {current_value} (threshold: {threshold})",
                    source="alert_manager",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=threshold,
                    labels=rule.labels.copy(),
                    annotations=rule.annotations.copy()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {rule.rule_id}: {e}")
            return None
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        # Try gauge first
        gauge = self.metrics_collector.get_gauge(metric_name)
        if gauge:
            return gauge.get()
        
        # Try counter
        counter = self.metrics_collector.get_counter(metric_name)
        if counter:
            return counter.get()
        
        # Try histogram (get count)
        histogram = self.metrics_collector.get_histogram(metric_name)
        if histogram:
            return float(histogram.get_count())
        
        return None


class NotificationManager:
    """Manages alert notifications across different channels"""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.notification_history: deque = deque(maxlen=1000)
        
    def add_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.channel_id} ({channel.channel_type})")
    
    def remove_channel(self, channel_id: str):
        """Remove notification channel"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            logger.info(f"Removed notification channel: {channel_id}")
    
    def send_notification(self, alert: Alert, channel_ids: List[str]):
        """Send alert notification to specified channels"""
        for channel_id in channel_ids:
            if channel_id not in self.channels:
                logger.warning(f"Unknown notification channel: {channel_id}")
                continue
            
            channel = self.channels[channel_id]
            if not channel.enabled:
                continue
            
            try:
                success = self._send_to_channel(alert, channel)
                
                # Record notification
                self.notification_history.append({
                    'alert_id': alert.alert_id,
                    'channel_id': channel_id,
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                })
                
                if success:
                    logger.info(f"Alert {alert.alert_id} sent to {channel_id}")
                else:
                    logger.error(f"Failed to send alert {alert.alert_id} to {channel_id}")
                    
            except Exception as e:
                logger.error(f"Error sending alert {alert.alert_id} to {channel_id}: {e}")
    
    def _send_to_channel(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert to specific channel"""
        if channel.channel_type == "email":
            return self._send_email(alert, channel.config)
        elif channel.channel_type == "webhook":
            return self._send_webhook(alert, channel.config)
        elif channel.channel_type == "log":
            return self._send_log(alert, channel.config)
        else:
            logger.warning(f"Unsupported channel type: {channel.channel_type}")
            return False
    
    def _send_email(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification"""
        try:
            smtp_server = config.get('smtp_server', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            to_addresses = config.get('to_addresses', [])
            from_address = config.get('from_address', 'memmimic@localhost')
            
            if not to_addresses:
                logger.warning("No email addresses configured for email channel")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_address
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = f"MemMimic Alert: {alert.name} ({alert.severity.value.upper()})"
            
            # Email body
            body = f"""
MemMimic Alert Notification

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Message: {alert.message}
Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Metric Details:
- Metric Name: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}

Alert ID: {alert.alert_id}

This is an automated message from MemMimic monitoring system.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_webhook(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        try:
            import requests
            
            url = config.get('url')
            if not url:
                logger.warning("No webhook URL configured")
                return False
            
            headers = config.get('headers', {})
            timeout = config.get('timeout', 10)
            
            # Prepare payload
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'memmimic_monitoring'
            }
            
            # Send webhook
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _send_log(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send log notification"""
        try:
            log_level = config.get('level', 'warning').upper()
            log_message = f"ALERT: {alert.name} ({alert.severity.value}) - {alert.message}"
            
            if log_level == 'CRITICAL':
                logger.critical(log_message)
            elif log_level == 'ERROR':
                logger.error(log_message)
            elif log_level == 'WARNING':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send log notification: {e}")
            return False


class AlertManager:
    """
    Enterprise alert management system for MemMimic.
    
    Provides intelligent alerting with rule evaluation, notification routing,
    escalation policies, and alert correlation.
    """
    
    def __init__(self, metrics_collector, evaluation_interval: float = 30.0):
        self.metrics_collector = metrics_collector
        self.evaluation_interval = evaluation_interval
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=5000)
        
        # Alert evaluator and notifications
        self.evaluator = AlertEvaluator(metrics_collector)
        self.notification_manager = NotificationManager()
        
        # Alert correlation and suppression
        self.suppressed_alerts: Set[str] = set()
        self.alert_groups: Dict[str, List[str]] = {}  # Group alerts by similar characteristics
        
        # Background processing
        self._evaluation_thread = None
        self._stop_evaluation = threading.Event()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'notifications_sent': 0,
            'alerts_resolved': 0
        }
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        # Initialize default notification channels
        self._initialize_default_channels()
        
        self._start_evaluation_loop()
        
        logger.info("Alert Manager initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules for MemMimic"""
        default_rules = [
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                condition="memmimic_memory_usage_percent > 80",
                severity=AlertSeverity.WARNING,
                threshold=80,
                duration_seconds=300,  # 5 minutes
                annotations={
                    'description': 'Memory usage is above 80%',
                    'runbook': 'Check for memory leaks or increase memory allocation'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="critical_memory_usage",
                name="Critical Memory Usage",
                condition="memmimic_memory_usage_percent > 95",
                severity=AlertSeverity.CRITICAL,
                threshold=95,
                duration_seconds=60,  # 1 minute
                annotations={
                    'description': 'Memory usage is critically high',
                    'runbook': 'Immediate action required - system may crash'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                condition="memmimic_errors_total > 10",
                severity=AlertSeverity.ERROR,
                threshold=10,
                duration_seconds=300,
                annotations={
                    'description': 'Error rate is elevated',
                    'runbook': 'Check system logs for error patterns'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="low_health_score",
                name="Low System Health Score",
                condition="memmimic_health_score < 0.7",
                severity=AlertSeverity.WARNING,
                threshold=0.7,
                duration_seconds=600,  # 10 minutes
                annotations={
                    'description': 'System health score is below acceptable threshold',
                    'runbook': 'Check individual health check components'
                },
                notify_channels=["default_log"]
            ),
            AlertRule(
                rule_id="database_slow",
                name="Database Response Time High",
                condition="memmimic_db_query_duration_seconds > 1.0",
                severity=AlertSeverity.WARNING,
                threshold=1.0,
                duration_seconds=300,
                annotations={
                    'description': 'Database queries are taking too long',
                    'runbook': 'Check database performance and optimize queries'
                },
                notify_channels=["default_log"]
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        # Default log channel
        log_channel = NotificationChannel(
            channel_id="default_log",
            channel_type="log",
            config={'level': 'warning'}
        )
        self.notification_manager.add_channel(log_channel)
        
        logger.info("Default notification channels initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
    
    def update_alert_rule(self, rule: AlertRule):
        """Update existing alert rule"""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Updated alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_manager.add_channel(channel)
    
    def fire_alert(self, alert: Alert):
        """Manually fire an alert"""
        with self._lock:
            # Check if alert should be suppressed
            if self._should_suppress_alert(alert):
                logger.debug(f"Alert suppressed: {alert.alert_id}")
                return
            
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_severity'][alert.severity.value] += 1
            
            # Update metrics
            self.metrics_collector.increment_counter("memmimic_alerts_total")
            self.metrics_collector.increment_counter(f"memmimic_alerts_{alert.severity.value}_total")
            
            # Send notifications
            if alert.alert_id in self.alert_rules:
                rule = self.alert_rules[alert.alert_id]
                self.notification_manager.send_notification(alert, rule.notify_channels)
                self.stats['notifications_sent'] += 1
            
            logger.info(f"Alert fired: {alert.name} ({alert.severity.value})")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                alert.updated_at = datetime.now()
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.updated_at = datetime.now()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.stats['alerts_resolved'] += 1
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60):
        """Suppress alert for specified duration"""
        with self._lock:
            self.suppressed_alerts.add(alert_id)
            
            # Schedule unsuppression (simplified - could use proper scheduler)
            def unsuppress():
                time.sleep(duration_minutes * 60)
                with self._lock:
                    self.suppressed_alerts.discard(alert_id)
                logger.info(f"Alert unsuppressed: {alert_id}")
            
            threading.Thread(target=unsuppress, daemon=True).start()
            logger.info(f"Alert suppressed for {duration_minutes} minutes: {alert_id}")
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Check direct suppression
        if alert.alert_id in self.suppressed_alerts:
            return True
        
        # Check for similar active alerts (deduplication)
        similar_key = f"{alert.name}_{alert.source}_{alert.metric_name}"
        for active_alert in self.active_alerts.values():
            active_key = f"{active_alert.name}_{active_alert.source}_{active_alert.metric_name}"
            if similar_key == active_key:
                # Similar alert already active
                return True
        
        return False
    
    def _start_evaluation_loop(self):
        """Start alert rule evaluation loop"""
        def evaluation_worker():
            while not self._stop_evaluation.wait(self.evaluation_interval):
                try:
                    self._evaluate_all_rules()
                    self._cleanup_resolved_alerts()
                except Exception as e:
                    logger.error(f"Alert evaluation failed: {e}")
        
        self._evaluation_thread = threading.Thread(target=evaluation_worker, daemon=True)
        self._evaluation_thread.start()
        logger.info("Alert evaluation loop started")
    
    def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        with self._lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                alert = self.evaluator.evaluate_rule(rule)
                if alert:
                    # Check if we already have a similar active alert
                    similar_exists = any(
                        active_alert.name == alert.name and 
                        active_alert.metric_name == alert.metric_name
                        for active_alert in self.active_alerts.values()
                    )
                    
                    if not similar_exists:
                        self.fire_alert(alert)
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            # Clean up alert history (handled by deque maxlen)
            
            # Auto-resolve alerts that are no longer triggering
            self._auto_resolve_alerts()
    
    def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are no longer triggering"""
        alerts_to_resolve = []
        
        with self._lock:
            for alert_id, alert in self.active_alerts.items():
                # Find matching rule
                matching_rule = None
                for rule in self.alert_rules.values():
                    if rule.name == alert.name:
                        matching_rule = rule
                        break
                
                if matching_rule:
                    # Re-evaluate the condition
                    current_alert = self.evaluator.evaluate_rule(matching_rule)
                    if not current_alert:
                        # Condition no longer met, auto-resolve
                        alerts_to_resolve.append(alert_id)
            
            # Resolve alerts
            for alert_id in alerts_to_resolve:
                self.resolve_alert(alert_id)
    
    def get_alerts(self, status: Optional[AlertStatus] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional status filter"""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if status:
                alerts = [alert for alert in alerts if alert.status == status]
            
            alerts.sort(key=lambda x: x.created_at, reverse=True)
            return [alert.to_dict() for alert in alerts]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert management summary"""
        with self._lock:
            return {
                'total_alerts': self.stats['total_alerts'],
                'active_alerts': len(self.active_alerts),
                'alerts_by_severity': dict(self.stats['alerts_by_severity']),
                'alerts_resolved': self.stats['alerts_resolved'],
                'notifications_sent': self.stats['notifications_sent'],
                'alert_rules': len(self.alert_rules),
                'notification_channels': len(self.notification_manager.channels),
                'suppressed_alerts': len(self.suppressed_alerts)
            }
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        with self._lock:
            return [
                {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'condition': rule.condition,
                    'severity': rule.severity.value,
                    'threshold': rule.threshold,
                    'enabled': rule.enabled,
                    'notify_channels': rule.notify_channels
                }
                for rule in self.alert_rules.values()
            ]
    
    def shutdown(self):
        """Shutdown alert manager"""
        try:
            self._stop_evaluation.set()
            if self._evaluation_thread and self._evaluation_thread.is_alive():
                self._evaluation_thread.join(timeout=5.0)
            
            logger.info("Alert manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Alert manager shutdown failed: {e}")


def create_alert_manager(metrics_collector, evaluation_interval: float = 30.0) -> AlertManager:
    """
    Factory function to create alert manager.
    
    Args:
        metrics_collector: MetricsCollector instance
        evaluation_interval: Alert rule evaluation interval in seconds
        
    Returns:
        AlertManager instance
    """
    return AlertManager(metrics_collector, evaluation_interval)