"""
Enterprise Security Dashboard

Real-time security monitoring, metrics collection, alerting system,
and comprehensive security posture visualization for enterprise environments.
"""

import json
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from .audit import SecurityAuditLogger, SecurityEvent, SecurityEventType, SeverityLevel

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Security alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


class MetricType(Enum):
    """Types of security metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class SecurityMetric:
    """Security metric definition."""
    metric_id: str
    name: str
    metric_type: MetricType
    description: str
    value: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    source_event: Optional[str] = None
    related_metrics: List[str] = field(default_factory=list)
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    alert_channels: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    widget_type: str  # metric, chart, table, alert_list, status
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, w, h
    refresh_interval: int = 60  # seconds
    enabled: bool = True


@dataclass
class SecurityReport:
    """Security summary report."""
    report_id: str
    report_type: str  # daily, weekly, monthly, incident
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SecurityDashboardError(Exception):
    """Security dashboard related errors."""
    pass


class EnterpriseSecurityDashboard:
    """
    Enterprise security dashboard and monitoring system.
    
    Features:
    - Real-time security metrics collection and visualization
    - Multi-channel alerting system with escalation policies
    - Customizable dashboard widgets and layouts
    - Automated security reporting and analytics
    - Integration with external monitoring systems
    - Security KPI tracking and trend analysis
    - Incident correlation and investigation tools
    - Compliance monitoring and reporting
    """
    
    def __init__(self, db_path: str = "memmimic_security_dashboard.db",
                 config_path: str = "security_dashboard_config.json",
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize security dashboard.
        
        Args:
            db_path: Path to dashboard database
            config_path: Path to dashboard configuration file
            audit_logger: Security audit logger instance
        """
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Security metrics storage
        self.metrics: Dict[str, SecurityMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at 1min intervals
        
        # Alerts and notifications
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Dashboard configuration
        self.widgets: Dict[str, DashboardWidget] = {}
        self.dashboard_config: Dict[str, Any] = {}
        
        # Alerting configuration
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: Dict[str, Dict[str, Any]] = {}
        
        # Background processing
        self.running = False
        self.metric_collector_thread = None
        self.alert_processor_thread = None
        
        # Initialize database
        self._initialize_database()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize default metrics and widgets
        self._initialize_default_configuration()
        
        # Start background processing
        self._start_background_processing()
        
        logger.info("EnterpriseSecurityDashboard initialized")
    
    def _initialize_database(self) -> None:
        """Initialize security dashboard database."""
        with self._get_db_connection() as conn:
            # Security metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_metrics (
                    metric_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    description TEXT,
                    value REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tags TEXT,
                    unit TEXT DEFAULT 'count',
                    thresholds TEXT
                )
            ''')
            
            # Security alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_alerts (
                    alert_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    source_event TEXT,
                    related_metrics TEXT,
                    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolved_by TEXT,
                    alert_channels TEXT,
                    metadata TEXT
                )
            ''')
            
            # Dashboard widgets table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_widgets (
                    widget_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    widget_type TEXT NOT NULL,
                    config TEXT,
                    position TEXT,
                    refresh_interval INTEGER DEFAULT 60,
                    enabled BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Security reports table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    summary TEXT,
                    metrics TEXT,
                    alerts TEXT,
                    recommendations TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON security_metrics (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_id_timestamp ON security_metrics (metric_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON security_alerts (severity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON security_alerts (triggered_at)')
            
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
    
    def _load_configuration(self) -> None:
        """Load dashboard configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.dashboard_config = config.get('dashboard', {})
                self.alert_rules = config.get('alert_rules', {})
                self.notification_channels = config.get('notification_channels', {})
                
                # Load widgets
                for widget_config in config.get('widgets', []):
                    widget = DashboardWidget(
                        widget_id=widget_config['widget_id'],
                        title=widget_config['title'],
                        widget_type=widget_config['widget_type'],
                        config=widget_config.get('config', {}),
                        position=widget_config.get('position', {}),
                        refresh_interval=widget_config.get('refresh_interval', 60),
                        enabled=widget_config.get('enabled', True)
                    )
                    self.widgets[widget.widget_id] = widget
                
            except Exception as e:
                logger.warning(f"Failed to load dashboard configuration: {e}")
    
    def _initialize_default_configuration(self) -> None:
        """Initialize default dashboard configuration."""
        # Default security metrics
        default_metrics = [
            SecurityMetric(
                metric_id="auth_failures_per_minute",
                name="Authentication Failures",
                metric_type=MetricType.RATE,
                description="Rate of authentication failures per minute",
                unit="failures/min",
                thresholds={"warning": 10.0, "critical": 50.0}
            ),
            SecurityMetric(
                metric_id="active_sessions",
                name="Active User Sessions",
                metric_type=MetricType.GAUGE,
                description="Number of currently active user sessions",
                unit="sessions",
                thresholds={"warning": 1000.0, "critical": 5000.0}
            ),
            SecurityMetric(
                metric_id="threat_detections_per_hour",
                name="Threat Detections",
                metric_type=MetricType.RATE,
                description="Number of threat detections per hour",
                unit="detections/hour",
                thresholds={"warning": 10.0, "critical": 100.0}
            ),
            SecurityMetric(
                metric_id="api_requests_per_minute",
                name="API Request Rate",
                metric_type=MetricType.RATE,
                description="API requests per minute",
                unit="requests/min",
                thresholds={"warning": 1000.0, "critical": 10000.0}
            ),
            SecurityMetric(
                metric_id="compliance_score",
                name="Compliance Score",
                metric_type=MetricType.GAUGE,
                description="Overall compliance score percentage",
                unit="percent",
                thresholds={"critical": 70.0, "warning": 85.0}
            ),
            SecurityMetric(
                metric_id="zero_trust_score",
                name="Zero Trust Score",
                metric_type=MetricType.GAUGE,
                description="Zero trust security posture score",
                unit="percent",
                thresholds={"critical": 60.0, "warning": 80.0}
            )
        ]
        
        for metric in default_metrics:
            if metric.metric_id not in self.metrics:
                self.metrics[metric.metric_id] = metric
        
        # Default dashboard widgets
        default_widgets = [
            DashboardWidget(
                widget_id="security_overview",
                title="Security Overview",
                widget_type="status",
                config={"metrics": ["compliance_score", "zero_trust_score"]},
                position={"x": 0, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="threat_detection_chart",
                title="Threat Detections (24h)",
                widget_type="chart",
                config={
                    "metric": "threat_detections_per_hour",
                    "chart_type": "line",
                    "time_range": "24h"
                },
                position={"x": 6, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                widget_id="authentication_metrics",
                title="Authentication Metrics",
                widget_type="metric",
                config={"metrics": ["auth_failures_per_minute", "active_sessions"]},
                position={"x": 0, "y": 4, "w": 4, "h": 3}
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Security Alerts",
                widget_type="alert_list",
                config={"max_alerts": 10, "severity_filter": ["critical", "warning"]},
                position={"x": 4, "y": 4, "w": 8, "h": 3}
            ),
            DashboardWidget(
                widget_id="api_activity",
                title="API Activity",
                widget_type="chart",
                config={
                    "metric": "api_requests_per_minute",
                    "chart_type": "area",
                    "time_range": "1h"
                },
                position={"x": 0, "y": 7, "w": 12, "h": 3}
            )
        ]
        
        for widget in default_widgets:
            if widget.widget_id not in self.widgets:
                self.widgets[widget.widget_id] = widget
                self._store_dashboard_widget(widget)
        
        # Default alert rules
        if not self.alert_rules:
            self.alert_rules = {
                "high_auth_failures": {
                    "metric": "auth_failures_per_minute",
                    "condition": "greater_than",
                    "threshold": 50.0,
                    "severity": "critical",
                    "channels": ["email", "dashboard"],
                    "cooldown": 300  # 5 minutes
                },
                "compliance_score_low": {
                    "metric": "compliance_score",
                    "condition": "less_than",
                    "threshold": 70.0,
                    "severity": "critical",
                    "channels": ["email", "dashboard"],
                    "cooldown": 3600  # 1 hour
                },
                "zero_trust_score_low": {
                    "metric": "zero_trust_score",
                    "condition": "less_than",
                    "threshold": 60.0,
                    "severity": "warning",
                    "channels": ["dashboard"],
                    "cooldown": 1800  # 30 minutes
                }
            }
        
        # Default notification channels
        if not self.notification_channels:
            self.notification_channels = {
                "email": {
                    "type": "email",
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "security@example.com",
                    "password": "",  # Should be configured
                    "recipients": ["security-team@example.com"],
                    "enabled": False  # Disabled until properly configured
                },
                "slack": {
                    "type": "slack",
                    "webhook_url": "",  # Should be configured
                    "channel": "#security-alerts",
                    "enabled": False
                }
            }
        
        logger.info(f"Initialized {len(default_metrics)} default metrics and {len(default_widgets)} widgets")
    
    def _start_background_processing(self) -> None:
        """Start background processing threads."""
        self.running = True
        
        # Start metric collector thread
        self.metric_collector_thread = threading.Thread(target=self._metric_collector_worker, daemon=True)
        self.metric_collector_thread.start()
        
        # Start alert processor thread
        self.alert_processor_thread = threading.Thread(target=self._alert_processor_worker, daemon=True)
        self.alert_processor_thread.start()
        
        logger.info("Background processing started for security dashboard")
    
    def _metric_collector_worker(self) -> None:
        """Background worker for collecting security metrics."""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Collect metrics from various security components
                self._collect_security_metrics()
                
                # Store metrics in database
                self._persist_metrics(current_time)
                
                # Sleep for 60 seconds (1 minute intervals)
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in metric collector: {e}")
                time.sleep(60)
    
    def _alert_processor_worker(self) -> None:
        """Background worker for processing alerts."""
        while self.running:
            try:
                # Check metrics against alert rules
                self._process_alert_rules()
                
                # Process pending alert notifications
                self._process_alert_notifications()
                
                # Clean up resolved alerts
                self._cleanup_old_alerts()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in alert processor: {e}")
                time.sleep(30)
    
    def _collect_security_metrics(self) -> None:
        """Collect metrics from various security components."""
        try:
            # Import security components to collect metrics
            from .authentication import get_auth_manager
            from .threat_detection import get_threat_detector
            from .compliance import get_compliance_engine
            from .zero_trust import get_zerotrust_engine
            
            # Collect authentication metrics
            try:
                auth_manager = get_auth_manager()
                # Get active session count (would need implementation in auth manager)
                # For now, simulate the metric
                self.update_metric("active_sessions", 150.0)
                
                # Auth failure rate (would get from auth manager)
                self.update_metric("auth_failures_per_minute", 5.0)
                
            except Exception as e:
                logger.warning(f"Failed to collect authentication metrics: {e}")
            
            # Collect threat detection metrics
            try:
                threat_detector = get_threat_detector()
                dashboard_data = threat_detector.get_threat_dashboard_data()
                
                # Calculate threat detection rate
                total_threats = dashboard_data.get("metrics", {}).get("threat_detections", 0)
                self.update_metric("threat_detections_per_hour", total_threats / 24.0)  # Approximate hourly rate
                
            except Exception as e:
                logger.warning(f"Failed to collect threat detection metrics: {e}")
            
            # Collect compliance metrics
            try:
                compliance_engine = get_compliance_engine()
                dashboard_data = compliance_engine.get_compliance_dashboard_data()
                
                # Calculate overall compliance score
                total_controls = 0
                compliant_controls = 0
                
                for framework_data in dashboard_data.get("frameworks", {}).values():
                    total_controls += framework_data.get("total_controls", 0)
                    compliant_controls += framework_data.get("status_counts", {}).get("compliant", 0)
                
                if total_controls > 0:
                    compliance_score = (compliant_controls / total_controls) * 100
                    self.update_metric("compliance_score", compliance_score)
                
            except Exception as e:
                logger.warning(f"Failed to collect compliance metrics: {e}")
            
            # Collect zero-trust metrics
            try:
                zerotrust_engine = get_zerotrust_engine()
                dashboard_data = zerotrust_engine.get_zero_trust_dashboard_data()
                
                # Calculate zero trust score based on device trust levels
                devices = dashboard_data.get("devices", {})
                total_devices = devices.get("total", 0)
                trust_levels = devices.get("by_trust_level", {})
                
                if total_devices > 0:
                    # Calculate weighted trust score
                    trust_weights = {
                        "verified": 1.0,
                        "high_trust": 0.8,
                        "medium_trust": 0.6,
                        "low_trust": 0.3,
                        "untrusted": 0.0
                    }
                    
                    weighted_score = sum(
                        trust_levels.get(level, 0) * weight
                        for level, weight in trust_weights.items()
                    )
                    
                    zero_trust_score = (weighted_score / total_devices) * 100
                    self.update_metric("zero_trust_score", zero_trust_score)
                
            except Exception as e:
                logger.warning(f"Failed to collect zero-trust metrics: {e}")
            
            # API request metrics (would integrate with API gateway)
            self.update_metric("api_requests_per_minute", 250.0)  # Simulated
            
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
    
    def update_metric(self, metric_id: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Update a security metric value."""
        if metric_id not in self.metrics:
            logger.warning(f"Unknown metric: {metric_id}")
            return
        
        metric = self.metrics[metric_id]
        metric.value = value
        metric.timestamp = datetime.now(timezone.utc)
        
        if tags:
            metric.tags.update(tags)
        
        # Add to history
        self.metric_history[metric_id].append({
            "timestamp": metric.timestamp,
            "value": value,
            "tags": metric.tags.copy()
        })
    
    def _persist_metrics(self, timestamp: datetime) -> None:
        """Persist current metrics to database."""
        with self._get_db_connection() as conn:
            for metric in self.metrics.values():
                conn.execute('''
                    INSERT INTO security_metrics (
                        metric_id, name, metric_type, description, value,
                        timestamp, tags, unit, thresholds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.metric_id, metric.name, metric.metric_type.value,
                    metric.description, metric.value, timestamp.isoformat(),
                    json.dumps(metric.tags), metric.unit, json.dumps(metric.thresholds)
                ))
            
            conn.commit()
    
    def _process_alert_rules(self) -> None:
        """Process alert rules against current metrics."""
        for rule_id, rule_config in self.alert_rules.items():
            try:
                metric_id = rule_config["metric"]
                condition = rule_config["condition"]
                threshold = rule_config["threshold"]
                severity = rule_config["severity"]
                channels = rule_config["channels"]
                cooldown = rule_config.get("cooldown", 300)  # Default 5 minutes
                
                if metric_id not in self.metrics:
                    continue
                
                metric = self.metrics[metric_id]
                
                # Check if alert is in cooldown
                existing_alert = None
                for alert in self.active_alerts.values():
                    if (alert.component == rule_id and 
                        not alert.resolved_at and
                        (datetime.now(timezone.utc) - alert.triggered_at).total_seconds() < cooldown):
                        existing_alert = alert
                        break
                
                if existing_alert:
                    continue  # Still in cooldown
                
                # Evaluate condition
                alert_triggered = False
                
                if condition == "greater_than" and metric.value > threshold:
                    alert_triggered = True
                elif condition == "less_than" and metric.value < threshold:
                    alert_triggered = True
                elif condition == "equals" and metric.value == threshold:
                    alert_triggered = True
                
                if alert_triggered:
                    # Create new alert
                    alert_id = f"alert_{int(time.time())}_{rule_id}"
                    
                    alert = SecurityAlert(
                        alert_id=alert_id,
                        title=f"Security Alert: {metric.name}",
                        description=f"{metric.name} is {metric.value} {metric.unit}, which {condition.replace('_', ' ')} threshold of {threshold}",
                        severity=AlertSeverity(severity),
                        component=rule_id,
                        related_metrics=[metric_id],
                        alert_channels=[AlertChannel(ch) for ch in channels],
                        metadata={
                            "metric_value": metric.value,
                            "threshold": threshold,
                            "condition": condition
                        }
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    self._store_security_alert(alert)
                    
                    # Log alert creation
                    self.audit_logger.log_security_event(SecurityEvent(
                        event_type=SecurityEventType.SECURITY_VIOLATION,
                        component="security_dashboard",
                        severity=SeverityLevel.HIGH if severity == "critical" else SeverityLevel.MEDIUM,
                        details=f"Security alert triggered: {alert.title}",
                        metadata={
                            "alert_id": alert_id,
                            "rule_id": rule_id,
                            "metric_value": metric.value,
                            "threshold": threshold
                        }
                    ))
                    
                    logger.warning(f"Security alert triggered: {alert.title}")
                
            except Exception as e:
                logger.error(f"Error processing alert rule {rule_id}: {e}")
    
    def _process_alert_notifications(self) -> None:
        """Process pending alert notifications."""
        for alert in self.active_alerts.values():
            if alert.resolved_at or alert.acknowledged_at:
                continue  # Skip resolved or acknowledged alerts
            
            for channel in alert.alert_channels:
                try:
                    self._send_alert_notification(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send alert notification via {channel.value}: {e}")
    
    def _send_alert_notification(self, alert: SecurityAlert, channel: AlertChannel) -> None:
        """Send alert notification via specified channel."""
        if channel == AlertChannel.EMAIL:
            self._send_email_notification(alert)
        elif channel == AlertChannel.SLACK:
            self._send_slack_notification(alert)
        elif channel == AlertChannel.WEBHOOK:
            self._send_webhook_notification(alert)
        # Dashboard notifications are handled in the UI
    
    def _send_email_notification(self, alert: SecurityAlert) -> None:
        """Send email notification for alert."""
        email_config = self.notification_channels.get("email", {})
        if not email_config.get("enabled", False):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Security Alert: {alert.title}
            
            Severity: {alert.severity.value.upper()}
            Component: {alert.component}
            Triggered: {alert.triggered_at.isoformat()}
            
            Description:
            {alert.description}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            
            ---
            MemMimic Enterprise Security Dashboard
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, alert: SecurityAlert) -> None:
        """Send Slack notification for alert."""
        slack_config = self.notification_channels.get("slack", {})
        if not slack_config.get("enabled", False):
            return
        
        try:
            webhook_url = slack_config["webhook_url"]
            if not webhook_url:
                return
            
            # Color coding based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffcc00",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8b0000"
            }
            
            payload = {
                "channel": slack_config.get("channel", "#security-alerts"),
                "username": "MemMimic Security",
                "text": f"Security Alert: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ffcc00"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "footer": "MemMimic Enterprise Security Dashboard",
                    "ts": int(alert.triggered_at.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook_notification(self, alert: SecurityAlert) -> None:
        """Send webhook notification for alert."""
        webhook_config = self.notification_channels.get("webhook", {})
        if not webhook_config.get("enabled", False):
            return
        
        try:
            webhook_url = webhook_config["url"]
            
            payload = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "component": alert.component,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            }
            
            headers = {"Content-Type": "application/json"}
            if "auth_token" in webhook_config:
                headers["Authorization"] = f"Bearer {webhook_config['auth_token']}"
            
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if (alert.resolved_at and alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
        
        if alerts_to_remove:
            logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a security alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = acknowledged_by
        
        self._store_security_alert(alert)
        
        # Log acknowledgment
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.FUNCTION_CALL,
            component="security_dashboard",
            severity=SeverityLevel.LOW,
            details=f"Security alert acknowledged: {alert.title}",
            metadata={
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by
            }
        ))
        
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve a security alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved_at = datetime.now(timezone.utc)
        alert.resolved_by = resolved_by
        
        self._store_security_alert(alert)
        
        # Log resolution
        self.audit_logger.log_security_event(SecurityEvent(
            event_type=SecurityEventType.FUNCTION_CALL,
            component="security_dashboard",
            severity=SeverityLevel.LOW,
            details=f"Security alert resolved: {alert.title}",
            metadata={
                "alert_id": alert_id,
                "resolved_by": resolved_by
            }
        ))
        
        return True
    
    def generate_security_report(self, report_type: str = "daily",
                                period_start: Optional[datetime] = None,
                                period_end: Optional[datetime] = None) -> SecurityReport:
        """Generate comprehensive security report."""
        if not period_end:
            period_end = datetime.now(timezone.utc)
        
        if not period_start:
            if report_type == "daily":
                period_start = period_end - timedelta(days=1)
            elif report_type == "weekly":
                period_start = period_end - timedelta(days=7)
            elif report_type == "monthly":
                period_start = period_end - timedelta(days=30)
            else:
                period_start = period_end - timedelta(days=1)
        
        report_id = f"report_{int(time.time())}_{report_type}"
        
        # Collect metrics for the period
        period_metrics = {}
        for metric_id, metric in self.metrics.items():
            period_metrics[metric_id] = metric.value
        
        # Count alerts in the period
        period_alerts = []
        for alert in self.alert_history:
            if period_start <= alert.triggered_at <= period_end:
                period_alerts.append(alert.alert_id)
        
        # Generate summary
        summary = {
            "total_alerts": len(period_alerts),
            "critical_alerts": len([a for a in self.alert_history 
                                  if a.severity == AlertSeverity.CRITICAL and 
                                  period_start <= a.triggered_at <= period_end]),
            "avg_compliance_score": period_metrics.get("compliance_score", 0),
            "avg_zero_trust_score": period_metrics.get("zero_trust_score", 0),
            "total_threat_detections": period_metrics.get("threat_detections_per_hour", 0) * 24,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat()
        }
        
        # Generate recommendations
        recommendations = []
        if summary["critical_alerts"] > 0:
            recommendations.append(f"Address {summary['critical_alerts']} critical security alerts")
        
        if summary["avg_compliance_score"] < 85:
            recommendations.append("Improve compliance posture - current score is below recommended threshold")
        
        if summary["avg_zero_trust_score"] < 80:
            recommendations.append("Enhance zero-trust security controls to improve overall posture")
        
        if summary["total_threat_detections"] > 100:
            recommendations.append("Review threat detection patterns for potential security concerns")
        
        report = SecurityReport(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            summary=summary,
            metrics=period_metrics,
            alerts=period_alerts,
            recommendations=recommendations
        )
        
        self._store_security_report(report)
        
        return report
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for frontend."""
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "widgets": [],
            "active_alerts": [],
            "system_status": "healthy"
        }
        
        # Current metrics
        for metric_id, metric in self.metrics.items():
            dashboard_data["metrics"][metric_id] = {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "thresholds": metric.thresholds,
                "tags": metric.tags
            }
        
        # Widget configurations
        for widget in self.widgets.values():
            if widget.enabled:
                dashboard_data["widgets"].append({
                    "widget_id": widget.widget_id,
                    "title": widget.title,
                    "widget_type": widget.widget_type,
                    "config": widget.config,
                    "position": widget.position,
                    "refresh_interval": widget.refresh_interval
                })
        
        # Active alerts
        for alert in self.active_alerts.values():
            if not alert.resolved_at:
                dashboard_data["active_alerts"].append({
                    "alert_id": alert.alert_id,
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "component": alert.component,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "acknowledged": alert.acknowledged_at is not None
                })
        
        # Determine overall system status
        critical_alerts = [a for a in dashboard_data["active_alerts"] 
                          if a["severity"] == "critical"]
        
        if critical_alerts:
            dashboard_data["system_status"] = "critical"
        elif len(dashboard_data["active_alerts"]) > 5:
            dashboard_data["system_status"] = "warning"
        
        return dashboard_data
    
    def _store_dashboard_widget(self, widget: DashboardWidget) -> None:
        """Store dashboard widget in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO dashboard_widgets (
                    widget_id, title, widget_type, config, position,
                    refresh_interval, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                widget.widget_id, widget.title, widget.widget_type,
                json.dumps(widget.config), json.dumps(widget.position),
                widget.refresh_interval, widget.enabled
            ))
            conn.commit()
    
    def _store_security_alert(self, alert: SecurityAlert) -> None:
        """Store security alert in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO security_alerts (
                    alert_id, title, description, severity, component,
                    source_event, related_metrics, triggered_at, acknowledged_at,
                    resolved_at, acknowledged_by, resolved_by, alert_channels, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.title, alert.description,
                alert.severity.value, alert.component, alert.source_event,
                json.dumps(alert.related_metrics), alert.triggered_at.isoformat(),
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_by, alert.resolved_by,
                json.dumps([ch.value for ch in alert.alert_channels]),
                json.dumps(alert.metadata)
            ))
            conn.commit()
    
    def _store_security_report(self, report: SecurityReport) -> None:
        """Store security report in database."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO security_reports (
                    report_id, report_type, period_start, period_end,
                    summary, metrics, alerts, recommendations, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id, report.report_type,
                report.period_start.isoformat(), report.period_end.isoformat(),
                json.dumps(report.summary), json.dumps(report.metrics),
                json.dumps(report.alerts), json.dumps(report.recommendations),
                report.generated_at.isoformat()
            ))
            conn.commit()
    
    def shutdown(self) -> None:
        """Shutdown security dashboard."""
        self.running = False
        
        if self.metric_collector_thread and self.metric_collector_thread.is_alive():
            self.metric_collector_thread.join(timeout=5)
        
        if self.alert_processor_thread and self.alert_processor_thread.is_alive():
            self.alert_processor_thread.join(timeout=5)
        
        logger.info("EnterpriseSecurityDashboard shutdown completed")


# Global security dashboard instance
_global_security_dashboard: Optional[EnterpriseSecurityDashboard] = None


def get_security_dashboard() -> EnterpriseSecurityDashboard:
    """Get the global security dashboard."""
    global _global_security_dashboard
    if _global_security_dashboard is None:
        _global_security_dashboard = EnterpriseSecurityDashboard()
    return _global_security_dashboard


def initialize_security_dashboard(db_path: str = "memmimic_security_dashboard.db",
                                 config_path: str = "security_dashboard_config.json",
                                 audit_logger: Optional[SecurityAuditLogger] = None) -> EnterpriseSecurityDashboard:
    """Initialize the global security dashboard."""
    global _global_security_dashboard
    _global_security_dashboard = EnterpriseSecurityDashboard(db_path, config_path, audit_logger)
    return _global_security_dashboard