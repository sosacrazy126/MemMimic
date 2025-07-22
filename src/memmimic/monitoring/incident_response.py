"""
Enterprise Incident Response System for MemMimic

Provides automated incident response, escalation procedures, 
remediation actions, and incident management workflows.
"""

import asyncio
import time
import threading
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from .metrics_collector import get_metrics_collector
from .alert_manager import Alert, AlertSeverity
from .security_monitor import SecurityEvent, SecuritySeverity

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status states"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    POST_MORTEM = "post_mortem"


class ResponseAction(Enum):
    """Types of automated response actions"""
    ALERT = "alert"
    RESTART_COMPONENT = "restart_component"
    SCALE_RESOURCES = "scale_resources"
    ISOLATE_COMPONENT = "isolate_component"
    BACKUP_DATA = "backup_data"
    THROTTLE_REQUESTS = "throttle_requests"
    BLOCK_IP = "block_ip"
    DISABLE_FEATURE = "disable_feature"
    ESCALATE = "escalate"
    CUSTOM_SCRIPT = "custom_script"


@dataclass
class ResponseActionConfig:
    """Configuration for automated response action"""
    action_type: ResponseAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: int = 10


@dataclass
class Incident:
    """Incident data structure"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.DETECTED
    source_events: List[str] = field(default_factory=list)  # Alert/event IDs
    affected_components: Set[str] = field(default_factory=set)
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    response_actions: List[str] = field(default_factory=list)  # Action IDs
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'source_events': self.source_events,
            'affected_components': list(self.affected_components),
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'response_actions': self.response_actions,
            'tags': list(self.tags),
            'metadata': self.metadata
        }


@dataclass
class ResponseActionResult:
    """Result of executing a response action"""
    action_id: str
    action_type: ResponseAction
    success: bool
    message: str
    executed_at: datetime = field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    output: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'action_type': self.action_type.value,
            'success': self.success,
            'message': self.message,
            'executed_at': self.executed_at.isoformat(),
            'execution_time_seconds': self.execution_time_seconds,
            'output': self.output,
            'error_details': self.error_details
        }


class IncidentDetector:
    """Detects incidents from alerts and security events"""
    
    def __init__(self):
        # Correlation rules for incident detection
        self.correlation_rules = {
            'memory_exhaustion': {
                'conditions': [
                    {'metric': 'memmimic_memory_usage_percent', 'operator': '>', 'threshold': 90},
                    {'metric': 'memmimic_errors_total', 'operator': '>', 'threshold': 5}
                ],
                'severity': IncidentSeverity.HIGH,
                'title': 'Memory Exhaustion Detected',
                'description': 'System is running low on memory and experiencing errors',
                'affected_components': {'memory_manager', 'cache_manager'},
                'response_actions': ['restart_component', 'alert']
            },
            'database_failure': {
                'conditions': [
                    {'alert_name': 'Database Response Time High'},
                    {'metric': 'memmimic_db_connections_active', 'operator': '>', 'threshold': 15}
                ],
                'severity': IncidentSeverity.CRITICAL,
                'title': 'Database Performance Degradation',
                'description': 'Database is experiencing severe performance issues',
                'affected_components': {'database', 'storage'},
                'response_actions': ['backup_data', 'scale_resources', 'escalate']
            },
            'security_breach': {
                'conditions': [
                    {'security_event': 'injection_attempt'},
                    {'security_event': 'brute_force_attack'}
                ],
                'severity': IncidentSeverity.CRITICAL,
                'title': 'Security Incident Detected',
                'description': 'Multiple security events detected indicating potential breach',
                'affected_components': {'api', 'authentication'},
                'response_actions': ['block_ip', 'disable_feature', 'escalate']
            },
            'system_overload': {
                'conditions': [
                    {'metric': 'memmimic_cpu_usage_percent', 'operator': '>', 'threshold': 90},
                    {'metric': 'memmimic_load_average_1m', 'operator': '>', 'threshold': 8}
                ],
                'severity': IncidentSeverity.HIGH,
                'title': 'System Overload',
                'description': 'System is experiencing high CPU load and may be unresponsive',
                'affected_components': {'system', 'api'},
                'response_actions': ['throttle_requests', 'scale_resources']
            }
        }
    
    def detect_incidents(self, alerts: List[Alert], security_events: List[SecurityEvent],
                        current_metrics: Dict[str, float]) -> List[Incident]:
        """Detect incidents based on alerts, security events, and metrics"""
        detected_incidents = []
        
        for rule_id, rule in self.correlation_rules.items():
            if self._evaluate_correlation_rule(rule, alerts, security_events, current_metrics):
                incident = self._create_incident_from_rule(rule_id, rule, alerts, security_events)
                detected_incidents.append(incident)
        
        return detected_incidents
    
    def _evaluate_correlation_rule(self, rule: Dict[str, Any], alerts: List[Alert],
                                 security_events: List[SecurityEvent],
                                 current_metrics: Dict[str, float]) -> bool:
        """Evaluate if correlation rule conditions are met"""
        conditions = rule['conditions']
        matched_conditions = 0
        
        for condition in conditions:
            if 'metric' in condition:
                # Metric-based condition
                metric_name = condition['metric']
                operator = condition['operator']
                threshold = condition['threshold']
                
                if metric_name in current_metrics:
                    current_value = current_metrics[metric_name]
                    if self._evaluate_condition(current_value, operator, threshold):
                        matched_conditions += 1
            
            elif 'alert_name' in condition:
                # Alert-based condition
                alert_name = condition['alert_name']
                if any(alert.name == alert_name for alert in alerts):
                    matched_conditions += 1
            
            elif 'security_event' in condition:
                # Security event-based condition
                event_type = condition['security_event']
                if any(event.event_type.value == event_type for event in security_events):
                    matched_conditions += 1
        
        # Rule matches if at least 2 conditions are met (or all if fewer than 2)
        required_matches = min(2, len(conditions))
        return matched_conditions >= required_matches
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a metric condition"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        else:
            return False
    
    def _create_incident_from_rule(self, rule_id: str, rule: Dict[str, Any],
                                 alerts: List[Alert], security_events: List[SecurityEvent]) -> Incident:
        """Create incident from correlation rule"""
        incident_id = f"{rule_id}_{int(time.time())}"
        
        # Collect source event IDs
        source_events = []
        for alert in alerts:
            if any(condition.get('alert_name') == alert.name for condition in rule['conditions']):
                source_events.append(alert.alert_id)
        
        for event in security_events:
            if any(condition.get('security_event') == event.event_type.value 
                  for condition in rule['conditions']):
                source_events.append(f"security_{int(event.timestamp.timestamp())}")
        
        return Incident(
            incident_id=incident_id,
            title=rule['title'],
            description=rule['description'],
            severity=rule['severity'],
            source_events=source_events,
            affected_components=set(rule['affected_components']),
            detected_at=datetime.now(),
            tags={rule_id},
            metadata={'correlation_rule': rule_id}
        )


class ResponseActionExecutor:
    """Executes automated response actions"""
    
    def __init__(self):
        self.action_handlers = {
            ResponseAction.ALERT: self._handle_alert_action,
            ResponseAction.RESTART_COMPONENT: self._handle_restart_component,
            ResponseAction.SCALE_RESOURCES: self._handle_scale_resources,
            ResponseAction.ISOLATE_COMPONENT: self._handle_isolate_component,
            ResponseAction.BACKUP_DATA: self._handle_backup_data,
            ResponseAction.THROTTLE_REQUESTS: self._handle_throttle_requests,
            ResponseAction.BLOCK_IP: self._handle_block_ip,
            ResponseAction.DISABLE_FEATURE: self._handle_disable_feature,
            ResponseAction.ESCALATE: self._handle_escalate,
            ResponseAction.CUSTOM_SCRIPT: self._handle_custom_script
        }
        
        # Track executed actions
        self.action_results: List[ResponseActionResult] = []
        
    async def execute_action(self, action_config: ResponseActionConfig,
                           incident: Incident) -> ResponseActionResult:
        """Execute a response action"""
        action_id = f"{action_config.action_type.value}_{int(time.time())}"
        start_time = time.time()
        
        try:
            handler = self.action_handlers.get(action_config.action_type)
            if not handler:
                raise ValueError(f"No handler for action type: {action_config.action_type}")
            
            # Execute action with timeout
            result = await asyncio.wait_for(
                handler(action_config.parameters, incident),
                timeout=action_config.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            action_result = ResponseActionResult(
                action_id=action_id,
                action_type=action_config.action_type,
                success=True,
                message=f"Action {action_config.action_type.value} completed successfully",
                execution_time_seconds=execution_time,
                output=str(result) if result else None
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            action_result = ResponseActionResult(
                action_id=action_id,
                action_type=action_config.action_type,
                success=False,
                message=f"Action {action_config.action_type.value} timed out",
                execution_time_seconds=execution_time,
                error_details="Timeout exceeded"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            action_result = ResponseActionResult(
                action_id=action_id,
                action_type=action_config.action_type,
                success=False,
                message=f"Action {action_config.action_type.value} failed: {str(e)}",
                execution_time_seconds=execution_time,
                error_details=str(e)
            )
        
        self.action_results.append(action_result)
        return action_result
    
    async def _handle_alert_action(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle alert action"""
        recipients = parameters.get('recipients', ['admin'])
        message = parameters.get('message', f"Incident detected: {incident.title}")
        
        # Log alert (in production, would send to notification system)
        logger.critical(f"INCIDENT ALERT: {message} (ID: {incident.incident_id})")
        
        return f"Alert sent to {', '.join(recipients)}"
    
    async def _handle_restart_component(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle component restart action"""
        component = parameters.get('component', 'unknown')
        
        # Simulate component restart (in production, would actually restart service)
        logger.warning(f"Simulating restart of component: {component}")
        await asyncio.sleep(2)  # Simulate restart time
        
        return f"Component {component} restarted"
    
    async def _handle_scale_resources(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle resource scaling action"""
        resource_type = parameters.get('resource_type', 'cpu')
        scale_factor = parameters.get('scale_factor', 1.5)
        
        # Simulate resource scaling
        logger.info(f"Simulating scaling {resource_type} by factor {scale_factor}")
        await asyncio.sleep(1)
        
        return f"Scaled {resource_type} by {scale_factor}x"
    
    async def _handle_isolate_component(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle component isolation action"""
        component = parameters.get('component', 'unknown')
        
        # Simulate component isolation
        logger.warning(f"Simulating isolation of component: {component}")
        
        return f"Component {component} isolated"
    
    async def _handle_backup_data(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle data backup action"""
        data_type = parameters.get('data_type', 'all')
        
        # Simulate data backup
        logger.info(f"Simulating backup of {data_type} data")
        await asyncio.sleep(3)  # Simulate backup time
        
        return f"Backup of {data_type} data completed"
    
    async def _handle_throttle_requests(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle request throttling action"""
        throttle_rate = parameters.get('throttle_rate', 0.5)
        duration_minutes = parameters.get('duration_minutes', 30)
        
        # Simulate request throttling
        logger.warning(f"Simulating request throttling at {throttle_rate} rate for {duration_minutes} minutes")
        
        return f"Request throttling activated at {throttle_rate} rate"
    
    async def _handle_block_ip(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle IP blocking action"""
        ip_addresses = parameters.get('ip_addresses', [])
        duration_minutes = parameters.get('duration_minutes', 60)
        
        # Simulate IP blocking
        logger.warning(f"Simulating blocking IPs: {ip_addresses} for {duration_minutes} minutes")
        
        return f"Blocked {len(ip_addresses)} IP addresses"
    
    async def _handle_disable_feature(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle feature disabling action"""
        feature = parameters.get('feature', 'unknown')
        
        # Simulate feature disabling
        logger.warning(f"Simulating disabling feature: {feature}")
        
        return f"Feature {feature} disabled"
    
    async def _handle_escalate(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle escalation action"""
        escalation_level = parameters.get('level', 'L2')
        
        # Simulate escalation
        logger.critical(f"Escalating incident {incident.incident_id} to {escalation_level}")
        
        return f"Incident escalated to {escalation_level}"
    
    async def _handle_custom_script(self, parameters: Dict[str, Any], incident: Incident) -> str:
        """Handle custom script execution action"""
        script_path = parameters.get('script_path', '')
        
        # Simulate script execution (in production, would actually run script)
        logger.info(f"Simulating execution of custom script: {script_path}")
        await asyncio.sleep(2)
        
        return f"Custom script {script_path} executed"


class IncidentResponseSystem:
    """
    Enterprise incident response system for MemMimic.
    
    Provides automated incident detection, response action execution,
    escalation procedures, and incident management workflows.
    """
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        
        # Components
        self.incident_detector = IncidentDetector()
        self.action_executor = ResponseActionExecutor()
        
        # Incident management
        self.active_incidents: Dict[str, Incident] = {}
        self.incident_history: List[Incident] = []
        
        # Response configurations
        self.response_configs: Dict[str, List[ResponseActionConfig]] = {
            'memory_exhaustion': [
                ResponseActionConfig(ResponseAction.ALERT, {'recipients': ['ops_team']}),
                ResponseActionConfig(ResponseAction.RESTART_COMPONENT, {'component': 'cache_manager'}),
                ResponseActionConfig(ResponseAction.SCALE_RESOURCES, {'resource_type': 'memory', 'scale_factor': 1.2})
            ],
            'database_failure': [
                ResponseActionConfig(ResponseAction.BACKUP_DATA, {'data_type': 'critical'}),
                ResponseActionConfig(ResponseAction.ALERT, {'recipients': ['dba_team', 'ops_team']}),
                ResponseActionConfig(ResponseAction.ESCALATE, {'level': 'L1'})
            ],
            'security_breach': [
                ResponseActionConfig(ResponseAction.BLOCK_IP, {'duration_minutes': 120}),
                ResponseActionConfig(ResponseAction.DISABLE_FEATURE, {'feature': 'public_api'}),
                ResponseActionConfig(ResponseAction.ESCALATE, {'level': 'security_team'})
            ],
            'system_overload': [
                ResponseActionConfig(ResponseAction.THROTTLE_REQUESTS, {'throttle_rate': 0.3, 'duration_minutes': 15}),
                ResponseActionConfig(ResponseAction.SCALE_RESOURCES, {'resource_type': 'cpu', 'scale_factor': 1.3})
            ]
        }
        
        # Background processing
        self._processing_thread = None
        self._stop_processing = threading.Event()
        
        # Statistics
        self.stats = {
            'incidents_detected': 0,
            'incidents_resolved': 0,
            'actions_executed': 0,
            'actions_successful': 0,
            'mean_time_to_resolution': 0.0
        }
        
        self._start_background_processing()
        
        logger.info("Incident Response System initialized")
    
    def process_alerts(self, alerts: List[Alert]):
        """Process alerts for incident detection"""
        try:
            # Get current metrics for correlation
            current_metrics = self._get_current_metrics()
            
            # Detect incidents
            detected_incidents = self.incident_detector.detect_incidents(
                alerts, [], current_metrics  # No security events for now
            )
            
            for incident in detected_incidents:
                self._handle_new_incident(incident)
                
        except Exception as e:
            logger.error(f"Failed to process alerts for incident detection: {e}")
    
    def process_security_events(self, security_events: List[SecurityEvent]):
        """Process security events for incident detection"""
        try:
            # Get current metrics for correlation
            current_metrics = self._get_current_metrics()
            
            # Detect incidents
            detected_incidents = self.incident_detector.detect_incidents(
                [], security_events, current_metrics
            )
            
            for incident in detected_incidents:
                self._handle_new_incident(incident)
                
        except Exception as e:
            logger.error(f"Failed to process security events for incident detection: {e}")
    
    def _handle_new_incident(self, incident: Incident):
        """Handle a newly detected incident"""
        # Check if similar incident already exists
        if self._has_similar_active_incident(incident):
            logger.debug(f"Similar incident already active, skipping: {incident.title}")
            return
        
        # Add to active incidents
        self.active_incidents[incident.incident_id] = incident
        self.stats['incidents_detected'] += 1
        
        # Update metrics
        self.metrics_collector.increment_counter("memmimic_incidents_total")
        self.metrics_collector.increment_counter(f"memmimic_incidents_{incident.severity.value}_total")
        
        logger.warning(f"New incident detected: {incident.title} (ID: {incident.incident_id})")
        
        # Start response actions
        asyncio.create_task(self._execute_incident_response(incident))
    
    def _has_similar_active_incident(self, incident: Incident) -> bool:
        """Check if similar incident is already active"""
        for active_incident in self.active_incidents.values():
            if (active_incident.title == incident.title and 
                active_incident.severity == incident.severity and
                active_incident.affected_components & incident.affected_components):
                return True
        return False
    
    async def _execute_incident_response(self, incident: Incident):
        """Execute automated response actions for incident"""
        try:
            # Get response actions for this incident type
            correlation_rule = incident.metadata.get('correlation_rule')
            if not correlation_rule or correlation_rule not in self.response_configs:
                logger.warning(f"No response configuration for incident: {incident.incident_id}")
                return
            
            incident.status = IncidentStatus.MITIGATING
            
            response_actions = self.response_configs[correlation_rule]
            
            for action_config in response_actions:
                try:
                    # Execute response action
                    result = await self.action_executor.execute_action(action_config, incident)
                    
                    incident.response_actions.append(result.action_id)
                    self.stats['actions_executed'] += 1
                    
                    if result.success:
                        self.stats['actions_successful'] += 1
                        logger.info(f"Response action completed: {result.message}")
                    else:
                        logger.error(f"Response action failed: {result.message}")
                    
                    # Brief delay between actions
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to execute response action: {e}")
            
            # Check if incident should be auto-resolved
            if await self._should_auto_resolve_incident(incident):
                self.resolve_incident(incident.incident_id, "auto_resolved")
            else:
                incident.status = IncidentStatus.INVESTIGATING
                logger.info(f"Incident requires manual investigation: {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute incident response: {e}")
            incident.status = IncidentStatus.INVESTIGATING
    
    async def _should_auto_resolve_incident(self, incident: Incident) -> bool:
        """Determine if incident should be auto-resolved"""
        # Simple heuristic: resolve if no high-severity alerts are active
        # In production, would have more sophisticated resolution logic
        
        # Wait a bit for actions to take effect
        await asyncio.sleep(30)
        
        # Check current system health
        current_metrics = self._get_current_metrics()
        
        # For memory exhaustion, check if memory usage is back to normal
        if 'memory_exhaustion' in incident.tags:
            memory_usage = current_metrics.get('memmimic_memory_usage_percent', 0)
            return memory_usage < 80
        
        # For system overload, check CPU usage
        if 'system_overload' in incident.tags:
            cpu_usage = current_metrics.get('memmimic_cpu_usage_percent', 0)
            return cpu_usage < 70
        
        # Default: don't auto-resolve
        return False
    
    def resolve_incident(self, incident_id: str, resolved_by: str = "system") -> bool:
        """Manually resolve an incident"""
        if incident_id not in self.active_incidents:
            return False
        
        incident = self.active_incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.assigned_to = resolved_by
        incident.updated_at = datetime.now()
        
        # Move to history
        self.incident_history.append(incident)
        del self.active_incidents[incident_id]
        
        # Update statistics
        self.stats['incidents_resolved'] += 1
        
        # Calculate resolution time
        if incident.detected_at and incident.resolved_at:
            resolution_time = (incident.resolved_at - incident.detected_at).total_seconds() / 60.0  # minutes
            self._update_mean_resolution_time(resolution_time)
        
        logger.info(f"Incident resolved: {incident.title} (ID: {incident_id}) by {resolved_by}")
        return True
    
    def _update_mean_resolution_time(self, resolution_time: float):
        """Update mean time to resolution"""
        current_mean = self.stats['mean_time_to_resolution']
        resolved_count = self.stats['incidents_resolved']
        
        # Calculate new mean
        if resolved_count == 1:
            self.stats['mean_time_to_resolution'] = resolution_time
        else:
            self.stats['mean_time_to_resolution'] = (
                (current_mean * (resolved_count - 1) + resolution_time) / resolved_count
            )
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        current_metrics = {}
        
        # Get gauge values
        for name, gauge in self.metrics_collector._gauges.items():
            current_metrics[name] = gauge.get()
        
        # Get counter values
        for name, counter in self.metrics_collector._counters.items():
            current_metrics[name] = counter.get()
        
        return current_metrics
    
    def _start_background_processing(self):
        """Start background incident processing"""
        def processing_worker():
            while not self._stop_processing.wait(60):  # Process every minute
                try:
                    self._cleanup_old_incidents()
                    self._check_incident_escalation()
                except Exception as e:
                    logger.error(f"Background incident processing failed: {e}")
        
        self._processing_thread = threading.Thread(target=processing_worker, daemon=True)
        self._processing_thread.start()
        logger.info("Background incident processing started")
    
    def _cleanup_old_incidents(self):
        """Clean up old resolved incidents from history"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        self.incident_history = [
            incident for incident in self.incident_history
            if incident.resolved_at and incident.resolved_at >= cutoff_time
        ]
    
    def _check_incident_escalation(self):
        """Check if incidents need escalation"""
        current_time = datetime.now()
        
        for incident in self.active_incidents.values():
            # Escalate critical incidents open for more than 15 minutes
            if (incident.severity == IncidentSeverity.CRITICAL and
                incident.status not in [IncidentStatus.RESOLVED] and
                (current_time - incident.created_at).total_seconds() > 900):  # 15 minutes
                
                logger.critical(f"Escalating critical incident: {incident.incident_id}")
                # Would trigger escalation notification in production
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get incident response system summary"""
        return {
            'active_incidents': len(self.active_incidents),
            'incidents_detected': self.stats['incidents_detected'],
            'incidents_resolved': self.stats['incidents_resolved'],
            'actions_executed': self.stats['actions_executed'],
            'actions_successful': self.stats['actions_successful'],
            'action_success_rate': (
                self.stats['actions_successful'] / max(self.stats['actions_executed'], 1)
            ),
            'mean_time_to_resolution_minutes': self.stats['mean_time_to_resolution'],
            'system_status': 'operational'
        }
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get list of active incidents"""
        incidents = list(self.active_incidents.values())
        incidents.sort(key=lambda x: x.created_at, reverse=True)
        return [incident.to_dict() for incident in incidents]
    
    def get_incident_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get incident history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_incidents = [
            incident for incident in self.incident_history
            if incident.created_at >= cutoff_time
        ]
        
        recent_incidents.sort(key=lambda x: x.created_at, reverse=True)
        return [incident.to_dict() for incident in recent_incidents]
    
    def shutdown(self):
        """Shutdown incident response system"""
        try:
            self._stop_processing.set()
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
            
            logger.info("Incident response system shutdown completed")
            
        except Exception as e:
            logger.error(f"Incident response system shutdown failed: {e}")


def create_incident_response(metrics_collector) -> IncidentResponseSystem:
    """
    Factory function to create incident response system.
    
    Args:
        metrics_collector: MetricsCollector instance
        
    Returns:
        IncidentResponseSystem instance
    """
    return IncidentResponseSystem(metrics_collector)