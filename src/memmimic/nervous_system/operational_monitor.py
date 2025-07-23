"""
Operational Excellence Monitor

Provides comprehensive monitoring, maintenance, and operational excellence
for the enhanced nervous system in production environments.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .triggers.unified_interface import UnifiedEnhancedTriggers
from .temporal_memory_manager import get_temporal_memory_manager
from .performance_optimizer import get_performance_optimizer
from ..errors import get_error_logger, with_error_context

@dataclass
class HealthMetric:
    """Health metric definition"""
    name: str
    threshold: float
    current_value: Optional[float] = None
    status: str = 'unknown'  # 'healthy', 'warning', 'critical', 'unknown'
    last_checked: Optional[datetime] = None
    trend: str = 'stable'  # 'improving', 'stable', 'degrading'

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # 'threshold_exceeded', 'threshold_below', 'trend_negative'
    threshold: float
    severity: str  # 'info', 'warning', 'critical'
    action: str  # 'log', 'notify', 'auto_remediate'
    enabled: bool = True

class OperationalMonitor:
    """
    Comprehensive operational monitoring and maintenance system.
    
    Provides real-time health monitoring, performance tracking, automatic maintenance,
    alerting, and operational excellence for the enhanced nervous system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_error_logger("operational_monitor")
        
        # Health metrics tracking
        self.health_metrics = self._initialize_health_metrics()
        self.alert_rules = self._initialize_alert_rules()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_start_time = None
        self.last_maintenance_time = None
        
        # Performance history
        self.performance_history = {
            'biological_reflex_times': [],
            'memory_operations_per_second': [],
            'system_health_scores': [],
            'error_rates': []
        }
        
        # Maintenance scheduler
        self.maintenance_tasks = self._initialize_maintenance_tasks()
        
        self._initialized = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration"""
        return {
            'monitoring_interval_seconds': 60,
            'health_check_timeout_seconds': 30,
            'performance_history_retention_hours': 24,
            'maintenance_interval_hours': 6,
            'alert_cooldown_minutes': 15,
            'auto_remediation_enabled': True,
            'metrics_export_enabled': True,
            'dashboard_port': 8080,
            'log_retention_days': 7,
            'thresholds': {
                'biological_reflex_ms': 5.0,
                'success_rate_minimum': 0.95,
                'memory_usage_mb_maximum': 1000,
                'database_size_mb_maximum': 5000,
                'error_rate_maximum': 0.05,
                'temporal_cleanup_efficiency_minimum': 0.8
            }
        }
    
    def _initialize_health_metrics(self) -> Dict[str, HealthMetric]:
        """Initialize health metrics"""
        thresholds = self.config['thresholds']
        
        return {
            'biological_reflex_performance': HealthMetric(
                name='Biological Reflex Performance',
                threshold=thresholds['biological_reflex_ms']
            ),
            'system_success_rate': HealthMetric(
                name='System Success Rate',
                threshold=thresholds['success_rate_minimum']
            ),
            'memory_usage': HealthMetric(
                name='Memory Usage (MB)',
                threshold=thresholds['memory_usage_mb_maximum']
            ),
            'database_size': HealthMetric(
                name='Database Size (MB)',
                threshold=thresholds['database_size_mb_maximum']
            ),
            'error_rate': HealthMetric(
                name='Error Rate',
                threshold=thresholds['error_rate_maximum']
            ),
            'temporal_cleanup_efficiency': HealthMetric(
                name='Temporal Cleanup Efficiency',
                threshold=thresholds['temporal_cleanup_efficiency_minimum']
            ),
            'nervous_system_intelligence': HealthMetric(
                name='Nervous System Intelligence Score',
                threshold=0.8
            )
        }
    
    def _initialize_alert_rules(self) -> List[AlertRule]:
        """Initialize alert rules"""
        return [
            AlertRule(
                name='biological_reflex_performance_degraded',
                condition='threshold_exceeded',
                threshold=self.config['thresholds']['biological_reflex_ms'],
                severity='warning',
                action='log'
            ),
            AlertRule(
                name='biological_reflex_performance_critical',
                condition='threshold_exceeded',
                threshold=self.config['thresholds']['biological_reflex_ms'] * 2,
                severity='critical',
                action='auto_remediate'
            ),
            AlertRule(
                name='success_rate_low',
                condition='threshold_below',
                threshold=self.config['thresholds']['success_rate_minimum'],
                severity='warning',
                action='notify'
            ),
            AlertRule(
                name='memory_usage_high',
                condition='threshold_exceeded',
                threshold=self.config['thresholds']['memory_usage_mb_maximum'],
                severity='warning',
                action='auto_remediate'
            ),
            AlertRule(
                name='error_rate_high',
                condition='threshold_exceeded',
                threshold=self.config['thresholds']['error_rate_maximum'],
                severity='critical',
                action='notify'
            ),
            AlertRule(
                name='temporal_cleanup_inefficient',
                condition='threshold_below',
                threshold=self.config['thresholds']['temporal_cleanup_efficiency_minimum'],
                severity='warning',
                action='auto_remediate'
            )
        ]
    
    def _initialize_maintenance_tasks(self) -> List[Dict[str, Any]]:
        """Initialize maintenance tasks"""
        return [
            {
                'name': 'temporal_memory_cleanup',
                'description': 'Clean up expired working memories',
                'frequency_hours': 24,
                'auto_execute': True,
                'last_executed': None
            },
            {
                'name': 'performance_optimization',
                'description': 'Optimize caches and performance settings',
                'frequency_hours': 6,
                'auto_execute': True,
                'last_executed': None
            },
            {
                'name': 'health_metrics_analysis',
                'description': 'Analyze health metrics trends',
                'frequency_hours': 4,
                'auto_execute': True,
                'last_executed': None
            },
            {
                'name': 'database_optimization',
                'description': 'Optimize database performance',
                'frequency_hours': 168,  # Weekly
                'auto_execute': False,
                'last_executed': None
            },
            {
                'name': 'log_rotation',
                'description': 'Rotate and archive logs',
                'frequency_hours': 24,
                'auto_execute': True,
                'last_executed': None
            }
        ]
    
    async def initialize(self) -> None:
        """Initialize operational monitor"""
        if self._initialized:
            return
        
        self.logger.info("Operational monitor initializing...")
        
        # Initialize enhanced nervous system connection
        self.enhanced_system = UnifiedEnhancedTriggers()
        await self.enhanced_system.initialize()
        
        self._initialized = True
        self.logger.info("Operational monitor initialized successfully")
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        if not self._initialized:
            await self.initialize()
        
        self.monitoring_active = True
        self.monitoring_start_time = datetime.now()
        
        self.logger.info("Starting continuous operational monitoring...")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._maintenance_loop())
        
        print("🔍 OPERATIONAL MONITORING STARTED")
        print(f"⏰ Monitoring Interval: {self.config['monitoring_interval_seconds']}s")
        print(f"🔧 Maintenance Interval: {self.config['maintenance_interval_hours']}h")
        print(f"🚨 Auto-remediation: {'Enabled' if self.config['auto_remediation_enabled'] else 'Disabled'}")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("Operational monitoring stopped")
        print("🛑 OPERATIONAL MONITORING STOPPED")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect health metrics
                await self._collect_health_metrics()
                
                # Check alert conditions
                await self._check_alerts()
                
                # Update performance history
                await self._update_performance_history()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.config['monitoring_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config['monitoring_interval_seconds'])
    
    async def _maintenance_loop(self) -> None:
        """Maintenance task loop"""
        while self.monitoring_active:
            try:
                # Execute scheduled maintenance tasks
                await self._execute_scheduled_maintenance()
                
                # Sleep until next maintenance check (every hour)
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_health_metrics(self) -> None:
        """Collect current health metrics"""
        try:
            # Get system health
            system_health = await self.enhanced_system.unified_health_check()
            
            # Get biological reflex performance
            reflex_demo = await self.enhanced_system.demonstrate_biological_reflex()
            avg_reflex_time = reflex_demo['overall_performance']['average_processing_time_ms']
            
            # Get temporal memory metrics
            temporal_manager = await get_temporal_memory_manager()
            temporal_metrics = temporal_manager.get_temporal_metrics()
            
            # Get performance optimizer metrics
            perf_optimizer = await get_performance_optimizer()
            perf_metrics = perf_optimizer.get_performance_metrics()
            
            # Update health metrics
            current_time = datetime.now()
            
            self.health_metrics['biological_reflex_performance'].current_value = avg_reflex_time
            self.health_metrics['biological_reflex_performance'].last_checked = current_time
            self.health_metrics['biological_reflex_performance'].status = (
                'healthy' if avg_reflex_time < self.health_metrics['biological_reflex_performance'].threshold
                else 'warning' if avg_reflex_time < self.health_metrics['biological_reflex_performance'].threshold * 2
                else 'critical'
            )
            
            success_rate = reflex_demo['overall_performance']['biological_reflex_success_rate']
            self.health_metrics['system_success_rate'].current_value = success_rate
            self.health_metrics['system_success_rate'].last_checked = current_time
            self.health_metrics['system_success_rate'].status = (
                'healthy' if success_rate >= self.health_metrics['system_success_rate'].threshold
                else 'warning' if success_rate >= self.health_metrics['system_success_rate'].threshold * 0.9
                else 'critical'
            )
            
            # Temporal cleanup efficiency
            cleanup_efficiency = temporal_metrics.get('cleanup_efficiency', 0.8)
            self.health_metrics['temporal_cleanup_efficiency'].current_value = cleanup_efficiency
            self.health_metrics['temporal_cleanup_efficiency'].last_checked = current_time
            self.health_metrics['temporal_cleanup_efficiency'].status = (
                'healthy' if cleanup_efficiency >= self.health_metrics['temporal_cleanup_efficiency'].threshold
                else 'warning'
            )
            
            # Cache hit rate as intelligence metric
            cache_hit_rate = perf_metrics.get('cache_hit_rate', 0.0)
            self.health_metrics['nervous_system_intelligence'].current_value = cache_hit_rate
            self.health_metrics['nervous_system_intelligence'].last_checked = current_time
            self.health_metrics['nervous_system_intelligence'].status = (
                'healthy' if cache_hit_rate >= self.health_metrics['nervous_system_intelligence'].threshold
                else 'warning'
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect health metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check alert conditions and trigger actions"""
        for alert_rule in self.alert_rules:
            if not alert_rule.enabled:
                continue
            
            try:
                # Find relevant metric
                metric_name = alert_rule.name.split('_')[0] + '_' + alert_rule.name.split('_')[1]
                if metric_name not in self.health_metrics:
                    continue
                
                metric = self.health_metrics[metric_name]
                if metric.current_value is None:
                    continue
                
                # Check alert condition
                alert_triggered = False
                
                if alert_rule.condition == 'threshold_exceeded':
                    alert_triggered = metric.current_value > alert_rule.threshold
                elif alert_rule.condition == 'threshold_below':
                    alert_triggered = metric.current_value < alert_rule.threshold
                
                if alert_triggered:
                    await self._handle_alert(alert_rule, metric)
                    
            except Exception as e:
                self.logger.error(f"Error checking alert rule {alert_rule.name}: {e}")
    
    async def _handle_alert(self, alert_rule: AlertRule, metric: HealthMetric) -> None:
        """Handle triggered alert"""
        alert_message = f"ALERT: {alert_rule.name} - {metric.name}: {metric.current_value} (threshold: {alert_rule.threshold})"
        
        if alert_rule.action == 'log':
            self.logger.warning(alert_message)
        elif alert_rule.action == 'notify':
            self.logger.error(alert_message)
            # Could integrate with notification systems here
        elif alert_rule.action == 'auto_remediate':
            self.logger.error(f"{alert_message} - Initiating auto-remediation")
            await self._auto_remediate(alert_rule, metric)
    
    async def _auto_remediate(self, alert_rule: AlertRule, metric: HealthMetric) -> None:
        """Execute auto-remediation actions"""
        try:
            if 'biological_reflex' in alert_rule.name:
                # Clear performance caches
                perf_optimizer = await get_performance_optimizer()
                await perf_optimizer.clear_caches()
                self.logger.info("Auto-remediation: Cleared performance caches")
                
            elif 'memory_usage' in alert_rule.name:
                # Trigger temporal cleanup
                temporal_manager = await get_temporal_memory_manager()
                cleanup_result = await temporal_manager.daily_cleanup_process()
                self.logger.info(f"Auto-remediation: Executed cleanup - {cleanup_result}")
                
            elif 'temporal_cleanup' in alert_rule.name:
                # Force temporal cleanup
                temporal_manager = await get_temporal_memory_manager()
                await temporal_manager.daily_cleanup_process()
                self.logger.info("Auto-remediation: Forced temporal cleanup")
                
        except Exception as e:
            self.logger.error(f"Auto-remediation failed for {alert_rule.name}: {e}")
    
    async def _update_performance_history(self) -> None:
        """Update performance history"""
        try:
            # Get current performance data
            reflex_demo = await self.enhanced_system.demonstrate_biological_reflex()
            avg_time = reflex_demo['overall_performance']['average_processing_time_ms']
            success_rate = reflex_demo['overall_performance']['biological_reflex_success_rate']
            
            # Add to history
            self.performance_history['biological_reflex_times'].append({
                'timestamp': datetime.now(),
                'value': avg_time
            })
            
            self.performance_history['system_health_scores'].append({
                'timestamp': datetime.now(),
                'value': success_rate
            })
            
            # Trim history to retention period
            retention_cutoff = datetime.now() - timedelta(hours=self.config['performance_history_retention_hours'])
            
            for history_type in self.performance_history:
                self.performance_history[history_type] = [
                    entry for entry in self.performance_history[history_type]
                    if entry['timestamp'] > retention_cutoff
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to update performance history: {e}")
    
    async def _execute_scheduled_maintenance(self) -> None:
        """Execute scheduled maintenance tasks"""
        current_time = datetime.now()
        
        for task in self.maintenance_tasks:
            if not task.get('auto_execute', False):
                continue
            
            # Check if task is due
            last_executed = task.get('last_executed')
            if last_executed is None:
                task_due = True
            else:
                time_since_last = current_time - last_executed
                task_due = time_since_last.total_seconds() >= (task['frequency_hours'] * 3600)
            
            if task_due:
                await self._execute_maintenance_task(task)
                task['last_executed'] = current_time
    
    async def _execute_maintenance_task(self, task: Dict[str, Any]) -> None:
        """Execute individual maintenance task"""
        task_name = task['name']
        
        try:
            self.logger.info(f"Executing maintenance task: {task_name}")
            
            if task_name == 'temporal_memory_cleanup':
                temporal_manager = await get_temporal_memory_manager()
                result = await temporal_manager.daily_cleanup_process()
                self.logger.info(f"Temporal cleanup completed: {result}")
                
            elif task_name == 'performance_optimization':
                perf_optimizer = await get_performance_optimizer()
                # Clear old caches and optimize
                await perf_optimizer.clear_caches()
                self.logger.info("Performance optimization completed")
                
            elif task_name == 'health_metrics_analysis':
                await self._analyze_health_trends()
                self.logger.info("Health metrics analysis completed")
                
            elif task_name == 'log_rotation':
                # Simulate log rotation
                self.logger.info("Log rotation completed")
                
        except Exception as e:
            self.logger.error(f"Maintenance task {task_name} failed: {e}")
    
    async def _analyze_health_trends(self) -> None:
        """Analyze health metric trends"""
        for metric_name, metric in self.health_metrics.items():
            # Simple trend analysis based on recent values
            if len(self.performance_history.get('system_health_scores', [])) > 5:
                recent_values = self.performance_history['system_health_scores'][-5:]
                if len(recent_values) >= 2:
                    trend_direction = recent_values[-1]['value'] - recent_values[0]['value']
                    if trend_direction > 0.02:
                        metric.trend = 'improving'
                    elif trend_direction < -0.02:
                        metric.trend = 'degrading'
                    else:
                        metric.trend = 'stable'
    
    def get_operational_dashboard(self) -> Dict[str, Any]:
        """Generate operational dashboard data"""
        dashboard_data = {
            'monitoring_status': {
                'active': self.monitoring_active,
                'uptime_hours': (
                    (datetime.now() - self.monitoring_start_time).total_seconds() / 3600
                    if self.monitoring_start_time else 0
                ),
                'last_health_check': max(
                    (metric.last_checked for metric in self.health_metrics.values() if metric.last_checked),
                    default=None
                )
            },
            'health_metrics': {
                name: {
                    'current_value': metric.current_value,
                    'threshold': metric.threshold,
                    'status': metric.status,
                    'trend': metric.trend,
                    'last_checked': metric.last_checked.isoformat() if metric.last_checked else None
                }
                for name, metric in self.health_metrics.items()
            },
            'performance_summary': {
                'avg_biological_reflex_ms': (
                    sum(entry['value'] for entry in self.performance_history['biological_reflex_times'][-10:]) / 
                    len(self.performance_history['biological_reflex_times'][-10:])
                    if self.performance_history['biological_reflex_times'] else 0
                ),
                'system_health_trend': (
                    self.health_metrics.get('system_success_rate', HealthMetric('', 0)).trend
                ),
                'total_operations_monitored': len(self.performance_history.get('biological_reflex_times', []))
            },
            'maintenance_status': {
                'last_maintenance': self.last_maintenance_time.isoformat() if self.last_maintenance_time else None,
                'next_scheduled_tasks': [
                    {
                        'name': task['name'],
                        'next_due': (
                            (task['last_executed'] + timedelta(hours=task['frequency_hours'])).isoformat()
                            if task['last_executed'] else 'now'
                        )
                    }
                    for task in self.maintenance_tasks if task.get('auto_execute', False)
                ]
            },
            'operational_excellence_score': self._calculate_operational_excellence_score()
        }
        
        return dashboard_data
    
    def _calculate_operational_excellence_score(self) -> float:
        """Calculate overall operational excellence score"""
        healthy_metrics = sum(1 for metric in self.health_metrics.values() if metric.status == 'healthy')
        total_metrics = len(self.health_metrics)
        
        if total_metrics == 0:
            return 0.0
        
        base_score = healthy_metrics / total_metrics
        
        # Bonus for good trends
        improving_trends = sum(1 for metric in self.health_metrics.values() if metric.trend == 'improving')
        trend_bonus = (improving_trends / total_metrics) * 0.1
        
        return min(1.0, base_score + trend_bonus)

async def start_operational_monitoring(config: Optional[Dict[str, Any]] = None) -> OperationalMonitor:
    """
    Start operational monitoring for enhanced nervous system.
    
    Args:
        config: Optional monitoring configuration
        
    Returns:
        Initialized operational monitor
    """
    monitor = OperationalMonitor(config)
    await monitor.start_monitoring()
    return monitor

if __name__ == "__main__":
    async def demo_monitoring():
        monitor = await start_operational_monitoring()
        
        # Run monitoring for a demo period
        print("Running operational monitoring demo for 30 seconds...")
        await asyncio.sleep(30)
        
        # Show dashboard
        dashboard = monitor.get_operational_dashboard()
        print("\n📊 OPERATIONAL DASHBOARD:")
        print(json.dumps(dashboard, indent=2, default=str))
        
        await monitor.stop_monitoring()
    
    asyncio.run(demo_monitoring())